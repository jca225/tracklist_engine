"""Manifest-free structural gate: GT yaml must derive from its committed `.als`.

The BB ground-truth yaml fixtures (``labeling/fixtures/*_ground_truth.yaml``) are
hand-editable snapshots of the Ableton ``.als`` labeling session — the true source
of truth in ``~/aligning/``. Nothing stopped a committed yaml from silently
drifting away from the ``.als`` it was exported from: a manual ``slot_label`` remap
(commit ``518065f``) hand-edited BB11 slot 013 "Kill FM - Fresh" from ``013w1`` to
``013w3`` while the ``.als`` still says ``013w1``. No test caught it.

This gate re-derives the slot labels straight from a **committed** ``.als`` and
compares them to the committed yaml. It is **manifest-free on purpose**: a full
re-export (``export_als_to_gt.collect_kept_clip_rows``) hard-requires the set's
pull-inventory manifest for ``mix_duration_s`` + ``set_id``, which is absent in
CI (no ``~/aligning/``). But ``slot_from_path(clip.path)`` is deterministic from the
``.als`` clip path alone, and the drift that bit us (a ``slot_label`` *value*
change) is exactly a slot-label vocabulary change — so a slot-label structural
check needs no manifest.

**Invariant.** The *set* of ``slot_label`` values in the yaml equals the *set* of
``slot_from_path`` values over the ``.als``'s non-silent, non-mix clips.

We compare **sets, not multisets**. The export's hygiene pass (loop/sliver merge,
mix-span split, micro-sliver drop — see ``export_als_to_gt``) collapses many raw
clips carrying one slot into a single yaml row, and that collapse depends on
manifest-driven warp timing this gate deliberately does not reproduce. So the
*multiset* legitimately differs between raw ``.als`` and yaml (empirically: raw BB11
has 272 non-silent slot occurrences over 141 distinct slots; the yaml has 148 rows
over the same 141 distinct slots). The *set* matches exactly. A value drift
(``013w3`` present in the yaml but absent from the ``.als``; ``013w1`` present in the
``.als`` but absent from the yaml) surfaces cleanly in the symmetric difference
either way — which is exactly the ``013w3`` bug this catches.

Empty / mix-placeholder clips (a dragged-in ``mix.m4a`` / ``instrumental-N.flac``
whose path carries no ``NNN__`` slot token) yield no slot label on either side and
are excluded from the comparison — they carry no slot identity to verify.

**What this gate does NOT catch** (bounded blind spots of a *set* check — a drift
is caught iff it changes the distinct-label *vocabulary*, which the 518065f-style
"remap a label to a value not in the .als" edit does, so the class that bit us is
covered; these do not):

- **Label↔row association** — swapping ``slot_label`` between two yaml rows (right
  vocabulary, wrong track/timing). Missed.
- **Multiplicity** — duplicating a row, or deleting one of a slot's duplicate rows.
  The yaml has 10 rows on duplicated labels (029×3, 029w2×3, 036×2, 037×2); edits
  among those keep the set invariant. Missed.
- **Relabel *within* the existing vocabulary** — retagging a duplicated-label row
  to another label already present. Missed. (A relabel of a *uniquely*-labelled row
  — 131 of 141 labels — IS caught, because it removes that label from the set.)
- **Non-label fields** — ``set_start_s``/``set_end_s`` timing, ``ref_*``,
  ``claimed_stem``, ``track_id``, ``track`` name. All invisible here (timings need
  the manifest/warp mapper this gate omits). Missed.

Also one **fail-safe false-positive** to expect first: ``collect_kept_clip_rows``
additionally drops clips for *manifest-driven* reasons (outside mix-warp span,
parking, non-positive span, micro-sliver) that this gate cannot see, so a future
legitimate re-export where a slot survives *only* via such a clip would trip the
gate on a correct yaml — forcing human reconciliation rather than passing silently.
"""

from __future__ import annotations

import sys
from pathlib import Path

from core.result import Err, Ok
from labeling.als import load_als_xml, parse_layer_clips
from labeling.als.identity import slot_from_path
from labeling.schema import load

_REPO = Path(__file__).resolve().parent.parent.parent

# Registry of (set_id -> committed .als) for gated sets. The gate runs over every
# entry here (guardrails + pre-commit). Only sets whose canonical `.als` is
# checked into the repo belong here.
# BB12's canonical .als is `bb12_align.als` (== export_als_to_gt.DEFAULT_ALS),
# identified empirically: its export matches the committed bb12 slot set exactly
# (the project moved to ~/aligning/_labeling/1fsnxchk/ while the set-dir stayed
# put — master-plan §4 / D22). Committed at labeling/fixtures/als/1fsnxchk.als.
GATED_SETS: dict[str, Path] = {
    "2nvzlh2k": _REPO / "labeling" / "fixtures" / "als" / "2nvzlh2k.als",  # BB11
    "1fsnxchk": _REPO / "labeling" / "fixtures" / "als" / "1fsnxchk.als",  # BB12
}

# Committed GT yaml fixture per gated set (companion to GATED_SETS). Same keys.
GATED_GT_YAMLS: dict[str, Path] = {
    "2nvzlh2k": _REPO / "labeling" / "fixtures" / "bb11_ground_truth.yaml",  # BB11
    "1fsnxchk": _REPO / "labeling" / "fixtures" / "bb12_ground_truth.yaml",  # BB12
}


def iter_gated() -> list[tuple[str, Path, Path]]:
    """``(set_id, yaml_path, als_path)`` for every gated set."""
    return [(sid, GATED_GT_YAMLS[sid], GATED_SETS[sid]) for sid in GATED_SETS]


def slot_labels_from_als(als_path: Path) -> set[str]:
    """Distinct slot labels of the ``.als``'s non-silent, non-mix clips.

    Mirrors the clip-drop rules of ``collect_kept_clip_rows``: ``parse_layer_clips``
    already excludes the ``1-mix``/``2-mix`` tracks, and we skip any clip Live keeps
    but marks silent (``clip.silence_reason`` — deactivated track / disabled clip /
    fader at 0). Empty slots (paths with no ``NNN__`` token) are dropped.
    """
    root = load_als_xml(als_path)
    out: set[str] = set()
    for clip in parse_layer_clips(root):
        if clip.silence_reason:
            continue
        slot = slot_from_path(clip.path)
        if slot:
            out.add(slot)
    return out


def slot_labels_from_yaml(yaml_path: Path) -> set[str]:
    """Distinct non-empty ``slot_label`` values in a committed GT yaml."""
    match load(yaml_path):
        case Ok(gt):
            return {t.slot_label for t in gt.tracks if t.slot_label}
        case Err(e):
            raise ValueError(f"cannot load {yaml_path}: {e.detail}")


def check_yaml_matches_als(yaml_path: Path, als_path: Path) -> tuple[bool, str]:
    """``(ok, reason)`` — does the committed yaml's slot set match the ``.als``?

    ``ok`` iff the two slot-label sets are equal. On mismatch ``reason`` lists the
    symmetric difference: slots in the yaml but not the ``.als`` (drifted / hand-
    invented labels) and slots in the ``.als`` but not the yaml (dropped / renamed).
    """
    yaml_path = Path(yaml_path)
    als_path = Path(als_path)
    if not als_path.is_file():
        return False, f"committed .als not found: {als_path}"
    if not yaml_path.is_file():
        return False, f"GT yaml not found: {yaml_path}"

    als_slots = slot_labels_from_als(als_path)
    yaml_slots = slot_labels_from_yaml(yaml_path)
    if als_slots == yaml_slots:
        return True, f"ok: {len(yaml_slots)} slot labels match {als_path.name}"

    yaml_only = sorted(yaml_slots - als_slots)
    als_only = sorted(als_slots - yaml_slots)
    parts = [
        f"GT yaml slot labels drifted from {als_path.name}:",
        f"  in yaml but NOT in .als ({len(yaml_only)}): {yaml_only or '-'}",
        f"  in .als but NOT in yaml ({len(als_only)}): {als_only or '-'}",
    ]
    return False, "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    """Verify every gated GT yaml against its committed ``.als``.

    Exit 0 if all match (or a set's committed ``.als`` is absent — the pre-commit
    guard), 1 on the first drift. Same invariant `make check` enforces; kept as a
    standalone entrypoint so the pre-commit hook can surface a drift directly.
    """
    del argv
    bad = 0
    for set_id, yaml_path, als_path in iter_gated():
        if not als_path.is_file():
            print(f"gt-als-gate: skip {set_id} (no committed {als_path.name})")
            continue
        ok, reason = check_yaml_matches_als(yaml_path, als_path)
        if ok:
            print(f"gt-als-gate: {set_id} OK — {reason}")
        else:
            bad += 1
            print(f"gt-als-gate: {set_id} FAIL\n{reason}", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
