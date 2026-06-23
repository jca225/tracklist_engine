#!/usr/bin/env python
"""Backfill Acquisition Case Records for Big Bootie 12 from existing artifacts.

BB12 is the gold corpus: 166 hand-resolved GT rows whose outcomes we already
know. This reconstructs one :class:`AcquisitionCase` per ``(slot_label,
recording_id)`` by joining:

  * ``labeling/fixtures/bb12_ground_truth.yaml`` — the *outcome* (ref_source,
    claimed_stem, loops) per played layer.
  * ``labeling/fixtures/bb12_path_audit.json`` — the *source file* actually used
    (``als_path``) and join health (``status`` / ``reasons``).

The two files are positionally aligned (validated on label, not timing — the
audit predates the ~430 s mix-timebase fix). Multiple GT rows for one
``(slot, recording)`` collapse into a single case: re-entries merge, and a
``demucs`` + ``online_candidate`` pair becomes two attempts plus a preference
pair (the online studio stem beats the separated one).

This produces the replay corpus the harness is later scored against. It is
*reconstruction*, not authority — GT ``ref_source`` and
``track_audio_correction`` remain the source of truth.

Run from repo root::

    venvs/audio/bin/python -m scripts.backfill_bb12_cases
    venvs/audio/bin/python -m scripts.backfill_bb12_cases --print 5
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from core.acquisition_case import (
    AcquisitionCase,
    Actor,
    Attempt,
    AttemptAction,
    AttemptVerdict,
    CaseClaim,
    CaseStatus,
    ProblemClass,
    Resolution,
    TrainingSignal,
    default_path,
    save_cases,
)
from core.identity import normalize_stem, normalize_version
from core.slot_inventory import derive_layer_role

SET_ID = "1fsnxchk"
DEFAULT_GT = Path("labeling/fixtures/bb12_ground_truth.yaml")
DEFAULT_AUDIT = Path("labeling/fixtures/bb12_path_audit.json")

# How good a source is, for picking the case's resolution among siblings.
_SOURCE_RANK = {"online_candidate": 3, "reference": 2, "demucs": 1, None: 0}

_ACTION_FOR_SOURCE = {
    "online_candidate": AttemptAction.YT_SEARCH,
    "demucs": AttemptAction.SEPARATE,
    "reference": AttemptAction.INGEST_URL,
    "mix": AttemptAction.MIX_EXTRACT,
    None: AttemptAction.MANUAL,
}

_VERSION_RE = re.compile(
    r"\((?:[^()]*\s)?(remix|rework|bootleg|mashup|edit|vip)\)", re.IGNORECASE
)


# ── parsing helpers ───────────────────────────────────────────────────────────


def _primary_slot(label: str | None) -> str | None:
    if not label:
        return None
    m = re.match(r"^(\d+)", label)
    return m.group(1) if m else None


def _infer_version(track_name: str) -> str:
    m = _VERSION_RE.search(track_name or "")
    return normalize_version(m.group(1)) if m else "original"


def _group_key(gt_row: dict[str, Any]) -> tuple[str, str]:
    """One case per (slot, recording); fall back to track name when null."""
    slot = gt_row.get("slot_label") or f"@{gt_row.get('track')}"
    rec = gt_row.get("track_id") or f"mix:{gt_row.get('track')}"
    return (str(slot), str(rec))


def _audit_source(als_path: str | None) -> str | None:
    if not als_path:
        return None
    if "/candidates/" in als_path:
        return "online_candidate"
    if "/stems/" in als_path:
        return "demucs"
    if "/tracks/" in als_path:
        return "reference"
    return None


# ── case construction ─────────────────────────────────────────────────────────


def _build_case(
    key: tuple[str, str],
    members: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    concurrency: dict[str | None, int],
) -> AcquisitionCase:
    """Fold all GT/audit rows sharing one (slot, recording) into one case."""
    slot_label, recording_id = key
    # Coalesce unmatched audit rows (the mix-extract cases) to empty dicts.
    members = [(g, a or {}) for g, a in members]
    gt0 = members[0][0]
    # Prefer a canonical "Artist - Title" member name over a raw candidate filename.
    names = [str(g.get("track") or "").strip() for g, _ in members]
    track_name = next((n for n in names if " - " in n), names[0])

    # Faithful slot label (drop the @name fallback prefix from the key).
    real_slot = gt0.get("slot_label") or ""
    is_concurrent = (concurrency.get(_primary_slot(real_slot), 0) > 1) or any(
        "w" in (g.get("slot_label") or "") for g, _ in members
    )
    claimed_stem = normalize_stem(gt0.get("claimed_stem"))
    layer_role = derive_layer_role(
        real_slot, is_concurrent=is_concurrent, claimed_stem=gt0.get("claimed_stem")
    )

    unalignable = any(g.get("unalignable") for g, _ in members)
    source_notes = [str(g["source_note"]) for g, _ in members if g.get("source_note")]
    needs_ears = any(a.get("status") == "needs_ears" for _, a in members)
    audit_reasons = sorted({r for _, a in members for r in (a.get("reasons") or [])})
    has_loop = any(g.get("is_loop") or g.get("ref_segments") for g, _ in members)

    # One attempt per distinct (ref_source, source_path) actually present.
    attempts: list[Attempt] = []
    seen: set[tuple[str | None, str | None]] = set()
    src_to_path: dict[str | None, str | None] = {}
    for g, a in members:
        src = g.get("ref_source") or _audit_source(a.get("als_path"))
        path = a.get("als_path")
        if (src, path) in seen:
            continue
        seen.add((src, path))
        src_to_path.setdefault(src, path)

    # Decide the winning source (highest rank present).
    present = [
        g.get("ref_source") or _audit_source(a.get("als_path")) for g, a in members
    ]
    winner = max(present, key=lambda s: _SOURCE_RANK.get(s, 0)) if present else None

    for src in sorted(src_to_path, key=lambda s: -_SOURCE_RANK.get(s, 0)):
        path = src_to_path[src]
        is_winner = src == winner
        attempts.append(
            Attempt(
                action=_ACTION_FOR_SOURCE.get(src, AttemptAction.MANUAL),
                actor=Actor.BACKFILL,
                url=path,
                platform=src,
                verdict=AttemptVerdict.PROMOTE if is_winner else AttemptVerdict.REJECT,
                reject_reason=(
                    None
                    if is_winner
                    else f"superseded by {winner} (better source for this stem)"
                ),
                notes="reconstructed from GT + path audit",
            )
        )

    # Problem classes (multi-tag, conservative).
    problems: list[ProblemClass] = []
    sources = set(present)
    if "demucs" in sources:
        problems.append(ProblemClass.SUBOPTIMAL_STEM)
    if "online_candidate" in sources:
        problems.append(ProblemClass.MISSING_ASSET)  # default inventory lacked the stem
    if claimed_stem != "regular" and sources <= {"demucs", None}:
        problems.append(ProblemClass.WRONG_STEM)
    if "unresolved_manifest" in audit_reasons:
        problems.append(ProblemClass.UNRESOLVED_MANIFEST)
    if has_loop:
        problems.append(ProblemClass.STRUCTURE)
    if unalignable:
        problems += [ProblemClass.PHANTOM_TRACK, ProblemClass.MISSING_ASSET]
    # de-dup, order-preserving
    problems = list(dict.fromkeys(problems))

    # Status.
    if unalignable:
        status = CaseStatus.UNRESOLVABLE
    elif needs_ears:
        status = CaseStatus.HUMAN_REVIEW
    else:
        status = CaseStatus.RESOLVED

    # Training signal: online studio stem beats demucs for the same recording.
    training = TrainingSignal()
    if "online_candidate" in src_to_path and "demucs" in src_to_path:
        win_path = src_to_path["online_candidate"]
        lose_path = src_to_path["demucs"]
        if win_path and lose_path:
            training = TrainingSignal(
                negatives=(lose_path,),
                preference_pairs=((win_path, lose_path),),
            )

    resolution = Resolution(
        track_audio_id=None,  # GT carries recording_id, not track_audio_id
        ref_source=winner,
        source_path=src_to_path.get(winner),
        gt_confirmed=status == CaseStatus.RESOLVED,
    )

    return AcquisitionCase(
        set_id=SET_ID,
        slot_label=real_slot or track_name,
        layer_role=layer_role,
        claim=CaseClaim(
            recording_id=recording_id,
            display_name=track_name,
            version=_infer_version(track_name),
            stem=claimed_stem,
            variant="regular",
        ),
        status=status,
        problem_classes=tuple(problems),
        attempts=tuple(attempts),
        resolution=resolution,
        training=training,
        notes=" | ".join(source_notes),
    )


def _norm(s: Any) -> str:
    """Normalize a label for joining: unify dashes, collapse space, lowercase."""
    t = str(s or "").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", t).strip().lower()


def _join_rows(
    gt: list[dict[str, Any]], audit: list[dict[str, Any]]
) -> tuple[list[tuple[dict[str, Any], dict[str, Any] | None]], int]:
    """Pair each GT row with its audit row by (slot, label), FIFO, not position.

    The audit file is in a different order than GT (play-order vs slot-order) and
    omits ``track_id``, so a positional ``zip`` silently mismatches the tail.
    We match on normalized ``(slot, label)`` first, fall back to label-only, and
    never reuse an audit row. Returns the pairings (in GT order) and the count of
    GT rows that found no audit row.
    """
    by_slot_label: dict[tuple[str, str], collections.deque[int]] = (
        collections.defaultdict(collections.deque)
    )
    by_label: dict[str, collections.deque[int]] = collections.defaultdict(
        collections.deque
    )
    for idx, a in enumerate(audit):
        by_slot_label[(_norm(a.get("gt_slot")), _norm(a.get("gt_label")))].append(idx)
        by_label[_norm(a.get("gt_label"))].append(idx)

    used: set[int] = set()

    def _take(dq: collections.deque[int] | None) -> int | None:
        while dq:
            idx = dq.popleft()
            if idx not in used:
                used.add(idx)
                return idx
        return None

    paired: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    unmatched = 0
    for g in gt:
        idx = _take(
            by_slot_label.get((_norm(g.get("slot_label")), _norm(g.get("track"))))
        )
        if idx is None:
            idx = _take(by_label.get(_norm(g.get("track"))))
        if idx is None:
            unmatched += 1
            paired.append((g, None))
        else:
            paired.append((g, audit[idx]))
    return paired, unmatched


def build_cases(gt_path: Path, audit_path: Path) -> list[AcquisitionCase]:
    gt = yaml.safe_load(gt_path.read_text(encoding="utf-8"))["tracks"]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))["rows"]

    paired, unmatched = _join_rows(gt, audit)
    if unmatched:
        # Expected: the 2 mix-extract rows (mix / mix_instrumental) have no audit
        # row. Anything beyond a handful means the join key drifted — surface it.
        print(f"note: {unmatched} GT row(s) had no audit match (built from GT only)")

    concurrency = collections.Counter(
        _primary_slot(g.get("slot_label")) for g in gt if g.get("slot_label")
    )

    groups: dict[tuple[str, str], list[tuple[dict, dict | None]]] = (
        collections.OrderedDict()
    )
    for g, a in paired:
        groups.setdefault(_group_key(g), []).append((g, a))

    return [
        _build_case(k, members, concurrency=concurrency)
        for k, members in groups.items()
    ]


# ── summary / CLI ─────────────────────────────────────────────────────────────


def _summarize(cases: list[AcquisitionCase], n_gt_rows: int) -> str:
    by_status = collections.Counter(c.status.value for c in cases)
    by_role = collections.Counter(c.layer_role for c in cases)
    by_winner = collections.Counter(
        (c.resolution.ref_source if c.resolution else None) for c in cases
    )
    by_problem = collections.Counter(p.value for c in cases for p in c.problem_classes)
    with_pref = sum(1 for c in cases if c.training.preference_pairs)
    lines = [
        f"GT rows: {n_gt_rows}  ->  cases: {len(cases)} "
        f"(collapsed {n_gt_rows - len(cases)} re-entry/multi-source rows)",
        f"  status:   {dict(by_status)}",
        f"  role:     {dict(by_role)}",
        f"  winner:   {dict(by_winner)}",
        f"  problems: {dict(by_problem)}",
        f"  preference pairs (online>demucs): {with_pref}",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", type=Path, default=DEFAULT_GT)
    ap.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    ap.add_argument("--out", type=Path, default=None, help="output .jsonl path")
    ap.add_argument(
        "--print", type=int, default=0, metavar="N", help="print N sample cases"
    )
    ap.add_argument("--dry-run", action="store_true", help="do not write the file")
    args = ap.parse_args()

    if not args.gt.exists():
        sys.exit(f"missing GT fixture: {args.gt}")
    if not args.audit.exists():
        sys.exit(f"missing path audit: {args.audit}")

    gt_rows = yaml.safe_load(args.gt.read_text(encoding="utf-8"))["tracks"]
    cases = build_cases(args.gt, args.audit)
    print(_summarize(cases, len(gt_rows)))

    if args.print:
        from core.acquisition_case import case_to_dict

        print("\n--- sample cases ---")
        for c in cases[: args.print]:
            print(json.dumps(case_to_dict(c), ensure_ascii=False, indent=2))

    out = args.out or default_path(SET_ID)
    if args.dry_run:
        print(f"\n[dry-run] would write {len(cases)} cases -> {out}")
        return
    save_cases(out, cases)
    print(f"\nwrote {len(cases)} cases -> {out}")


if __name__ == "__main__":
    main()
