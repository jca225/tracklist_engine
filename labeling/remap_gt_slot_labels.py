"""Remap GT ``slot_label`` values to canonical pi ``set_track_slots`` labels.

GT rows keyed by Ableton clip numbers often drift from scrape slot labels after
inventory repair. This pass remaps by ``track_id`` (+ ``claimed_stem`` when one
id appears on multiple pi rows).

Usage::

    venvs/audio/bin/python -m labeling.remap_gt_slot_labels \\
        --yaml labeling/fixtures/bb12_ground_truth.yaml \\
        --set-id 1fsnxchk --dry-run
"""

from __future__ import annotations

import argparse
import copy
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from core.identity import normalize_stem  # noqa: E402
from core.result import Err, Ok  # noqa: E402
from core.ssh_sqlite import ssh_sqlite  # noqa: E402
from labeling.schema import load, replace, save  # noqa: E402


def fetch_remap_slot_rows(set_id: str, ssh_sqlite) -> list[dict[str, Any]]:
    """Load pi slots for remap, including scrape cue times for disambiguation."""
    try:
        rows = ssh_sqlite(f"""
            SELECT set_id, row_index, slot_label,
                   COALESCE(recording_id, track_id) AS recording_id,
                   track_id, claimed_stem,
                   cue_time_seconds, cue_seconds
            FROM set_track_slots
            WHERE set_id = '{set_id}'
            ORDER BY row_index;
        """)
    except Exception:
        rows = ssh_sqlite(f"""
            SELECT set_id, row_index, slot_label,
                   COALESCE(recording_id, track_id) AS recording_id,
                   track_id, claimed_stem, cue_seconds
            FROM set_track_slots
            WHERE set_id = '{set_id}'
            ORDER BY row_index;
        """)
    return [r for r in rows if r.get("slot_label") and r.get("recording_id")]


@dataclass(frozen=True)
class RemapEvent:
    track: str
    track_id: str
    old_slot_label: str
    new_slot_label: str
    status: str  # remapped | needs_human
    detail: str = ""


def _recording_id(row: dict[str, Any]) -> str:
    return str(row.get("recording_id") or row.get("track_id") or "").strip()


def _slot_label(row: dict[str, Any]) -> str:
    return str(row.get("slot_label") or "").strip()


def _claimed_stem(row: dict[str, Any]) -> str:
    return normalize_stem(str(row.get("claimed_stem") or "regular"))


def _cue_time_s(row: dict[str, Any]) -> float | None:
    for key in ("cue_time_seconds", "cue_seconds"):
        val = row.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


def _index_pi_slots(pi_slots: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_tid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pi_slots:
        tid = _recording_id(row)
        label = _slot_label(row)
        if not tid or not label:
            continue
        by_tid[tid].append(row)
    return dict(by_tid)


def _pick_pi_slot(
    gt_row: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    gt_index: int,
) -> dict[str, Any] | None:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    stem = _claimed_stem(gt_row)
    stem_matches = [c for c in candidates if _claimed_stem(c) == stem]
    if len(stem_matches) == 1:
        return stem_matches[0]
    pool = stem_matches if stem_matches else candidates

    if len(pool) == 1:
        return pool[0]

    set_start = gt_row.get("set_start_s")
    if set_start is not None:
        try:
            set_start_f = float(set_start)
        except (TypeError, ValueError):
            set_start_f = None
    else:
        set_start_f = None

    if set_start_f is not None:
        with_cue = [(c, _cue_time_s(c)) for c in pool]
        timed = [(c, abs(set_start_f - cue)) for c, cue in with_cue if cue is not None]
        if timed:
            best_delta = min(delta for _, delta in timed)
            closest = [c for c, delta in timed if delta == best_delta]
            if len(closest) == 1:
                return closest[0]
            pool = closest

    if len(pool) == 1:
        return pool[0]

    # Scrape-order tie-break: closest row_index to GT position in the yaml.
    def row_index(row: dict[str, Any]) -> int:
        try:
            return int(row.get("row_index", 10**9))
        except (TypeError, ValueError):
            return 10**9

    best_idx_delta = min(abs(row_index(c) - gt_index) for c in pool)
    by_order = [c for c in pool if abs(row_index(c) - gt_index) == best_idx_delta]
    if len(by_order) == 1:
        return by_order[0]

    return None


def remap_slots(
    gt_doc: dict[str, Any],
    pi_slots: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[RemapEvent]]:
    """Return a copy of ``gt_doc`` with remapped slot labels and event log."""
    out = copy.deepcopy(gt_doc)
    events: list[RemapEvent] = []
    by_tid = _index_pi_slots(pi_slots)

    for idx, row in enumerate(out.get("tracks") or []):
        if not isinstance(row, dict):
            continue
        old_label = _slot_label(row)
        if old_label == "mix":
            continue

        tid = str(row.get("track_id") or "").strip()
        track_name = str(row.get("track") or row.get("label") or tid or "?")
        if not tid:
            events.append(
                RemapEvent(
                    track=track_name,
                    track_id="",
                    old_slot_label=old_label,
                    new_slot_label=old_label,
                    status="needs_human",
                    detail="missing track_id",
                )
            )
            continue

        candidates = by_tid.get(tid, [])
        if not candidates:
            events.append(
                RemapEvent(
                    track=track_name,
                    track_id=tid,
                    old_slot_label=old_label,
                    new_slot_label=old_label,
                    status="needs_human",
                    detail="no pi slot for track_id",
                )
            )
            continue

        picked = _pick_pi_slot(row, candidates, gt_index=idx)
        if picked is None:
            events.append(
                RemapEvent(
                    track=track_name,
                    track_id=tid,
                    old_slot_label=old_label,
                    new_slot_label=old_label,
                    status="needs_human",
                    detail="ambiguous pi slot match",
                )
            )
            continue

        new_label = _slot_label(picked)
        if new_label == old_label:
            continue

        row["slot_label"] = new_label
        events.append(
            RemapEvent(
                track=track_name,
                track_id=tid,
                old_slot_label=old_label,
                new_slot_label=new_label,
                status="remapped",
            )
        )

    return out, events


def _apply_to_gt_set(yaml_path: Path, gt_doc: dict[str, Any]) -> None:
    match load(yaml_path):
        case Err(e):
            raise RuntimeError(f"GT load: {e.detail}")
        case Ok(gt):
            pass

    remapped_by_idx = {
        i: str(t.get("slot_label") or "")
        for i, t in enumerate(gt_doc.get("tracks") or [])
        if isinstance(t, dict)
    }
    new_tracks = tuple(
        replace(t, slot_label=remapped_by_idx.get(i, t.slot_label))
        for i, t in enumerate(gt.tracks)
    )
    match save(replace(gt, tracks=new_tracks), yaml_path):
        case Err(e):
            raise RuntimeError(f"GT save: {e.detail}")
        case _:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--yaml", type=Path, required=True)
    ap.add_argument("--set-id", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if args.dry_run and args.apply:
        print("Choose --dry-run or --apply, not both", file=sys.stderr)
        return 2
    if not args.dry_run and not args.apply:
        args.dry_run = True

    import yaml

    gt_doc = yaml.safe_load(args.yaml.read_text()) or {}
    pi_slots = fetch_remap_slot_rows(args.set_id, ssh_sqlite)
    out_doc, events = remap_slots(gt_doc, pi_slots)

    remapped = [e for e in events if e.status == "remapped"]
    needs_human = [e for e in events if e.status == "needs_human"]
    with_cue = sum(1 for r in pi_slots if _cue_time_s(r) is not None)

    print(f"pi_slots: {len(pi_slots)}")
    print(f"pi_slots_with_cue: {with_cue}")
    if pi_slots and with_cue == 0:
        print(
            "WARN: fetched pi slots have no cue_time_seconds/cue_seconds — "
            "time tie-break inactive",
            file=sys.stderr,
        )
    print(f"remapped: {len(remapped)}")
    print(f"needs_human: {len(needs_human)}")

    for e in remapped[:20]:
        print(
            f"  {e.old_slot_label} -> {e.new_slot_label}  {e.track_id}  {e.track[:50]}"
        )
    if len(remapped) > 20:
        print(f"  ... and {len(remapped) - 20} more")

    for e in needs_human:
        print(
            f"  NEEDS_HUMAN {e.old_slot_label}  {e.track_id}  {e.detail}  {e.track[:50]}"
        )

    if args.apply:
        _apply_to_gt_set(args.yaml, out_doc)
        print(f"Wrote {args.yaml}")
    else:
        print("(dry-run — no file written)")

    return 1 if needs_human else 0


if __name__ == "__main__":
    sys.exit(main())
