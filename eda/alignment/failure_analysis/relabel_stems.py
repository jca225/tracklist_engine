#!/usr/bin/env python3
"""Write GT-stem-corrected copies of the base timelines for the local re-measure.

The pipeline routes decoders on the timeline's ``claimed_stem`` (materialized
set_track_slots), corrupted by the row-text drop bug — 39% of true acappellas
read 'regular' and get chroma instead of HuBERT. This overwrites each span's
claimed_stem with the matched GT row's value and writes
``out/<set>_predicted_timeline_gtstem.json``, so ``joint_ref_decode --decoder
looptrace`` re-routes them correctly. Non-destructive: base + _lt untouched.

NB this corrects DECODE routing only. set_start for the mis-routed acappellas was
placed by infer WITHOUT HuBERT stem-placement (they were tagged 'regular'); that
would need a full infer re-run. So the re-measure isolates decode-routing, holding
placement at its current (partly mis-routed) value.

Usage:
    venvs/audio/bin/python -m eda.alignment.failure_analysis.relabel_stems
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.score_timeline_vs_gt import norm_slot  # noqa: E402

_ALN = _REPO / "workspaces" / "alignment_prototype"
SETS = [
    ("1fsnxchk", _REPO / "labeling/fixtures/bb12_ground_truth.yaml"),
    ("2nvzlh2k", _REPO / "labeling/fixtures/bb11_ground_truth.yaml"),
]


def _gt_index(gt_path: Path, set_id: str) -> dict[str, list[dict]]:
    rows = [
        r
        for r in yaml.safe_load(gt_path.read_text())["tracks"]
        if str(r.get("slot_label")) != "mix"
    ]
    id_map_path = _REPO / "labeling" / "fixtures" / "id_maps" / f"{set_id}.json"
    id_map = json.loads(id_map_path.read_text()) if id_map_path.exists() else {}
    by_tid: dict[str, list[dict]] = {}
    for r in rows:
        tid = str(r.get("track_id") or "")
        if tid in id_map and id_map[tid] != tid:
            r["track_id"] = tid = id_map[tid]
        if tid:
            by_tid.setdefault(tid, []).append(r)
    return by_tid


def main() -> int:
    for set_id, gt_path in SETS:
        base = _ALN / "out" / f"{set_id}_predicted_timeline.json"
        tl = json.loads(base.read_text())
        by_tid = _gt_index(gt_path, set_id)
        changed = 0
        for s in tl["spans"]:
            rows = by_tid.get(s["recording_id"])
            if not rows:
                continue
            g = min(rows, key=lambda r: abs(float(r["set_start_s"]) - s["set_start_s"]))
            gt_stem = g.get("claimed_stem") or "regular"
            if (s.get("claimed_stem") or "regular") != gt_stem:
                s["claimed_stem"] = gt_stem
                changed += 1
        dest = _ALN / "out" / f"{set_id}_predicted_timeline_gtstem.json"
        dest.write_text(json.dumps(tl, indent=2))
        print(f"[{set_id}] relabeled {changed}/{len(tl['spans'])} spans -> {dest.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
