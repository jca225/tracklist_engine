#!/usr/bin/env python3
"""Score a predicted timeline (infer + refine_ref_offsets) against GT.

End-to-end pipeline scorecard — unlike eval_ref_detection (which probes with
GT set positions), this scores the actual pipeline output: identity, set
placement, and ref offsets, per stem channel. Ref offsets are scored only on
straight-clip GT rows (loops/segments aren't representable by the current
single-(ref_start, stretch) span output — counted separately).

Usage:
    venvs/audio/bin/python -m workspaces.alignment_prototype.score_timeline_vs_gt \\
        --set-id 1fsnxchk [--gt labeling/fixtures/bb12_ground_truth.yaml]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

OUT_DIR = Path(__file__).resolve().parent / "out"


def norm_slot(s: str) -> str:
    """'006w2' -> '6w2', '013' -> '13' — GT zero-pads, set_track_slots doesn't."""
    m = re.match(r"^0*(\d+)(w\d+)?$", str(s).strip())
    return f"{m.group(1)}{m.group(2) or ''}" if m else str(s).strip()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", required=True)
    p.add_argument("--gt", type=Path,
                   default=_REPO / "labeling/fixtures/bb12_ground_truth.yaml")
    args = p.parse_args(argv)

    timeline = json.loads((OUT_DIR / f"{args.set_id}_predicted_timeline.json").read_text())
    gt_rows = [r for r in yaml.safe_load(args.gt.read_text())["tracks"]
               if str(r.get("slot_label")) != "mix"]
    # GT slot labels are the HUMAN's section numbering (002-155 on BB12),
    # not the tracklist's slot space — match by recording + time, never by
    # slot label.
    gt_by_tid: dict[str, list[dict]] = {}
    for r in gt_rows:
        if r.get("track_id"):
            gt_by_tid.setdefault(str(r["track_id"]), []).append(r)

    id_ok, id_bad, no_gt = 0, [], 0
    place_errs, ref_rows = [], []
    loops_hit = 0
    for s in timeline["spans"]:
        slot = norm_slot(s["slot_label"])
        # identity: any GT row overlapping the predicted span in time whose
        # track matches? (GT rows without track_id can't vote)
        overlapping = [r for r in gt_rows
                       if r.get("track_id")
                       and float(r["set_start_s"]) < s["set_end_s"] + 5
                       and float(r["set_end_s"]) > s["set_start_s"] - 5]
        if overlapping:
            if any(str(r["track_id"]) == s["recording_id"] for r in overlapping):
                id_ok += 1
            else:
                id_bad.append((slot, s["recording_id"],
                               sorted({str(r["track_id"]) for r in overlapping})[:3],
                               s["name"][:36]))
        # placement + ref: nearest same-recording GT row
        rows = gt_by_tid.get(s["recording_id"])
        if not rows:
            no_gt += 1
            continue
        g = min(rows, key=lambda r: abs(float(r["set_start_s"]) - s["set_start_s"]))
        place_errs.append((abs(float(g["set_start_s"]) - s["set_start_s"]), slot,
                           s["set_start_s"], float(g["set_start_s"]), s["name"][:36]))
        if g.get("is_loop") or g.get("ref_segments"):
            loops_hit += 1
            continue
        ratio = float(g.get("tempo_ratio") or 1.0)
        if not (0.9 <= ratio <= 1.15):
            loops_hit += 1
            continue
        expected = float(g["ref_start_s"]) + (s["set_start_s"] - float(g["set_start_s"])) * ratio
        ref_rows.append((abs(s["ref_start_s"] - expected), slot,
                         s.get("claimed_stem") or "regular",
                         s["ref_start_s"], expected, s["name"][:36]))

    n = len(timeline["spans"])
    print(f"=== end-to-end pipeline vs GT ({args.set_id}, {n} predicted spans) ===")
    nid = id_ok + len(id_bad)
    print(f"identity: {id_ok}/{nid} ({100 * id_ok / max(nid, 1):.0f}%)  "
          f"[{no_gt} spans had no same-slot GT row]")
    pe = np.array([r[0] for r in place_errs])
    print(f"set placement |pred-gt|: median={np.median(pe):.1f}s  "
          f"<5s: {100 * (pe < 5).mean():.0f}%  <15s: {100 * (pe < 15).mean():.0f}%  "
          f"p90={np.percentile(pe, 90):.1f}s  (n={len(pe)})")
    re_ = np.array([r[0] for r in ref_rows])
    if re_.size:
        print(f"ref offset |pred-gt| (straight clips, n={len(re_)}; "
              f"{loops_hit} loop/segment spans excluded): "
              f"median={np.median(re_):.1f}s  <2s: {100 * (re_ < 2).mean():.0f}%  "
              f"<5s: {100 * (re_ < 5).mean():.0f}%  p90={np.percentile(re_, 90):.1f}s")
        for stem in ("regular", "acappella", "instrumental"):
            e = np.array([r[0] for r in ref_rows if r[2] == stem])
            if e.size:
                print(f"  {stem:13} n={len(e):3} median={np.median(e):.1f}s  "
                      f"<2s: {100 * (e < 2).mean():.0f}%  <5s: {100 * (e < 5).mean():.0f}%")

    print("\nworst placement:")
    for err, slot, pred, gt_v, name in sorted(place_errs, reverse=True)[:8]:
        print(f"  {err:7.1f}s {slot:6} pred={pred:8.1f} gt={gt_v:8.1f}  {name}")
    print("\nworst ref offset:")
    for err, slot, _stem, pred, exp, name in sorted(ref_rows, reverse=True)[:8]:
        print(f"  {err:7.1f}s {slot:6} pred={pred:8.1f} gt={exp:8.1f}  {name}")
    if id_bad:
        print(f"\nidentity misses ({len(id_bad)}):")
        for slot, rid, tids, name in id_bad[:10]:
            print(f"  {slot:6} pred={rid} gt={','.join(tids[:3])}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
