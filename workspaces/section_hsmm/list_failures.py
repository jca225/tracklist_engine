#!/usr/bin/env python3
"""List exactly where the vocal aligner fails: missed GT events + false spans,
with names and the GT attribute (loop/cut, pitch, short) that explains each.

Reads the fused timeline (out/<set>_fused_timeline.json) vs GT — no recompute.

Usage:
    venvs/audio/bin/python -m workspaces.section_hsmm.list_failures --set-id 1fsnxchk
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out"


def flags(r):
    s0, s1 = float(r["set_start_s"]), float(r["set_end_s"])
    f = []
    if bool(r.get("is_loop")) or len(r.get("ref_segments") or []) > 1:
        f.append("loop/cut")
    if abs(float(r.get("pitch_shift_semi") or 0)) >= 1:
        f.append("pitch")
    if (s1 - s0) < 12:
        f.append("short")
    return f or ["clean"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", default="1fsnxchk")
    args = p.parse_args(argv)

    tl = json.loads((OUT / f"{args.set_id}_fused_timeline.json").read_text())
    voc = [s for s in tl["spans"] if s["channel"] == "overlay"]
    import yaml
    gt = [r for r in yaml.safe_load(
        (_REPO / "labeling/fixtures/bb12_ground_truth.yaml").read_text())["tracks"]
        if r.get("claimed_stem") == "acappella" and r.get("track_id")]

    def ov(a0, a1, b0, b1):
        return a0 < b1 + 3 and a1 > b0 - 3

    missed, caught = [], 0
    for r in gt:
        tid, s0, s1 = str(r["track_id"]), float(r["set_start_s"]), float(r["set_end_s"])
        if any(sp["recording_id"] == tid and ov(sp["set_start_s"], sp["set_end_s"], s0, s1)
               for sp in voc):
            caught += 1
        else:
            missed.append((f"{r['track'][:42]}", s0, s1 - s0, flags(r)))

    false_pos = []
    for sp in voc:
        if not any(str(r["track_id"]) == sp["recording_id"]
                   and ov(sp["set_start_s"], sp["set_end_s"], float(r["set_start_s"]),
                          float(r["set_end_s"])) for r in gt):
            false_pos.append((sp["name"][:42], sp["set_start_s"]))

    print(f"=== vocal aligner failures ({args.set_id}) ===")
    print(f"GT vocal events: {len(gt)}  | caught: {caught}  | MISSED: {len(missed)}")
    print(f"predicted vocal spans: {len(voc)}  | FALSE (no GT match): {len(false_pos)}\n")

    from collections import Counter
    bycause = Counter(f for _, _, _, fl in missed for f in fl)
    print("MISSED events by cause flag:", dict(bycause))
    print("\nMISSED events (name @ mix-min, dur, why):")
    for name, s0, dur, fl in sorted(missed, key=lambda x: x[1]):
        print(f"  {s0/60:4.1f}m  {dur:4.0f}s  {','.join(fl):14}  {name}")
    print(f"\nFALSE positives (predicted vocal, no GT) — first 12 of {len(false_pos)}:")
    for name, s0 in sorted(false_pos)[:12]:
        print(f"  {s0/60:4.1f}m  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
