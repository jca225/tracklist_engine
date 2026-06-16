#!/usr/bin/env python3
"""Does sub-window matching fix loop/cut unverifiability? (the causal test)

Loop/cut-ups are ~45% of unverifiable events. The verifier matches ONE
contiguous 15s window, but a looped/chopped vocal isn't contiguous in the ref,
so the match fails. This re-tests verifiability with the best of several SHORT
sub-windows (each can fit within a single segment) vs the single long window,
split by loop/cut vs not. If loop/cut events jump, it's a decoding-side fix.

Usage:
    venvs/audio/bin/python -m workspaces.section_hsmm.loop_cut_probe --set-id 1fsnxchk
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.refine_ref_offsets import (  # noqa: E402
    STRETCHES, detect_offset, find_aligning_dir, ref_audio_for,
)
from workspaces.section_hsmm.similarity_probe import FPS, _feat  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", default="1fsnxchk")
    p.add_argument("--long-s", type=float, default=15.0)
    p.add_argument("--sub-s", type=float, default=8.0)
    p.add_argument("--stride-s", type=float, default=5.0)
    args = p.parse_args(argv)

    set_dir = find_aligning_dir(args.set_id)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    for t in manifest["tracks"]:
        if t.get("recording_id"):
            by_tid.setdefault(t["recording_id"], t)
    import yaml
    rows = [r for r in yaml.safe_load(
        (_REPO / "labeling/fixtures/bb12_ground_truth.yaml").read_text())["tracks"]
        if r.get("claimed_stem") == "acappella" and r.get("track_id")]
    pool = [t for t in by_tid if ref_audio_for({"claimed_stem": "acappella"},
                                               by_tid[t]) is not None]
    mixf = _feat(set_dir / "mix_vocals.flac", f"{args.set_id}_mix_vocals", "mfcc")

    def windows(s0, s1):
        long_n = int(min(s1 - s0, args.long_s) * FPS)
        a0 = int(s0 * FPS)
        a0 = min(a0, max(0, mixf.shape[1] - long_n))
        wins_long = [mixf[:, a0:a0 + long_n]]
        sub_n = int(args.sub_s * FPS)
        subs = []
        t = s0
        while t + args.sub_s <= s1 + 0.1:
            a = min(int(t * FPS), max(0, mixf.shape[1] - sub_n))
            subs.append(mixf[:, a:a + sub_n])
            t += args.stride_s
        if not subs:
            subs = wins_long
        return wins_long[0], subs

    print(f"verifying {len(rows)} events vs {len(pool)} pool "
          f"(long={args.long_s:.0f}s vs sub={args.sub_s:.0f}s) …", file=sys.stderr)
    recs = []
    ref_cache: dict[str, np.ndarray] = {}
    for r in rows:
        tid = str(r["track_id"])
        if tid not in pool:
            continue
        s0, s1 = float(r["set_start_s"]), float(r["set_end_s"])
        wlong, subs = windows(s0, s1)
        if wlong.shape[1] < 8:
            continue
        loopcut = bool(r.get("is_loop")) or len(r.get("ref_segments") or []) > 1
        peak_long, peak_sub = {}, {}
        for c in pool:
            if c not in ref_cache:
                ref_cache[c] = _feat(ref_audio_for({"claimed_stem": "acappella"}, by_tid[c]),
                                     f"ref_{c}_voc", "mfcc")
            rf = ref_cache[c]
            if rf.shape[1] <= wlong.shape[1]:
                continue
            peak_long[c] = detect_offset(wlong, rf, STRETCHES)[1]
            peak_sub[c] = max(detect_offset(w, rf, STRETCHES)[1]
                              for w in subs if rf.shape[1] > w.shape[1] >= 8)
        if tid not in peak_long or tid not in peak_sub:
            continue
        vl = all(peak_long[tid] >= v for k, v in peak_long.items() if k != tid)
        vs = all(peak_sub[tid] >= v for k, v in peak_sub.items() if k != tid)
        recs.append({"loopcut": loopcut, "long": vl, "sub": vs})

    def show(label, sub):
        if not sub:
            return
        print(f"{label:>16} {len(sub):>4} {100*np.mean([r['long'] for r in sub]):11.0f}% "
              f"{100*np.mean([r['sub'] for r in sub]):14.0f}%")

    print(f"\n=== loop/cut sub-window probe ({args.set_id}, {len(recs)} events) ===")
    print(f"{'subset':>16} {'n':>4} {'long 15s':>12} {'best sub-win':>15}")
    show("ALL", recs)
    show("loop/cut", [r for r in recs if r["loopcut"]])
    show("non-loop/cut", [r for r in recs if not r["loopcut"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
