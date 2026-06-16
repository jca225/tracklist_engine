#!/usr/bin/env python3
"""Diagnostic: split the vocal-channel misses to find recoverable headroom.

For each GT acappella event, two independent questions:
  (a) VERIFIABLE? — given the event's own window, does the true track rank #1
      among all candidates (the strong stretch-search matched filter = the 70%
      primitive, per event)?
  (b) CAUGHT? — did the blind v9 decode actually name it?

Cross-tabulating (a)x(b):
  verifiable & missed  -> RECOVERABLE by propose-then-verify (decode lost a
                          signal that was there).
  unverifiable & missed -> needs better separation/feature/overlap handling.
Also reports, for missed events, whether a CONCURRENT acappella overlapped
(tests the 'concurrent vocals are the ceiling' claim instead of asserting it).

Usage:
    venvs/audio/bin/python -m workspaces.section_hsmm.diagnose_misses --set-id 1fsnxchk
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
from workspaces.section_hsmm.decode_null import PENS, viterbi_null  # noqa: E402
from workspaces.section_hsmm.decode_v7 import build_vocal_vocab  # noqa: E402
from workspaces.section_hsmm.decode_v8 import spans_from_path  # noqa: E402
from workspaces.section_hsmm.decode_v9 import diag_smooth_multislope  # noqa: E402
from workspaces.section_hsmm.mfcc_emit import pooled_mfcc  # noqa: E402
from workspaces.section_hsmm.similarity_probe import FPS, _feat  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", default="1fsnxchk")
    p.add_argument("--frame-s", type=float, default=1.0)
    p.add_argument("--win-frames", type=int, default=6)
    p.add_argument("--beta", type=float, default=-0.05)
    p.add_argument("--max-win-s", type=float, default=15.0)
    args = p.parse_args(argv)

    set_dir = find_aligning_dir(args.set_id)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    for t in manifest["tracks"]:
        if t.get("recording_id"):
            by_tid.setdefault(t["recording_id"], t)
    import yaml
    gt_rows = [r for r in yaml.safe_load(
        (_REPO / "labeling/fixtures/bb12_ground_truth.yaml").read_text())["tracks"]
        if str(r.get("slot_label")) != "mix" and r.get("track_id")]
    for tid, t in by_tid.items():
        t["_stem"] = next((r.get("claimed_stem") or "regular" for r in gt_rows
                           if str(r["track_id"]) == tid), "regular")
    events = [(str(r["track_id"]), float(r["set_start_s"]), float(r["set_end_s"]))
              for r in gt_rows if (r.get("claimed_stem") or "regular") == "acappella"]
    pool = [t for t in by_tid if ref_audio_for({"claimed_stem": "acappella"},
                                               by_tid[t]) is not None]

    # --- blind v9 decode -> caught per event ---
    print("blind v9 decode …", file=sys.stderr)
    vocab = build_vocal_vocab(by_tid, args.frame_s)
    mixp = pooled_mfcc(set_dir / "mix_vocals.flac", f"{args.set_id}_mix_vocals",
                       f"{args.set_id}_mix_vocals_pool{args.frame_s}", args.frame_s)
    emis = diag_smooth_multislope((mixp @ vocab.emit_ref.T).astype(np.float32),
                                  vocab.slices, args.win_frames, (0.7, 0.85, 1.0, 1.18, 1.4))
    path = viterbi_null(emis, vocab, beta=args.beta, null_enter_pen=0.0, **PENS)
    spans = spans_from_path(path, vocab, args.frame_s)

    def caught(tid, s0, s1):
        return any(ptid == tid and ps0 < s1 + 3 and ps1 > s0 - 3
                   for ptid, ps0, ps1 in spans)

    # --- per-event verifiability (strong matched filter in own window) ---
    print(f"verifiability of {len(events)} events vs {len(pool)}-track pool …", file=sys.stderr)
    mixf = _feat(set_dir / "mix_vocals.flac", f"{args.set_id}_mix_vocals", "mfcc")
    rows = []
    for tid, s0, s1 in events:
        if tid not in pool:
            continue
        a = int(s0 * FPS)
        n = int(min(s1 - s0, args.max_win_s) * FPS)
        a = min(a, max(0, mixf.shape[1] - n))
        win = mixf[:, a:a + n]
        if win.shape[1] < 8:
            continue
        peaks = {}
        for c in pool:
            rf = _feat(ref_audio_for({"claimed_stem": "acappella"}, by_tid[c]),
                       f"ref_{c}_voc", "mfcc")
            if rf.shape[1] > win.shape[1]:
                peaks[c] = detect_offset(win, rf, STRETCHES)[1]
        if tid not in peaks:
            continue
        rank = 1 + sum(1 for k, v in peaks.items() if k != tid and v > peaks[tid])
        conc = sum(1 for t2, o0, o1 in events
                   if t2 != tid and o0 < s1 and o1 > s0)
        rows.append({"tid": tid, "s0": s0, "verifiable": rank == 1, "rank": rank,
                     "caught": caught(tid, s0, s1), "concurrent": conc > 0})

    n = len(rows)
    vf = [r for r in rows if r["verifiable"]]
    ca = [r for r in rows if r["caught"]]
    vf_miss = [r for r in rows if r["verifiable"] and not r["caught"]]
    unv_miss = [r for r in rows if not r["verifiable"] and not r["caught"]]
    print(f"\n=== miss diagnostic ({args.set_id}, {n} acappella events) ===")
    print(f"VERIFIABLE in own window (true ranks #1, full pool): {len(vf)}/{n} = "
          f"{100*len(vf)/n:.0f}%   (this is the given-region ceiling)")
    print(f"CAUGHT by blind v9 decode: {len(ca)}/{n} = {100*len(ca)/n:.0f}%\n")
    print("cross-tab:")
    print(f"  verifiable & caught   : {sum(1 for r in rows if r['verifiable'] and r['caught'])}")
    print(f"  verifiable & MISSED   : {len(vf_miss)}  <- RECOVERABLE by propose-then-verify")
    print(f"  unverifiable & caught : {sum(1 for r in rows if not r['verifiable'] and r['caught'])}")
    print(f"  unverifiable & missed : {len(unv_miss)}  <- needs separation/feature/overlap")
    # the concurrency challenge: of recoverable + unverifiable misses, how many concurrent?
    print(f"\nconcurrency among MISSED events (tests 'concurrent = ceiling'):")
    miss = [r for r in rows if not r["caught"]]
    print(f"  missed events: {len(miss)}  | with a concurrent acappella: "
          f"{sum(1 for r in miss if r['concurrent'])} "
          f"({100*sum(1 for r in miss if r['concurrent'])/max(len(miss),1):.0f}%)")
    print(f"  verifiable-but-missed with concurrency: "
          f"{sum(1 for r in vf_miss if r['concurrent'])}/{len(vf_miss)}")
    print(f"\nheadroom: blind {100*len(ca)/n:.0f}% -> verifiable ceiling "
          f"{100*len(vf)/n:.0f}%; propose-then-verify targets the {len(vf_miss)} "
          f"verifiable-but-missed (+{100*len(vf_miss)/n:.0f} pts if fully recovered)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
