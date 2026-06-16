#!/usr/bin/env python3
"""Vocal-channel event taxonomy from BB12 GT — grounds a vocal-specific Viterbi.

John's enumeration of acappella events maps to states/transitions a vocal
decoder must model (the bed Viterbi doesn't): loops, <5s 'one-liner' stabs
(often repeated pre-drop), part->different-part of the same acappella, acappella
overlaid on another, acappella->different acappella, acappella->none, none->
acappella. This counts each in the hand labels so the model isn't guessed.

Usage:
    venvs/audio/bin/python -m workspaces.section_hsmm.vocal_events
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt", type=Path,
                   default=_REPO / "labeling/fixtures/bb12_ground_truth.yaml")
    args = p.parse_args(argv)

    rows = [r for r in yaml.safe_load(args.gt.read_text())["tracks"]
            if (r.get("claimed_stem") == "acappella") and r.get("track_id")]
    n = len(rows)
    dur = np.array([float(r["set_end_s"]) - float(r["set_start_s"]) for r in rows])
    print(f"=== BB12 vocal-event taxonomy ({n} acappella plays) ===\n")

    # E2 one-liners: short plays
    print("[len] acappella play length:")
    print(f"   median={np.median(dur):.1f}s  <5s: {(dur<5).sum()}  "
          f"<10s: {(dur<10).sum()}  <15s: {(dur<15).sum()}  (of {n})")

    # E1 loops + E3 part->different-part (within-acappella movement)
    loops = [r for r in rows if r.get("is_loop")]
    seg_rows = [r for r in rows if r.get("ref_segments")]
    seg_durs, fwd, back = [], [], []
    for r in seg_rows:
        segs = r["ref_segments"]
        for s in segs:
            if s.get("ref_end_s") is not None:
                seg_durs.append(float(s["ref_end_s"]) - float(s["ref_start_s"]))
        for a, b in zip(segs, segs[1:]):
            d = float(b["ref_start_s"]) - float(a["ref_end_s"])
            (fwd if d > 0 else back).append(abs(d))
    print(f"\n[E1 loop] is_loop flagged: {len(loops)}  | "
          f"backward loop-jumps in segments: {len(back)} "
          f"(median {np.median(back):.1f}s)" if back else f"\n[E1 loop] is_loop: {len(loops)}")
    print(f"[E3 part->part same acap] multi-segment plays: {len(seg_rows)}/{n}  | "
          f"forward section-jumps: {len(fwd)} (median {np.median(fwd):.0f}s)" if fwd else
          f"[E3] multi-segment: {len(seg_rows)}")
    if seg_durs:
        sd = np.array(seg_durs)
        print(f"   segment durations: median={np.median(sd):.1f}s  "
              f"<5s (stabs): {(sd<5).sum()}/{len(sd)}")

    # E2 repeated one-liners: same tid replayed within 30s
    by_tid: dict[str, list[float]] = {}
    for r in rows:
        by_tid.setdefault(str(r["track_id"]), []).append(float(r["set_start_s"]))
    quick_replays = 0
    for tid, starts in by_tid.items():
        starts.sort()
        quick_replays += sum(1 for a, b in zip(starts, starts[1:]) if b - a < 30)
    print(f"\n[E2 one-liner replays] same acappella replayed within 30s: {quick_replays}")

    # E4-E7 transitions between consecutive acappella spans (by set_start)
    ac = sorted(rows, key=lambda r: float(r["set_start_s"]))
    overlap = switch = to_none = 0
    for a, b in zip(ac, ac[1:]):
        gap = float(b["set_start_s"]) - float(a["set_end_s"])
        if gap < -2:
            overlap += 1                        # E4 concurrent acappellas
        elif gap <= 2:
            switch += 1                         # E5 acappella -> acappella
        else:
            to_none += 1                        # E6/E7 acappella -> none -> acappella
    print(f"\n[transitions between acappella spans] (n={len(ac)-1})")
    print(f"   E4 overlaid (concurrent, gap<-2s): {overlap}")
    print(f"   E5 acap->acap (clean switch |gap|<=2s): {switch}")
    print(f"   E6/E7 acap->none->acap (gap>2s): {to_none}")

    # fraction of mix timeline with ANY acappella (vocal channel is intermittent)
    spans = sorted((float(r["set_start_s"]), float(r["set_end_s"])) for r in rows)
    merged: list[list[float]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    voc = sum(e - s for s, e in merged)
    print(f"\n[intermittency] acappella present for {voc:.0f}s "
          f"(union of spans) — the vocal channel is bursty, NULL is the default state")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
