#!/usr/bin/env python3
"""Empirically derive HSMM state & transition types from BB12 hand labels.

Instead of hand-setting the Markov chain, read what Two Friends actually did in
the ground truth: how long plays are (the "make it shorter" hunch), where in a
song they enter, how they move within a song (linear vs section-jump vs loop),
how songs layer (mashups), and how they hand off between songs (cut / crossfade
/ overlap). The output is a quantified taxonomy to ground the decode's states
and transition penalties.

Usage:
    venvs/audio/bin/python -m workspaces.section_hsmm.gt_taxonomy [--gt PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[2]


def q(a: list[float] | np.ndarray, label: str) -> str:
    a = np.asarray([x for x in a if x is not None], dtype=float)
    if a.size == 0:
        return f"{label}: (none)"
    return (f"{label}: n={a.size} median={np.median(a):.1f} "
            f"p10={np.percentile(a,10):.1f} p90={np.percentile(a,90):.1f} "
            f"min={a.min():.1f} max={a.max():.1f}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt", type=Path,
                   default=_REPO / "labeling/fixtures/bb12_ground_truth.yaml")
    args = p.parse_args(argv)

    rows = [r for r in yaml.safe_load(args.gt.read_text())["tracks"]
            if str(r.get("slot_label")) != "mix" and r.get("track_id")]
    n = len(rows)
    print(f"=== BB12 GT taxonomy ({n} plays) ===\n")

    # --- A. plays, reprises, stems ---
    by_tid: dict[str, list[dict]] = {}
    for r in rows:
        by_tid.setdefault(str(r["track_id"]), []).append(r)
    plays_per = [len(v) for v in by_tid.values()]
    stems: dict[str, int] = {}
    for r in rows:
        stems[r.get("claimed_stem") or "regular"] = stems.get(r.get("claimed_stem") or "regular", 0) + 1
    print(f"[A] distinct songs: {len(by_tid)}  | stems: {stems}")
    print(f"    reprises: {sum(1 for x in plays_per if x>1)} songs played >1x "
          f"(max {max(plays_per)}x); play-count dist {dict(sorted({x: plays_per.count(x) for x in set(plays_per)}.items()))}\n")

    # --- B. play durations (the "make it shorter" test) ---
    set_dur = [float(r["set_end_s"]) - float(r["set_start_s"]) for r in rows]
    ref_dur = [float(r["ref_end_s"]) - float(r["ref_start_s"]) for r in rows
               if r.get("ref_end_s") is not None]
    aud = []
    for r in rows:
        if r.get("audible_end_s") is not None:
            aud.append(float(r["audible_end_s"]) - float(r["set_start_s"]))
    print("[B] PLAY LENGTH (the shortening hunch)")
    print("   ", q(set_dur, "set span (s)"))
    print("   ", q(ref_dur, "ref span used (s)"))
    if aud:
        print("   ", q(aud, "audible span (s)"))
    short = sum(1 for d in set_dur if d < 30)
    print(f"    plays < 30s: {short}/{n} ({100*short/n:.0f}%)  "
          f"< 20s: {sum(1 for d in set_dur if d<20)}\n")

    # --- C. tempo / pitch / entry point ---
    tr = [float(r["tempo_ratio"]) for r in rows if r.get("tempo_ratio") is not None]
    ps = [float(r["pitch_shift_semi"]) for r in rows if r.get("pitch_shift_semi") is not None]
    rs = [float(r["ref_start_s"]) for r in rows if r.get("ref_start_s") is not None]
    print("[C] REALIZATION")
    print("   ", q(tr, "tempo_ratio"))
    print("   ", q(ps, "pitch_shift_semi"))
    print("   ", q(rs, "ref entry point (s into song)"))
    early = sum(1 for x in rs if x < 20)
    print(f"    entries < 20s into song (intro): {early}/{len(rs)} ({100*early/max(len(rs),1):.0f}%)\n")

    # --- D. within-song movement: linear vs loop vs section-jump ---
    loops = sum(1 for r in rows if r.get("is_loop"))
    seg_rows = [r for r in rows if r.get("ref_segments")]
    seg_counts = [len(r["ref_segments"]) for r in seg_rows]
    fwd_jumps, back_jumps = [], []
    for r in seg_rows:
        segs = r["ref_segments"]
        for a, b in zip(segs, segs[1:]):
            d = float(b["ref_start_s"]) - float(a["ref_end_s"])
            (fwd_jumps if d > 0 else back_jumps).append(d)
    print("[D] WITHIN-SONG MOVEMENT")
    print(f"    linear single-clip plays: {n - len(seg_rows)}/{n}  "
          f"| multi-segment plays: {len(seg_rows)}  | flagged is_loop: {loops}")
    if seg_counts:
        print("   ", q(seg_counts, "segments per multi-seg play"))
    if fwd_jumps:
        print("   ", q(fwd_jumps, "forward section-jump (s skipped)"))
    if back_jumps:
        print("   ", q([abs(x) for x in back_jumps], "backward loop-jump (s)"))
    print()

    # --- E. reprise advancement: do later plays advance through the song? ---
    adv_mono = adv_tot = 0
    for tid, plays in by_tid.items():
        if len(plays) < 2:
            continue
        ps_sorted = sorted(plays, key=lambda r: float(r["set_start_s"]))
        starts = [float(r["ref_start_s"]) for r in ps_sorted if r.get("ref_start_s") is not None]
        for a, b in zip(starts, starts[1:]):
            adv_tot += 1
            if b >= a - 5:  # later play enters at/after earlier (allow 5s slack)
                adv_mono += 1
    print("[E] REPRISE STRUCTURE")
    if adv_tot:
        print(f"    consecutive-reprise transitions: {adv_tot}  | "
              f"later play enters >= earlier: {adv_mono} ({100*adv_mono/adv_tot:.0f}%)\n")
    else:
        print("    (no multi-play songs)\n")

    # --- F. layering (mashup concurrency) ---
    events = []
    for r in rows:
        events.append((float(r["set_start_s"]), 1))
        events.append((float(r["set_end_s"]), -1))
    events.sort()
    depth = 0
    depth_time: dict[int, float] = {}
    prev_t = events[0][0]
    for t, d in events:
        depth_time[depth] = depth_time.get(depth, 0.0) + (t - prev_t)
        depth += d
        prev_t = t
    tot = sum(depth_time.values()) or 1
    print("[F] LAYERING (concurrent songs)")
    for k in sorted(depth_time):
        print(f"    {k} song(s): {100*depth_time[k]/tot:4.0f}% of mix time")
    print()

    # --- G. song-to-song handoff (gap between consecutive plays) ---
    starts_sorted = sorted(rows, key=lambda r: float(r["set_start_s"]))
    gaps = []
    for a, b in zip(starts_sorted, starts_sorted[1:]):
        gaps.append(float(b["set_start_s"]) - float(a["set_end_s"]))
    overlap = sum(1 for g in gaps if g < -2)
    cut = sum(1 for g in gaps if -2 <= g <= 2)
    gap = sum(1 for g in gaps if g > 2)
    print("[G] HANDOFF between consecutive plays (by set_start order)")
    print(f"    overlap/layered (gap<-2s): {overlap}  | "
          f"clean cut (|gap|<=2s): {cut}  | silence-ish gap (>2s): {gap}")
    print("   ", q([abs(g) for g in gaps if g < -2], "overlap depth (s)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
