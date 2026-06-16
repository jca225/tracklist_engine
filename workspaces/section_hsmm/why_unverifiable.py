#!/usr/bin/env python3
"""Decompose WHY acappella events are unverifiable — instead of asserting overlap.

For each GT acappella event, compute verifiability (true track ranks #1 in its
own window vs full pool, strong stretch-search matched filter), then tabulate
verifiable vs not against GT attributes: is_loop / multi-segment (ref non-
contiguous), pitch-shift, short, low audible_frac, and TRUE concurrency (overlap
>= 5s with another acappella). Shows which actually predict the 34% ceiling.

Usage:
    venvs/audio/bin/python -m workspaces.section_hsmm.why_unverifiable --set-id 1fsnxchk
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


def attrs(r, all_events):
    s0, s1 = float(r["set_start_s"]), float(r["set_end_s"])
    segs = r.get("ref_segments") or []
    conc = sum(1 for t2, o0, o1 in all_events
               if (t2 is not r) and o0 < s1 - 5 and o1 > s0 + 5)  # >=5s real overlap
    return {
        "loop_or_cut": bool(r.get("is_loop")) or len(segs) > 1,
        "pitch": abs(float(r.get("pitch_shift_semi") or 0)) >= 1,
        "short": (s1 - s0) < 10,
        "low_audible": (r.get("audible_frac") is not None and float(r["audible_frac"]) < 0.7),
        "concurrent5s": conc > 0,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", default="1fsnxchk")
    p.add_argument("--max-win-s", type=float, default=15.0)
    args = p.parse_args(argv)

    set_dir = find_aligning_dir(args.set_id)
    manifest = json.loads((set_dir / "manifest.json").read_text())
    by_tid = {t["track_id"]: t for t in manifest["tracks"]}
    for t in manifest["tracks"]:
        if t.get("recording_id"):
            by_tid.setdefault(t["recording_id"], t)
    import yaml
    ac_rows = [r for r in yaml.safe_load(
        (_REPO / "labeling/fixtures/bb12_ground_truth.yaml").read_text())["tracks"]
        if (r.get("claimed_stem") == "acappella") and r.get("track_id")]
    ev_times = [(r, float(r["set_start_s"]), float(r["set_end_s"])) for r in ac_rows]
    pool = [t for t in by_tid if ref_audio_for({"claimed_stem": "acappella"},
                                               by_tid[t]) is not None]

    mixf = _feat(set_dir / "mix_vocals.flac", f"{args.set_id}_mix_vocals", "mfcc")
    print(f"verifying {len(ac_rows)} events vs {len(pool)}-track pool …", file=sys.stderr)
    recs = []
    for r in ac_rows:
        tid = str(r["track_id"])
        if tid not in pool:
            continue
        s0 = float(r["set_start_s"])
        a = int(s0 * FPS); n = int(min(float(r["set_end_s"]) - s0, args.max_win_s) * FPS)
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
        verifiable = all(peaks[tid] >= v for k, v in peaks.items() if k != tid)
        recs.append({"verifiable": verifiable, **attrs(r, ev_times)})

    n = len(recs)
    unv = [x for x in recs if not x["verifiable"]]
    vf = [x for x in recs if x["verifiable"]]
    print(f"\n=== why unverifiable ({args.set_id}, {n} events, "
          f"{len(vf)} verifiable / {len(unv)} not) ===")
    keys = ["loop_or_cut", "pitch", "short", "low_audible", "concurrent5s"]
    print(f"{'attribute':>14}  {'% of UNVERIF':>12}  {'% of VERIF':>11}  {'lift':>5}")
    for k in keys:
        pu = np.mean([x[k] for x in unv]) if unv else 0
        pv = np.mean([x[k] for x in vf]) if vf else 0
        print(f"{k:>14}  {100*pu:11.0f}%  {100*pv:10.0f}%  {pu-pv:+.2f}")
    none_of = [x for x in unv if not any(x[k] for k in keys)]
    print(f"\nunverifiable with NONE of these flags (clean single vocal that still "
          f"failed -> separation/feature): {len(none_of)}/{len(unv)}")
    print(f"unverifiable explained by loop/cut OR pitch: "
          f"{sum(1 for x in unv if x['loop_or_cut'] or x['pitch'])}/{len(unv)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
