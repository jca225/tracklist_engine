#!/usr/bin/env python3
"""Subsequence-DTW + match-rate verifier (Kim et al. ISMIR 2020 method) for vocals.

The paper aligns a mix subsequence to a full track with subsequence DTW and gates
by MATCH RATE (fraction of diagonal steps in the warping path; they threshold
0.4). Match-rate is the principled confidence that would have stopped our warp-DP
overfitting — a bendy path scores low. This applies it to our hardest channel:
matched filter proposes a top-K shortlist per event, subsequence DTW re-ranks by
match rate. Tests vs the sub-window matched-filter ceiling (71%).

Usage:
    venvs/audio/bin/python -m workspaces.section_hsmm.subseq_dtw_probe --set-id 1fsnxchk
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


def _ds(feat, factor):
    n = feat.shape[1] // factor
    if n == 0:
        n, factor = 1, feat.shape[1]
    pooled = feat[:, :n * factor].reshape(feat.shape[0], n, factor).mean(axis=2)
    pooled = np.nan_to_num(pooled, nan=0.0)
    nz = np.linalg.norm(pooled, axis=0, keepdims=True)
    nz[nz < 1e-6] = 1e-6           # silent frames -> tiny uniform, never zero
    return (pooled / nz).astype(np.float32)


def _match_rate(query, refseq):
    """subsequence DTW matching short QUERY within long REFSEQ -> match_rate
    (fraction of diagonal warping-path steps; Kim et al.'s confidence).
    euclidean on L2-normed features (~cosine) avoids NaN on silent frames."""
    import librosa
    if refseq.shape[1] <= query.shape[1] or query.shape[1] < 4:
        return 0.0
    _D, wp = librosa.sequence.dtw(X=query, Y=refseq, metric="euclidean",
                                  subseq=True, backtrack=True)
    diag = sum(1 for a, b in zip(wp, wp[1:]) if a[0] - b[0] == 1 and a[1] - b[1] == 1)
    return diag / max(len(wp) - 1, 1)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", default="1fsnxchk")
    p.add_argument("--win-s", type=float, default=12.0)
    p.add_argument("--topk", type=int, default=12)
    p.add_argument("--ds", type=int, default=6, help="downsample factor (~43/ds fps)")
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
    refc: dict[str, np.ndarray] = {}

    def ref(c):
        if c not in refc:
            refc[c] = _feat(ref_audio_for({"claimed_stem": "acappella"}, by_tid[c]),
                            f"ref_{c}_voc", "mfcc")
        return refc[c]

    print(f"subseq-DTW verify {len(rows)} events (shortlist {args.topk}, ds {args.ds}) …",
          file=sys.stderr)
    mf_ok = dtw_ok = n = 0
    for r in rows:
        tid = str(r["track_id"])
        if tid not in pool:
            continue
        s0 = float(r["set_start_s"])
        a = int(s0 * FPS); ln = int(min(float(r["set_end_s"]) - s0, args.win_s) * FPS)
        a = min(a, max(0, mixf.shape[1] - ln))
        win = mixf[:, a:a + ln]
        if win.shape[1] < 8:
            continue
        # propose: matched-filter peaks over full pool -> shortlist
        peaks = {c: detect_offset(win, ref(c), STRETCHES)[1]
                 for c in pool if ref(c).shape[1] > win.shape[1]}
        if tid not in peaks:
            continue
        n += 1
        mf_ok += int(all(peaks[tid] >= v for k, v in peaks.items() if k != tid))
        shortlist = sorted(peaks, key=peaks.get, reverse=True)[:args.topk]
        if tid not in shortlist:
            continue  # DTW can't recover what the shortlist dropped
        Yd = _ds(win, args.ds)
        mr = {c: _match_rate(Yd, _ds(ref(c), args.ds)) for c in shortlist}
        dtw_ok += int(all(mr[tid] >= v for k, v in mr.items() if k != tid))

    print(f"\n=== subseq-DTW vs matched-filter ({args.set_id}, {n} events) ===")
    print(f"matched-filter retrieval@1 (full pool):     {100*mf_ok/max(n,1):.0f}%")
    print(f"subseq-DTW match-rate re-rank (on shortlist): {100*dtw_ok/max(n,1):.0f}%")
    print(f"(sub-window matched-filter ceiling = 71%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
