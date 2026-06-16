#!/usr/bin/env python3
"""v9 — windowed (diagonal-smoothed) MFCC emission for the vocal channel.

v8's per-frame MFCC cosine got 35% event recall vs the 70% windowed-matched-
filter verification primitive. The gap is the emission: a single frame of vocal
timbre isn't unique, but a contiguous ~6s window is. v9 diagonal-smooths the
emission so the score at state (track k, ref-frame j) for mix-frame t reflects
the match of [t, t+w] against [j, j+w] (slope-1 diagonal — warp shown to cost
only ~4pts). Same NULL-default Viterbi + event scoring as v8.

Usage:
    venvs/audio/bin/python -m workspaces.section_hsmm.decode_v9 --set-id 1fsnxchk \
        [--frame-s 1.0] [--win-frames 6] [--beta -0.08]
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

from workspaces.alignment_prototype.refine_ref_offsets import find_aligning_dir  # noqa: E402
from workspaces.section_hsmm.decode_null import PENS, viterbi_null  # noqa: E402
from workspaces.section_hsmm.decode_v7 import build_vocal_vocab  # noqa: E402
from workspaces.section_hsmm.decode_v8 import event_score, spans_from_path  # noqa: E402
from workspaces.section_hsmm.mfcc_emit import pooled_mfcc  # noqa: E402


def diagonal_smooth(emis: np.ndarray, slices, w: int) -> np.ndarray:
    """Average each forward diagonal of length w within every candidate block:
    out[t, j] = mean_{d<w} emis[t+d, j+d]. Turns per-frame cosine into a windowed
    matched-filter score (sequence-aware)."""
    out = np.empty_like(emis)
    T = emis.shape[0]
    for lo, hi in slices:
        block = emis[:, lo:hi]
        L = hi - lo
        sm = np.zeros((T, L), dtype=emis.dtype)
        cnt = np.zeros((T, L), dtype=emis.dtype)
        for d in range(w):
            if d == 0:
                sm += block; cnt += 1
            else:
                sm[:T - d, :L - d] += block[d:, d:]
                cnt[:T - d, :L - d] += 1
        out[:, lo:hi] = sm / np.maximum(cnt, 1)
    return out


def diag_smooth_multislope(emis: np.ndarray, slices, w: int, slopes) -> np.ndarray:
    """Like diagonal_smooth but the diagonal may TILT: for each slope s, the ref
    advances round(d*s) per d mix-frames; take the best slope per (t,j). A
    locally-linear approximation of non-linear (Ableton warp-marker) warping."""
    out = np.empty_like(emis)
    T = emis.shape[0]
    for lo, hi in slices:
        block = emis[:, lo:hi]
        L = hi - lo
        best = np.full((T, L), -1e9, dtype=emis.dtype)
        for s in slopes:
            sm = np.zeros((T, L), dtype=emis.dtype)
            cnt = np.zeros((T, L), dtype=emis.dtype)
            for d in range(w):
                cj = int(round(d * s))
                if cj >= L:
                    break
                sm[:T - d, :L - cj] += block[d:, cj:]
                cnt[:T - d, :L - cj] += 1
            best = np.maximum(best, sm / np.maximum(cnt, 1))
        out[:, lo:hi] = best
    return out


def warp_dp_smooth(emis: np.ndarray, slices, gamma: float) -> np.ndarray:
    """DTW-style warp-tolerant emission: forward DP with local steps (1,1),(1,2),
    (2,1) so the alignment path can BEND (ref faster/slower moment-to-moment) —
    true non-linear warp, not just the fixed slopes of multislope. gamma<1 gives
    an effective window ~1/(1-gamma). out[t,j] = best decayed warp-aligned
    accumulated similarity ending at (t,j)."""
    out = np.empty_like(emis)
    T = emis.shape[0]
    for lo, hi in slices:
        sim = emis[:, lo:hi]
        L = hi - lo
        blk = np.empty((T, L), dtype=emis.dtype)
        blk[0] = sim[0]
        prev1 = sim[0].copy()
        prev2 = np.full(L, -1e9, dtype=emis.dtype)
        for t in range(1, T):
            s11 = np.full(L, -1e9, dtype=emis.dtype); s11[1:] = prev1[:-1]   # (1,1)
            s12 = np.full(L, -1e9, dtype=emis.dtype); s12[2:] = prev1[:-2]   # (1,2)
            s21 = np.full(L, -1e9, dtype=emis.dtype); s21[1:] = prev2[:-1]   # (2,1)
            best = np.maximum(np.maximum(s11, s12), s21)
            cur = sim[t] + gamma * np.where(best > -1e8, best, 0.0)
            blk[t] = cur
            prev2, prev1 = prev1, cur
        out[:, lo:hi] = blk * (1.0 - gamma)     # rescale toward per-frame range
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", default="1fsnxchk")
    p.add_argument("--frame-s", type=float, default=1.0)
    p.add_argument("--win-frames", type=int, default=6)
    p.add_argument("--beta", type=float, default=-0.08)
    p.add_argument("--null-enter-pen", type=float, default=0.0)
    p.add_argument("--multislope", action="store_true",
                   help="also test tilted diagonals (non-linear warp tolerance)")
    p.add_argument("--warpdp", action="store_true",
                   help="also test DTW-style bending warp DP emission")
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

    print(f"building vocal vocab @ {args.frame_s}s …", file=sys.stderr)
    vocab = build_vocal_vocab(by_tid, args.frame_s)
    mix = pooled_mfcc(set_dir / "mix_vocals.flac", f"{args.set_id}_mix_vocals",
                      f"{args.set_id}_mix_vocals_pool{args.frame_s}", args.frame_s)
    emis = (mix @ vocab.emit_ref.T).astype(np.float32)
    n_ev = sum(1 for r in gt_rows if (r.get("claimed_stem") or "regular") == "acappella")
    print(f"  {len(vocab.tids)} tracks, {vocab.emit_ref.shape[0]} states, "
          f"{mix.shape[0]} frames, {n_ev} GT events", file=sys.stderr)

    print(f"=== v9 windowed-MFCC vocal decode ({args.set_id}, {args.frame_s}s, "
          f"beta={args.beta}) ===")
    print(f"  {'emission':>16} {'evRecall':>9} {'evPrec':>7} {'onset_med':>10}")

    def run(label, sm):
        path = viterbi_null(sm, vocab, beta=args.beta,
                            null_enter_pen=args.null_enter_pen, **PENS)
        spans = spans_from_path(path, vocab, args.frame_s)
        rec, prec, _, nsp, onsets = event_score(spans, gt_rows)
        om = f"{np.median(onsets):.1f}s" if onsets else "-"
        print(f"  {label:>16} {100*rec:7.0f}% {100*prec:6.0f}% {om:>10}  ({nsp} spans)")

    run("per-frame (w=1)", emis)
    w = args.win_frames
    run(f"diag w={w*args.frame_s:.0f}s", diagonal_smooth(emis, vocab.slices, w))
    if args.multislope:
        slopes = (0.7, 0.85, 1.0, 1.18, 1.4)
        run(f"multislope w={w*args.frame_s:.0f}s",
            diag_smooth_multislope(emis, vocab.slices, w, slopes))
    if args.warpdp:
        for g in (0.75, 0.83):
            run(f"warpDP gamma={g}", warp_dp_smooth(emis, vocab.slices, g))
    return 0


if __name__ == "__main__":
    sys.exit(main())
