#!/usr/bin/env python3
"""Per-stem set_start placement — the HuBERT analog of
``mix_fp_hits.span_from_offset_votes`` for the vocal axis.

For acappella spans the full-mix landmark fingerprint is weak (vocals don't
fingerprint cleanly) and chroma is key-fragile, so ``decode_placements`` can't
place them. HuBERT is phonetic and key-invariant ("lyrics don't transpose"), so
it localizes a vocal section in the mix where chroma/fp cannot.

Method (joint, no oracle ref_start): tile the reference vocal stem into windows,
slide each over a BAND of ``mix_vocals`` around the coarse prior set_start, and
collect (ref_off, mix_pos, peak) votes. Cluster the votes by their alignment
diagonal ``d = mix_pos - ref_off``; on the dominant diagonal, set_start is the
start of the densest contiguous run of mix positions (the ``_cluster_at`` trick —
robust to a stray on-diagonal vote from a repeated chorus). The band leashes the
otherwise-unleashed matched filter to the coarse prior, and the caller's fusion
guard keeps the prior when HuBERT agrees closely (protects near-hits).

Validated BB12 acappella (n=24, vs GT, prior = the infer set_start; band ±90s):
    prior/MERT   median 11.9s  <8s 42%  <15s 54%
    HuBERT joint median  6.9s  <8s 71%  <15s 83%
    FUSED guard8 median  6.8s  <8s 75%  <15s 83%   (strictly dominates the prior)

This refines set_start ONLY. The joint ref_start is repeat-ambiguous (~52s on the
same set) — leave ref_start to ``refine_ref_offsets`` + fibers/continuity.
Instrumental set_start is NOT handled here (chroma fails on instrumental presence,
and GT n=5 can't validate) — a separate effort.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from workspaces.alignment_prototype.refine_ref_offsets import (
    HOP,
    SR,
    STRETCHES,
    correlate_window,
)
from workspaces.section_hsmm.similarity_probe import _hubert

FPS = SR / HOP  # feature frames per second on the SR/HOP grid

# defaults validated on BB12 acappella
BAND_S = 90.0
GUARD_S = 8.0
W_S = 12.0
HOP_WIN_S = 6.0
TOL_S = 4.0
GAP_S = 8.0


def _detect_in(
    win: np.ndarray, mix_band: np.ndarray, stretches: tuple[float, ...]
) -> tuple[int, float, float]:
    """(mix_frame, peak, stretch) of ref window ``win`` sliding over ``mix_band``.

    Inverse of refine_ref_offsets.detect_offset: there the mix window slides over
    the full ref (-> ref_start); here a ref window slides over the mix (-> the
    mix frame where it plays). ``stretch`` resamples the ref window."""
    n = win.shape[1]
    best = (0, 0.0, 1.0)
    for st in stretches:
        m = int(round(n * st))
        idx = np.clip((np.arange(m) / st).astype(int), 0, n - 1)
        k, score = correlate_window(win[:, idx], mix_band)
        if score > best[1]:
            best = (k, score, st)
    return best


def place_joint(
    mix_feat: np.ndarray,
    ref_feat: np.ndarray,
    prior_ss: float,
    span_dur: float,
    *,
    band_s: float = BAND_S,
    w_s: float = W_S,
    hop_s: float = HOP_WIN_S,
    stretches: tuple[float, ...] = STRETCHES,
    tol_s: float = TOL_S,
    gap_s: float = GAP_S,
) -> tuple[float, float, float] | None:
    """(set_start_s, ref_start_s, peak) | None — dominant-diagonal vote-extent.

    ``mix_feat`` / ``ref_feat`` are (D, T) on the SR/HOP grid (e.g. HuBERT). The
    mix search is restricted to ``prior_ss ± band_s`` (+ span); ``ref_start_s`` is
    the on-diagonal ref offset at the span start (repeat-ambiguous — use for
    provenance only, not as the ref offset)."""
    lo = max(0, int((prior_ss - band_s) * FPS))
    hi = min(mix_feat.shape[1], int((prior_ss + band_s + span_dur) * FPS))
    mix_band = mix_feat[:, lo:hi]
    nwin = int(w_s * FPS)
    if mix_band.shape[1] < nwin:
        return None
    step = int(hop_s * FPS)
    nref = ref_feat.shape[1]
    votes = []  # (ref_off_s, mix_pos_s, peak)
    for ro in range(0, max(1, nref - nwin), step):
        win = ref_feat[:, ro : ro + nwin]
        if win.shape[1] < nwin // 2:
            continue
        k, peak, _st = _detect_in(win, mix_band, stretches)
        votes.append((ro / FPS, lo / FPS + k / FPS, peak))
    if not votes:
        return None
    v = np.array(votes)  # (V, 3)
    d = v[:, 1] - v[:, 0]  # alignment diagonal mix_pos - ref_off
    order = np.argsort(-v[:, 2])
    # dominant diagonal = peak-weighted densest tol-cluster among the strongest
    best_d, best_w = d[order[0]], -1.0
    for i in order[: max(3, len(order) // 2)]:
        w = v[np.abs(d - d[i]) <= tol_s, 2].sum()
        if w > best_w:
            best_w, best_d = w, d[i]
    on = np.abs(d - best_d) <= tol_s
    strong = on & (v[:, 2] >= 0.5 * v[on, 2].max())
    if not strong.any():
        strong = on
    si = np.argsort(v[strong, 1])
    mp, ro_s, pk = v[strong, 1][si], v[strong, 0][si], v[strong, 2][si]
    runs = np.split(np.arange(len(mp)), np.where(np.diff(mp) > gap_s)[0] + 1)
    run = max(runs, key=len)  # densest contiguous on-diagonal run = the span
    return float(mp[run[0]]), float(ro_s[run[0]]), float(pk[run].max())


def hubert_of(path: Path | str, layer: int = 9) -> np.ndarray | None:
    """HuBERT features (768, T) on the SR/HOP grid for an audio file, or None."""
    import librosa

    p = Path(path)
    if not p.is_file():
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(str(p), sr=SR, mono=True)
    return _hubert(y, layer)
