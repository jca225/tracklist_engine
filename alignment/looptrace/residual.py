#!/usr/bin/env python3
"""Phase 5 (redesigned) — residual tiebreak INSIDE the DP.

Phase 4's post-hoc instance re-selection regressed because its arbiter
compared landmark-support counts in regions the audit never covered. This
term instead injects the tie-break where the decision is actually made —
the DP emissions — and only where it is measurable:

  - RIVALS: candidate diagonals whose intercepts differ by (a multiple of)
    an audit repeat lag are images of the same content; landmark support
    cannot separate them. Everything else is untouched.
  - EVIDENCE: whitened-mel similarity between the mix at grid time t and
    the reference at each rival's mapped position, weighted by the Phase-4
    discriminability mask and restricted to AUDIT-COVERED frames
    (mask < 1); windows with no covered frames contribute nothing.
  - ZERO-SUM within a rival group (sim_i − group mean), added in the DP's
    normalized-share space: it cannot inflate rivals against non-rival
    diagonals or NULL, it can only redistribute among the tied.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from alignment.looptrace import selfsim
from alignment.looptrace.config import RES_V1, ResidualConfig
from alignment.looptrace.segments import Diagonal

_CACHE = Path(__file__).parent / ".lm_cache"


def mel_cached(ref_path: str) -> tuple[np.ndarray, float]:
    """Whitened-mel features of a reference, cached to disk."""
    tag = hashlib.md5(f"{ref_path}|mel-v1".encode()).hexdigest()[:16]
    f = _CACHE / f"{tag}_mel.npz"
    if f.is_file():
        d = np.load(f)
        return d["g"], float(d["hz"])
    g, hz = selfsim.mel_features(selfsim.load_audio(ref_path))
    _CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(f, g=g, hz=hz)
    return g, hz


def rival_groups(
    diags: list[Diagonal],
    song_pairs: list[dict],
    cfg: ResidualConfig = RES_V1,
) -> list[list[int]]:
    """Groups of diagonal indices that are repeat images of each other:
    |b_i - b_j| matches an audit repeat lag within tolerance."""
    lags = [p["lag_s"] for p in song_pairs]
    n = len(diags)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            d = abs(diags[i].intercept_s - diags[j].intercept_s)
            if any(abs(d - lg) <= cfg.lag_tol_s for lg in lags):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    # a real repeat has 2-4 instances; a bigger "group" means the audit-lag
    # matching chained transitively across unrelated diagonals (measured:
    # all 24 candidates in one group on BB12 slot 051) — degenerate, skip.
    return [g for g in groups.values() if 2 <= len(g) <= cfg.max_group]


def residual_bonus(
    mix_mel: np.ndarray,
    mix_hz: float,
    ref_mel: np.ndarray,
    ref_hz: float,
    mask: np.ndarray,
    mask_hz: float,
    diags: list[Diagonal],
    groups: list[list[int]],
    slope: float,
    grid: np.ndarray,
    cfg: ResidualConfig = RES_V1,
) -> np.ndarray:
    """(n_diags, T) zero-sum share-space bonus (0 outside rival groups)."""
    n = len(diags)
    bonus = np.zeros((n, grid.size), dtype=np.float64)
    half = max(1, int(cfg.window_s / 2 * ref_hz))
    for group in groups:
        # TIEBREAK ONLY (brief §8.3): when the landmark votes already
        # separate the rivals, the mel term must not override them —
        # measured to flip correct linear instances (BB12 linear 50->38)
        # when allowed to act on non-tied groups.
        votes = sorted((diags[gi].votes for gi in group), reverse=True)
        if votes[0] >= cfg.tie_ratio_max * max(votes[1], 1):
            continue
        for ti, t in enumerate(grid):
            m0 = int(t * mix_hz)
            mwin = mix_mel[:, max(0, m0 - half) : m0 + half]
            if mwin.shape[1] < 2:
                continue
            sims: list[float] = []
            for gi in group:
                p = slope * t + diags[gi].intercept_s
                r0 = int(p * ref_hz)
                rwin = ref_mel[:, max(0, r0 - half) : r0 + half]
                k = min(mwin.shape[1], rwin.shape[1])
                if k < 2:
                    sims.append(np.nan)
                    continue
                mi = np.clip(
                    ((r0 - half + np.arange(k)) / ref_hz * mask_hz).astype(int),
                    0,
                    mask.size - 1,
                )
                w = mask[mi].astype(np.float64)
                w[w >= cfg.covered_below] = 0.0  # uncovered frames: no vote
                if w.sum() < cfg.min_covered_weight:
                    sims.append(np.nan)
                    continue
                cos = (mwin[:, :k] * rwin[:, :k]).sum(axis=0)
                sims.append(float((cos * w).sum() / w.sum()))
            arr = np.array(sims, dtype=np.float64)
            ok = np.isfinite(arr)
            if ok.sum() < 2:
                continue
            mean = float(arr[ok].mean())
            for gi, s in zip(group, arr):
                if np.isfinite(s):
                    bonus[gi, ti] = cfg.weight * (s - mean)
    return bonus
