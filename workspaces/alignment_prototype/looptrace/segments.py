#!/usr/bin/env python3
"""Phase 3 — fixed-slope diagonal extraction (1D Hough) + segment-cover DP.

Every matched landmark point (t_mix, t_song) lying on a played diagonal of
slope s satisfies t_song = s*t_mix + b: transform each point to its
intercept b and histogram (1D Hough). Populated bins = candidate diagonals;
their inliers' mix-time density gives per-time support. The cover DP then
tiles mix time with candidates: maximize accumulated support minus
lambda * (number of transitions) minus gap penalties. NO monotonicity
constraint and NO backward-jump penalty (brief §2.4 — it hurt).

The unit of evidence is the LONG collinear accumulation per diagonal — the
property the per-window matched filter lacked. A wrong-repeat diagonal dies
where surrounding content diverges; the true one keeps collecting points
across chorus->verse boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from workspaces.alignment_prototype.looptrace.config import SEG_V1, SegmentConfig


@dataclass(frozen=True)
class Diagonal:
    intercept_s: float  # b: t_song = slope*t_mix + b
    votes: int


def hough_diagonals(
    points: np.ndarray, slope: float, cfg: SegmentConfig = SEG_V1
) -> list[Diagonal]:
    """Candidate diagonals from the intercept histogram."""
    if len(points) == 0:
        return []
    b = points[:, 1] - slope * points[:, 0]
    lo = float(b.min())
    idx = np.floor((b - lo) / cfg.hough_bin_s).astype(np.int64)
    counts = np.bincount(idx)
    # smooth +-1 bin (diagonal spread from slope error / quantization)
    sm = counts.astype(np.float64)
    if len(sm) > 2:
        sm = sm + np.roll(counts, 1) * 0.5 + np.roll(counts, -1) * 0.5
    order = np.argsort(-sm)
    cands: list[Diagonal] = []
    for k in order:
        if counts[k] == 0 or sm[k] < cfg.min_votes:
            break
        b0 = lo + (k + 0.5) * cfg.hough_bin_s
        if any(abs(b0 - c.intercept_s) < cfg.min_separation_s for c in cands):
            continue
        # refine to the inlier median — bin-center quantization (+-0.25 s)
        # would otherwise eat most of the +-0.25 s scoring tolerance
        near = b[np.abs(b - b0) <= cfg.hough_bin_s]
        if near.size:
            b0 = float(np.median(near))
        cands.append(Diagonal(round(float(b0), 3), int(counts[k])))
        if len(cands) >= cfg.max_candidates:
            break
    return cands


def _support(
    points: np.ndarray,
    diag: Diagonal,
    slope: float,
    grid: np.ndarray,
    cfg: SegmentConfig,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Per-grid-time inlier support for one diagonal (kernel-smoothed)."""
    b = points[:, 1] - slope * points[:, 0]
    m = np.abs(b - diag.intercept_s) <= cfg.inlier_tol_s
    if not m.any():
        return np.zeros_like(grid)
    t = points[m, 0]
    w = weights[m] if weights is not None else np.ones(t.size)
    out = np.zeros_like(grid)
    sig = cfg.support_sigma_s
    for ti, wi in zip(t, w):
        out += wi * np.exp(-0.5 * ((grid - ti) / sig) ** 2)
    return out


def cover_dp(
    points: np.ndarray,
    slope: float,
    span_dur_s: float,
    cfg: SegmentConfig = SEG_V1,
    *,
    weights: np.ndarray | None = None,
    diagonals: list[Diagonal] | None = None,
) -> list[tuple[float, float, float]]:
    """Segment list [(mix_start_s, song_start_s, song_end_s)] tiling the span.

    weights: optional per-point multiplier (loop-collapse evidence weight).
    States = candidate diagonals + NULL; emission = per-time support (NULL
    emits `null_level`); switching states costs `lam` (uniform — no
    directional prior). Runs of one diagonal become segments; NULL runs are
    gaps (the scorer's piecewise interpolation bridges them)."""
    diags = diagonals if diagonals is not None else hough_diagonals(points, slope, cfg)
    if not diags:
        return []
    grid = np.arange(0.0, max(span_dur_s, cfg.grid_step_s), cfg.grid_step_s)
    emis = np.stack(
        [_support(points, d, slope, grid, cfg, weights) for d in diags], axis=0
    )
    # hash-collision noise gives EVERY diagonal a large shared background
    # (measured ~90 vs the true diagonal's ~160): subtract the per-time
    # cross-candidate median, then normalize columns so emissions are
    # scale-free shares and `lam` means the same on every span. In landmark
    # deserts all shares ~0 and the constant NULL row wins.
    if emis.shape[0] > 1:
        emis = np.maximum(emis - np.median(emis, axis=0, keepdims=True), 0.0)
    emis = np.vstack([emis, np.full((1, grid.size), cfg.null_level)])  # NULL row
    emis = emis / (emis.sum(axis=0, keepdims=True) + 1e-9)
    n_states, t_steps = emis.shape
    lam = cfg.lam
    score = emis[:, 0].copy()
    back = np.zeros((n_states, t_steps), dtype=np.int32)
    for t in range(1, t_steps):
        best = score.max()
        arg = int(score.argmax())
        stay = score
        jump = best - lam
        nxt = np.where(stay >= jump, stay, jump) + emis[:, t]
        back[:, t] = np.where(stay >= jump, np.arange(n_states), arg)
        score = nxt
    path = np.zeros(t_steps, dtype=np.int32)
    path[-1] = int(score.argmax())
    for t in range(t_steps - 1, 0, -1):
        path[t - 1] = back[path[t], t]
    # runs of one diagonal -> segments
    segs: list[tuple[float, float, float]] = []
    null_state = n_states - 1
    i = 0
    while i < t_steps:
        s = path[i]
        j = i
        while j < t_steps and path[j] == s:
            j += 1
        if s != null_state and (j - i) * cfg.grid_step_s >= cfg.min_segment_s:
            m0, m1 = grid[i], grid[min(j, t_steps - 1)]
            b0 = diags[s].intercept_s
            segs.append(
                (
                    round(float(m0), 3),
                    round(float(slope * m0 + b0), 3),
                    round(float(slope * m1 + b0), 3),
                )
            )
        i = j
    return segs


def total_support(
    points: np.ndarray,
    slope: float,
    cfg: SegmentConfig = SEG_V1,
) -> float:
    """Slope-quality score: total votes over all candidate diagonals — used
    to pick/refine the slope once per span (brief §6.3)."""
    return float(sum(d.votes for d in hough_diagonals(points, slope, cfg)))
