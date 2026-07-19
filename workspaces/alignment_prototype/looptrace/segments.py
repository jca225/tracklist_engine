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


@dataclass(frozen=True)
class DiagonalRun:
    """One explicit non-NULL run from the segment-cover path."""

    mix_start_s: float
    mix_end_s: float
    song_start_s: float
    song_end_s: float
    diagonal: Diagonal
    evidence: float


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


def _noise_floor(
    points: np.ndarray,
    diags: list[Diagonal],
    slope: float,
    grid: np.ndarray,
    cfg: SegmentConfig,
    weights: np.ndarray | None,
    n_probe: int = 8,
) -> np.ndarray:
    """Per-time background support: median over random intercepts at least
    min_separation away from every candidate (deterministic probe set)."""
    b = points[:, 1] - slope * points[:, 0]
    lo, hi = float(b.min()), float(b.max())
    if hi - lo < 4 * cfg.min_separation_s:
        return np.zeros_like(grid)
    rng = np.random.default_rng(0)
    probes: list[np.ndarray] = []
    tries = 0
    while len(probes) < n_probe and tries < n_probe * 8:
        tries += 1
        b0 = float(rng.uniform(lo, hi))
        if any(abs(b0 - d.intercept_s) < cfg.min_separation_s for d in diags):
            continue
        probes.append(_support(points, Diagonal(b0, 0), slope, grid, cfg, weights))
    if not probes:
        return np.zeros_like(grid)
    return np.median(np.stack(probes, axis=0), axis=0)


def cover_dp_runs(
    points: np.ndarray,
    slope: float,
    span_dur_s: float,
    cfg: SegmentConfig = SEG_V1,
    *,
    weights: np.ndarray | None = None,
    diagonals: list[Diagonal] | None = None,
    bonus: np.ndarray | None = None,
) -> tuple[list[DiagonalRun], float]:
    """Decode explicit non-NULL runs, including their mix-time end boundaries.

    weights: optional per-point multiplier (loop-collapse evidence weight).
    States = candidate diagonals + NULL; emission = per-time support (NULL
    emits `null_level`); switching states costs `lam` (uniform — no
    directional prior). Runs of one diagonal become segments; NULL runs are
    gaps (the scorer's piecewise interpolation bridges them).

    `evidence` = the ABSOLUTE floor-subtracted support summed along the
    decoded path (non-NULL steps): how much above-background collinear
    evidence this decode explains. Unlike the share-normalized DP score it
    is comparable ACROSS slopes — used to pick the slope by decode quality
    (histogram peakiness alone was measured to lose to noise peaks on 4
    weak-evidence spans)."""
    diags = diagonals if diagonals is not None else hough_diagonals(points, slope, cfg)
    if not diags:
        return [], 0.0
    grid = np.arange(0.0, max(span_dur_s, cfg.grid_step_s), cfg.grid_step_s)
    emis = np.stack(
        [_support(points, d, slope, grid, cfg, weights) for d in diags], axis=0
    )
    # hash-collision noise gives EVERY diagonal a large shared background
    # (measured ~90 vs the true diagonal's ~160): subtract a per-time NOISE
    # floor, then normalize columns so emissions are scale-free shares and
    # `lam` means the same on every span. The floor is the median support
    # of RANDOM non-candidate intercepts — NOT the cross-candidate median,
    # which self-annihilates when all candidates are true (a jump span at
    # the correct slope has exactly 2 candidates, both real; their median
    # is their own average). In landmark deserts all shares ~0 and the
    # constant NULL row wins.
    floor = _noise_floor(points, diags, slope, grid, cfg, weights)
    emis = np.maximum(emis - floor[None, :], 0.0)
    raw = np.vstack([emis, np.zeros((1, grid.size))])  # absolute units + NULL
    emis = np.vstack([emis, np.full((1, grid.size), cfg.null_level)])  # NULL row
    emis = emis / (emis.sum(axis=0, keepdims=True) + 1e-9)
    if bonus is not None:  # zero-sum rival tiebreak, share space (residual.py)
        emis[:-1] += bonus
        emis = np.maximum(emis, 0.0)
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
    evidence = float(raw[path, np.arange(t_steps)].sum())
    # Runs of one diagonal become bounded segments. The legacy cover_dp return
    # shape omitted mix_end and inferred it from the next segment, which cannot
    # represent a genuine NULL gap. Keep the boundary here while the decoded
    # state path still makes it explicit.
    runs: list[DiagonalRun] = []
    null_state = n_states - 1
    i = 0
    while i < t_steps:
        s = path[i]
        j = i
        while j < t_steps and path[j] == s:
            j += 1
        if s != null_state and (j - i) * cfg.grid_step_s >= cfg.min_segment_s:
            m0 = float(grid[i])
            m1 = min(float(span_dur_s), float(grid[j - 1] + cfg.grid_step_s))
            b0 = diags[s].intercept_s
            run_ev = float(raw[s, i:j].sum())
            runs.append(
                DiagonalRun(
                    mix_start_s=round(m0, 3),
                    mix_end_s=round(m1, 3),
                    song_start_s=round(float(slope * m0 + b0), 3),
                    song_end_s=round(float(slope * m1 + b0), 3),
                    diagonal=diags[s],
                    evidence=run_ev,
                )
            )
        i = j
    return runs, evidence


def cover_dp(
    points: np.ndarray,
    slope: float,
    span_dur_s: float,
    cfg: SegmentConfig = SEG_V1,
    *,
    weights: np.ndarray | None = None,
    diagonals: list[Diagonal] | None = None,
    bonus: np.ndarray | None = None,
) -> tuple[list[tuple[float, float, float]], float]:
    """Legacy segment shape over :func:`cover_dp_runs`.

    Existing looptrace consumers intentionally retain
    ``(mix_start, song_start, song_end)``. New whole-mix consumers use the
    bounded runs so NULL gaps and re-entry boundaries survive materialization.
    """
    runs, evidence = cover_dp_runs(
        points,
        slope,
        span_dur_s,
        cfg,
        weights=weights,
        diagonals=diagonals,
        bonus=bonus,
    )
    last_grid_s = float(
        np.arange(0.0, max(span_dur_s, cfg.grid_step_s), cfg.grid_step_s)[-1]
    )
    legacy = []
    for run in runs:
        # Preserve cover_dp's historical final-cell convention exactly. The
        # bounded API deliberately extends the final cell to its right edge;
        # existing looptrace callers historically ended it at the grid centre.
        song_end = run.song_end_s
        if run.mix_end_s >= span_dur_s:
            song_end = round(
                float(slope * last_grid_s + run.diagonal.intercept_s), 3
            )
        legacy.append((run.mix_start_s, run.song_start_s, song_end))
    return legacy, evidence


def mel_emission(
    mix_mel: np.ndarray,
    mix_hz: float,
    ref_mel: np.ndarray,
    ref_hz: float,
    diags: list[Diagonal],
    slope: float,
    grid: np.ndarray,
    cfg: SegmentConfig = SEG_V1,
) -> np.ndarray:
    """(n_diags, T) hybrid content-verification term, share space.

    Whitened-mel cosine between the mix window at grid time t and the
    reference at each diagonal's mapped position — the legacy matched
    filter's evidence injected at looptrace's exact offsets. Per-time
    CONTRAST (cross-candidate mean subtracted, zero-sum) and CLIPPED to
    ±mel_cap so it tips landmark ties but cannot overturn a strong
    landmark margin (lever-3's unbounded variant flipped a correct linear
    span, BB12 051). Weight/cap fixture-tuned."""
    n = len(diags)
    out = np.zeros((n, grid.size), dtype=np.float64)
    half_m = max(1, int(cfg.mel_window_s / 2 * mix_hz))
    half_r = max(1, int(cfg.mel_window_s / 2 * ref_hz))
    for ti, t in enumerate(grid):
        m0 = int(t * mix_hz)
        mwin = mix_mel[:, max(0, m0 - half_m) : m0 + half_m]
        if mwin.shape[1] < 2:
            continue
        sims = np.full(n, np.nan)
        for i, d in enumerate(diags):
            r0 = int((slope * t + d.intercept_s) * ref_hz)
            rwin = ref_mel[:, max(0, r0 - half_r) : r0 + half_r]
            k = min(mwin.shape[1], rwin.shape[1])
            if k < 2:
                continue
            sims[i] = float((mwin[:, :k] * rwin[:, :k]).sum(axis=0).mean())
        ok = np.isfinite(sims)
        if ok.sum() < 2:
            continue
        contrast = sims - float(sims[ok].mean())
        contrast[~ok] = 0.0
        out[:, ti] = np.clip(cfg.mel_weight * contrast, -cfg.mel_cap, cfg.mel_cap)
    return out


def path_inlier_evidence(
    points: np.ndarray,
    segs: list[tuple[float, float, float]],
    slope: float,
    span_dur_s: float,
    cfg: SegmentConfig = SEG_V1,
    *,
    weights: np.ndarray | None = None,
) -> float:
    """Weighted count of points within `evidence_tol_s` of the decoded
    piecewise map — the cross-slope comparison currency.

    The DP-internal support (0.6 s tol + 1.5 s kernel) is too forgiving to
    compare slopes: a wrong slope's diagonal still captures ~16 s of the
    true diagonal's points (|dslope|*L inside tolerance) and can out-total
    the truth. At +-0.3 s a 7.5% slope error sheds inliers within ~4 s."""
    if not segs or len(points) == 0:
        return 0.0
    w = weights if weights is not None else np.ones(len(points))
    total = 0.0
    for i, (m0, s0, _s1) in enumerate(segs):
        m1 = segs[i + 1][0] if i + 1 < len(segs) else span_dur_s
        b0 = s0 - slope * m0
        b = points[:, 1] - slope * points[:, 0]
        sel = (
            (np.abs(b - b0) <= cfg.evidence_tol_s)
            & (points[:, 0] >= m0)
            & (points[:, 0] <= m1)
        )
        total += float(w[sel].sum())
    return total


def total_support(
    points: np.ndarray,
    slope: float,
    cfg: SegmentConfig = SEG_V1,
) -> float:
    """Slope-quality score: PEAKINESS of the intercept histogram (top bins
    above the background), used to pick the slope once per span.

    Not the candidate-vote sum: hash-collision noise fills bins at every
    slope, and summing many noise-heavy bins let octave-folded wrong slopes
    outscore the true one on real mixes (measured: slope 0.53/0.39 picks).
    The true slope CONCENTRATES matches into few bins; noise stays flat."""
    if len(points) == 0:
        return 0.0
    b = points[:, 1] - slope * points[:, 0]
    idx = np.floor((b - float(b.min())) / cfg.hough_bin_s).astype(np.int64)
    counts = np.bincount(idx).astype(np.float64)
    nz = counts[counts > 0]
    bg = float(np.median(nz)) if nz.size else 0.0
    top = np.sort(counts)[-3:]
    return float(np.maximum(top - bg, 0.0).sum())
