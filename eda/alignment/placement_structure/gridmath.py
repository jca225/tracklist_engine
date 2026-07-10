"""Shared grid arithmetic for the placement/structure EDA.

A "grid" is a sorted list of bar-start times in seconds (mix bar grid from
data/analysis/<set_id>_measure_times.json, or a reference track's downbeat
grid from track_analysis.downbeat_times_json). All functions are pure.
"""

from __future__ import annotations

import bisect
import cmath
import math
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseSample:
    """Where one event time falls inside its surrounding grid interval."""

    phase: float  # position in [0,1) within the bar
    dist_s: float  # seconds to the nearest bar line
    bar_len_s: float  # length of the surrounding bar
    bar_idx: int  # index of the nearest bar line


def phase_in_grid(t: float, grid: list[float]) -> PhaseSample | None:
    """Locate t inside grid; None if t is outside the grid span."""
    if len(grid) < 2 or t < grid[0] or t >= grid[-1]:
        return None
    i = bisect.bisect_right(grid, t) - 1
    lo, hi = grid[i], grid[i + 1]
    bar_len = hi - lo
    if bar_len <= 0:
        return None
    phase = (t - lo) / bar_len
    dist = min(t - lo, hi - t)
    nearest = i if (t - lo) <= (hi - t) else i + 1
    return PhaseSample(phase=phase, dist_s=dist, bar_len_s=bar_len, bar_idx=nearest)


def circular_stats(phases: list[float]) -> tuple[float, float, float]:
    """Resultant length R, mean phase, and Rayleigh-test p-value.

    Under uniform phases R -> 0; a placement grid shows up as R near 1 with
    tiny p. p uses the standard Rayleigh approximation (good for n >= 10).
    """
    n = len(phases)
    if n == 0:
        return 0.0, 0.0, 1.0
    vec = sum(cmath.exp(2j * math.pi * p) for p in phases) / n
    r = abs(vec)
    mean_phase = (cmath.phase(vec) / (2 * math.pi)) % 1.0
    z = n * r * r
    p = math.exp(-z) * (1 + (2 * z - z * z) / (4 * n))
    return r, mean_phase, max(0.0, min(1.0, p))


def phase_report(samples: list[PhaseSample], beats_per_bar: int = 4) -> dict:
    """Summary stats for a set of events against a bar grid.

    Reports both bar-level snap (phase within the bar) and beat-level snap
    (phase within the bar folded to quarter-bar), so beat-quantized-but-not-
    bar-quantized placement is still detected.
    """
    if not samples:
        return {"n": 0}
    phases = [s.phase for s in samples]
    beat_phases = [(s.phase * beats_per_bar) % 1.0 for s in samples]
    dists = sorted(s.dist_s for s in samples)
    beat_dists = sorted(
        min(bp, 1 - bp) * s.bar_len_s / beats_per_bar
        for s, bp in zip(samples, beat_phases)
    )
    bar_r, bar_mu, bar_p = circular_stats(phases)
    beat_r, beat_mu, beat_p = circular_stats(beat_phases)
    n = len(samples)
    hist = [0] * 8
    for p in phases:
        hist[min(7, int(p * 8))] += 1
    return {
        "n": n,
        "median_dist_to_barline_s": round(statistics.median(dists), 3),
        "p90_dist_to_barline_s": round(dists[int(0.9 * (n - 1))], 3),
        "frac_within_eighth_beat": round(
            sum(1 for d in beat_dists if d <= 0.0587) / n, 3
        ),  # 0.0587 s = 1/8 beat at 127.7 BPM
        "median_dist_to_beat_s": round(statistics.median(beat_dists), 3),
        "bar": {"R": round(bar_r, 3), "mean_phase": round(bar_mu, 3), "p": bar_p},
        "beat": {"R": round(beat_r, 3), "mean_phase": round(beat_mu, 3), "p": beat_p},
        "phase_hist8": hist,
    }


def fold_intervals(ivals: list[float], ref: float) -> tuple[list[float], float]:
    """Fold half/double-time tracker flips back onto the reference interval.

    An interval more than 1.5x the reference is a missed beat/bar (halve it);
    one under ref/1.5 is an inserted one (double it). Returns the folded
    series and the fraction of intervals that needed folding — a direct
    measure of tracker instability, separate from true tempo drift.
    """
    out: list[float] = []
    folded = 0
    for v in ivals:
        orig = v
        while v > 1.5 * ref:
            v /= 2
        while v < ref / 1.5:
            v *= 2
        if v != orig:
            folded += 1
        out.append(v)
    return out, (folded / len(ivals) if ivals else 0.0)


def local_bpm_curve(
    beat_times: list[float],
    *,
    min_ibi_s: float = 0.2,
    max_ibi_s: float = 2.0,
    smooth_window: int = 9,
) -> tuple[list[float], float]:
    """Instantaneous BPM per beat via running-median-smoothed inter-beat
    intervals. Returns (smoothed bpm curve, artifact fraction).

    IBIs outside [min_ibi_s, max_ibi_s] (30-300 BPM) are tracker artifacts
    (missed/doubled beats) and are dropped before smoothing.
    """
    ibis = [b - a for a, b in zip(beat_times, beat_times[1:])]
    clean = [i for i in ibis if min_ibi_s <= i <= max_ibi_s]
    artifact_frac = 1 - len(clean) / len(ibis) if ibis else 1.0
    if len(clean) < smooth_window:
        return [], artifact_frac
    half = smooth_window // 2
    smoothed = [
        60.0 / statistics.median(clean[max(0, k - half) : k + half + 1])
        for k in range(len(clean))
    ]
    return smoothed, artifact_frac
