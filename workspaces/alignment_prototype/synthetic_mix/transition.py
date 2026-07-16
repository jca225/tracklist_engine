"""Continuous tempo-ride synthetic spans — the §12 transition-regime probe.

The v2 generator warps each span by a CONSTANT `tempo_ratio` (one
`librosa.time_stretch` rate), so mix-time maps to ref-time by a straight line. A
real DJ *transition* rides tempo continuously — a moving warp curve. This module
builds a span whose ref-seconds-consumed-per-mix-second varies over the span (a
linear BPM ramp), renders it exactly by resampling the ref along the *integrated*
time map (varispeed — pitch rides with tempo, as an un-keylocked platter ride
does), and labels it with the true `ref_time(mix_time)` curve.

Convention (matches warp_model / GT `tempo_ratio`): `r(t)` = ref-seconds per
mix-second at mix-time `t`. `r == 1` is native; `r < 1` slows the ref (a
fast payload played slow). The instantaneous ratio ramps linearly from `r0` to
`r1` across the span; ref-time is its integral, so mix↔ref is *quadratic*, not
linear — that curvature is the whole point of the probe.

Feeds the aligner; how well it recovers the curve = the Aug-1 transition-scope
answer (decision #12). Keylocked (pitch-preserving) rides are a v2 — varispeed is
the clean, exactly-labelled first test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TempoRide:
    """A linear instantaneous-tempo-ratio ramp `r0 -> r1` over `mix_dur_s`.

    `r(t) = r0 + (r1 - r0) * t / T` for t in [0, T]. ref-time is the integral:
    `ref(t) = r0*t + (r1 - r0)*t^2 / (2T)` — quadratic in mix-time. `ref_start_s`
    is where in the ref the span begins (the play-in point).
    """

    r0: float
    r1: float
    mix_dur_s: float
    ref_start_s: float = 0.0

    def __post_init__(self) -> None:
        if self.mix_dur_s <= 0 or self.r0 <= 0 or self.r1 <= 0:
            raise ValueError("mix_dur_s, r0, r1 must be positive")

    def ref_time(self, t: float | np.ndarray) -> float | np.ndarray:
        """ref-seconds consumed by mix-time `t` (absolute, incl. ref_start_s)."""
        t = np.asarray(t, dtype=float)
        slope = (self.r1 - self.r0) / self.mix_dur_s
        consumed = self.r0 * t + 0.5 * slope * t * t
        return self.ref_start_s + consumed

    @property
    def ref_span_s(self) -> float:
        """Total ref-seconds the ride consumes (mean ratio * duration)."""
        return 0.5 * (self.r0 + self.r1) * self.mix_dur_s

    def mean_ratio(self) -> float:
        return 0.5 * (self.r0 + self.r1)

    def label_points(self, n: int = 33) -> list[tuple[float, float]]:
        """`n` (mix_t, ref_t) control points spanning the ride — the GT curve."""
        ts = np.linspace(0.0, self.mix_dur_s, n)
        return [(float(t), float(self.ref_time(t))) for t in ts]


def render_ride(ref: np.ndarray, sr: int, ride: TempoRide) -> np.ndarray:
    """Render `ride` by resampling `ref` along its integrated time map.

    For each output (mix) sample at mix-time `t`, pull the ref amplitude at
    `ref_time(t)` via linear interpolation. This IS a continuous varispeed warp
    whose time map exactly equals the label — no BPM math, no block artefacts.
    Ref-time beyond the available audio is zero-filled (silence past the tail).
    """
    if ref.ndim != 1:
        ref = np.asarray(ref).mean(axis=0) if ref.ndim == 2 else ref.ravel()
    n_out = max(1, int(round(ride.mix_dur_s * sr)))
    mix_t = np.arange(n_out, dtype=float) / sr
    ref_t = ride.ref_time(mix_t)  # absolute ref seconds per output sample
    ref_samples = np.arange(ref.shape[0], dtype=float) / sr
    out = np.interp(ref_t, ref_samples, ref, left=0.0, right=0.0)
    # zero anything that reads past the real ref tail (interp clamps, so detect)
    out[ref_t > ref_samples[-1]] = 0.0
    return out.astype(np.float32)


def recovered_curve_error(
    ride: TempoRide,
    decoded_points: list[tuple[float, float]],
) -> dict[str, float]:
    """Compare a decoder's (mix_t, ref_t) points against the true ride.

    Returns ref-time MAE / p90 (seconds) over the decoded mix-times, plus the
    recovered mean-ratio error — the headline transition-recovery numbers. A
    straight-line (constant-warp) decoder will show a bowed residual whose size
    is the ride's curvature, which is exactly what we want to quantify.
    """
    if not decoded_points:
        return {
            "ref_mae_s": float("nan"),
            "ref_p90_s": float("nan"),
            "mean_ratio_err": float("nan"),
            "n": 0.0,
        }
    mt = np.array([p[0] for p in decoded_points], dtype=float)
    rt = np.array([p[1] for p in decoded_points], dtype=float)
    truth = ride.ref_time(mt)
    err = np.abs(rt - truth)
    # recovered mean ratio = slope of a straight fit through decoded points
    if len(mt) >= 2 and np.ptp(mt) > 0:
        slope = np.polyfit(mt, rt, 1)[0]
    else:
        slope = float("nan")
    return {
        "ref_mae_s": float(np.mean(err)),
        "ref_p90_s": float(np.percentile(err, 90)),
        "mean_ratio_err": float(abs(slope - ride.mean_ratio())),
        "n": float(len(mt)),
    }
