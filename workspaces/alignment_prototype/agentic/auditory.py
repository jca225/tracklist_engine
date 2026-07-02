"""Auditory-neuroscience probes — the deterministic DSP layer the LLM adjudicates.

Design frame (Liu et al. 2026, "Dive into Claude Code"): a good agent is ~98%
deterministic harness and ~2% model decision logic. Here the *probes* are that
harness — cheap, calibrated signal processing — and the LLM policy
(``llm_policy.py``) is the thin adjudicator over their outputs. Every function
here is a pure, testable transform: the model never touches audio.

Each probe formalizes one primitive of biological auditory scene analysis. The
current probe set (chroma/HuBERT/fp/lyrics) is almost all *identity-and-match*
operators; the ear's primitives are mostly *grouping-and-segregation* operators
— "how many sources, and when did each start". Those are the missing rank of
the placement problem (see docs/auditory_probes_plan.md):

- ``onset_envelope`` / ``onset_align`` — cochlear-nucleus onset cells: fire only
  at energy onsets. Placement is fundamentally onset-alignment; a DJ's beatmatch
  preserves the onset grid. THE flagship placement probe.
- ``onset_asynchrony`` — the brain's strongest source-split cue: components that
  start >~30 ms apart segregate. A second onset stream ⇒ overlay present. A
  source-count estimator the fusion arbiter currently has to infer.
- ``harmonic_salience`` / ``harmonic_sieve_match`` — periodicity coding: group
  partials at integer multiples of a common f0. Pitch-invariant identity, robust
  where chroma breaks (the 31% re-pitched acappellas).
- ``modulation_spectrum`` / ``modulation_ratio`` — inferior-colliculus neurons
  tuned to modulation *rate*: a tempo/rhythm-envelope analyzer that yields a
  stretch/tempo-ratio estimate as a first-class observation.

Precisions are UNVALIDATED until measured against GT (see ActionSpec.validated
in actions.py) — do not let the ladder auto-commit on these until an eval pass
calibrates P(correct | probe fired).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Envelope frame rate for onset/modulation work. librosa's default hop of 512 at
# 22.05 kHz gives ~43 Hz; we resample everything onto this grid so cross-probe
# lags are directly comparable in seconds.
ENV_FPS = 43.06640625  # 22050 / 512


# ---------------------------------------------------------------------------
# Onset detection (cochlear-nucleus onset cells) + onset-alignment placement
# ---------------------------------------------------------------------------


def onset_envelope(spec: np.ndarray, *, log_compress: bool = True) -> np.ndarray:
    """Spectral-flux onset-strength envelope from a magnitude spectrogram.

    ``spec`` is (freq_bins, frames), magnitude. Half-wave-rectified first
    difference summed over frequency — the standard onset-strength function and
    a faithful model of onset-cell response (energy increases only). Returns a
    1-D envelope, one value per frame, peak-normalized to [0, 1].
    """
    if spec.ndim != 2 or spec.shape[1] < 2:
        return np.zeros(max(spec.shape[-1], 0), dtype=float)
    s = np.log1p(spec) if log_compress else spec
    flux = np.diff(s, axis=1)
    flux = np.maximum(flux, 0.0).sum(axis=0)  # half-wave rectify, sum over freq
    env = np.concatenate([[0.0], flux])  # align length to frame count
    peak = float(env.max())
    return env / peak if peak > 0 else env


@dataclass(frozen=True)
class OnsetAlign:
    lag_s: float  # ref onset grid leads mix by this many seconds at best match
    peak: float  # normalized cross-correlation at the lag, [0, 1]
    sharpness: float  # peak / second-best, >1 = a distinct alignment


def onset_align(
    mix_env: np.ndarray,
    ref_env: np.ndarray,
    *,
    fps: float = ENV_FPS,
    max_lag_s: float | None = None,
) -> OnsetAlign | None:
    """Localize ``ref_env`` inside ``mix_env`` by onset-envelope cross-correlation.

    Returns the lag (seconds into the mix where the ref's onset pattern best
    lands), a normalized correlation peak, and a sharpness ratio (peak vs the
    next-highest non-adjacent peak — the "is this a real alignment or noise"
    signal). ``None`` if either envelope is degenerate.
    """
    m = _zscore(mix_env)
    r = _zscore(ref_env)
    if m is None or r is None or r.size > m.size:
        return None
    # 'valid' correlation of the shorter ref sliding over the longer mix.
    corr = np.correlate(m, r, mode="valid")
    denom = float(np.sqrt((r * r).sum()) * np.sqrt((r * r).sum()))
    if denom <= 0:
        return None
    corr = corr / denom
    if max_lag_s is not None:
        corr = corr[: int(max_lag_s * fps) + 1]
        if corr.size == 0:
            return None
    best = int(np.argmax(corr))
    peak = float(corr[best])
    sharp = _peak_sharpness(corr, best, guard=int(0.5 * fps))
    return OnsetAlign(lag_s=best / fps, peak=peak, sharpness=sharp)


# ---------------------------------------------------------------------------
# Onset asynchrony (source-count / "one object or two")
# ---------------------------------------------------------------------------


def onset_asynchrony(
    low_env: np.ndarray,
    high_env: np.ndarray,
    *,
    fps: float = ENV_FPS,
    tol_ms: float = 30.0,
    min_strength: float = 0.15,
) -> float:
    """Fraction of onsets whose low-band and high-band arrivals disagree by
    >``tol_ms`` — an estimate of "two source streams here, not one".

    Partials that start together fuse (one source); a component onsetting
    >~30 ms late segregates (a second source — e.g. a vocal overlay over a bed).
    Score in [0, 1]: 0 = perfectly synchronous (single object), high = a second,
    time-shifted onset stream is present. Returns 0.0 when neither band has
    usable onsets.
    """
    lo = _peaks(low_env, min_strength)
    hi = _peaks(high_env, min_strength)
    if lo.size == 0 or hi.size == 0:
        return 0.0
    tol = tol_ms / 1000.0 * fps
    asynchronous = 0
    for p in lo:
        nearest = np.min(np.abs(hi - p))
        if nearest > tol:
            asynchronous += 1
    return asynchronous / lo.size


# ---------------------------------------------------------------------------
# Harmonic sieve (periodicity coding — pitch-invariant identity)
# ---------------------------------------------------------------------------


def harmonic_salience(
    spec: np.ndarray,
    *,
    sr: float,
    n_harmonics: int = 5,
    f0_bins: tuple[int, int] = (2, 60),
) -> np.ndarray:
    """Harmonic-product-spectrum salience per frame (f0 strength, key-invariant).

    For each frame, a harmonic sieve multiplies the spectrum by its
    down-sampled copies (2f, 3f, …) so energy survives only at true f0s. Returns
    a per-frame f0-salience vector summarizing periodicity strength — a
    structure cue that ignores absolute pitch, so it holds under DJ re-pitching
    where chroma's key sensitivity fails. ``spec`` is (freq, frames) magnitude.
    """
    if spec.ndim != 2 or spec.size == 0:
        return np.zeros(spec.shape[-1] if spec.ndim == 2 else 0, dtype=float)
    freq_bins, n_frames = spec.shape
    lo, hi = f0_bins
    hi = min(hi, freq_bins // n_harmonics)
    if hi <= lo:
        return np.zeros(n_frames, dtype=float)
    out = np.zeros(n_frames, dtype=float)
    for t in range(n_frames):
        col = spec[:, t]
        hps = np.ones(hi, dtype=float)
        for h in range(1, n_harmonics + 1):
            ds = col[::h][:hi] if h > 1 else col[:hi]
            hps[: ds.size] *= ds + 1e-9
        band = hps[lo:hi]
        out[t] = float(band.max()) if band.size else 0.0
    peak = float(out.max())
    return out / peak if peak > 0 else out


def harmonic_sieve_match(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two f0-salience series (pitch-invariant identity)."""
    return _cosine(a, b)


# ---------------------------------------------------------------------------
# Modulation spectrum (IC modulation-rate tuning → tempo / stretch)
# ---------------------------------------------------------------------------


def modulation_spectrum(
    env: np.ndarray,
    *,
    fps: float = ENV_FPS,
    max_rate_hz: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude spectrum of the amplitude envelope, up to ``max_rate_hz``.

    The rhythm/tempo periodicity content of a signal — the IC's modulation-rate
    analysis. Returns (rates_hz, magnitude). A beat at 2 Hz (120 BPM quarter
    notes) shows as a peak near 2 Hz.
    """
    e = env - env.mean() if env.size else env
    if e.size < 4:
        return np.zeros(0), np.zeros(0)
    mag = np.abs(np.fft.rfft(e))
    rates = np.fft.rfftfreq(e.size, d=1.0 / fps)
    keep = rates <= max_rate_hz
    return rates[keep], mag[keep]


def modulation_ratio(
    mix_env: np.ndarray,
    ref_env: np.ndarray,
    *,
    fps: float = ENV_FPS,
) -> float | None:
    """Estimate ref→mix time-stretch (tempo ratio) from dominant modulation rates.

    stretch = ref_rate / mix_rate: if the mix version is sped up, its dominant
    modulation rate is higher and the ratio < 1. Returns None if either lacks a
    clear modulation peak.
    """
    rm, mm = modulation_spectrum(mix_env, fps=fps)
    rr, mr = modulation_spectrum(ref_env, fps=fps)
    fm, fr = _dominant_rate(rm, mm), _dominant_rate(rr, mr)
    if fm is None or fr is None or fm <= 0:
        return None
    return fr / fm


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _zscore(x: np.ndarray) -> np.ndarray | None:
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return None
    sd = x.std()
    if sd <= 0:
        return None
    return (x - x.mean()) / sd


def _peak_sharpness(corr: np.ndarray, best: int, *, guard: int) -> float:
    if corr.size < 2:
        return 1.0
    masked = corr.copy()
    lo, hi = max(0, best - guard), min(corr.size, best + guard + 1)
    masked[lo:hi] = -np.inf
    second = float(masked.max()) if np.isfinite(masked).any() else 0.0
    peak = float(corr[best])
    if second <= 0:
        return float("inf") if peak > 0 else 1.0
    return peak / second


def _peaks(env: np.ndarray, min_strength: float) -> np.ndarray:
    env = np.asarray(env, dtype=float)
    if env.size < 3:
        return np.array([], dtype=int)
    thr = min_strength * (env.max() if env.max() > 0 else 1.0)
    up = (env[1:-1] > env[:-2]) & (env[1:-1] >= env[2:]) & (env[1:-1] >= thr)
    return np.nonzero(up)[0] + 1


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = min(a.size, b.size)
    if n == 0:
        return 0.0
    a, b = a[:n], b[:n]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _dominant_rate(rates: np.ndarray, mag: np.ndarray) -> float | None:
    if rates.size == 0:
        return None
    # ignore the DC/very-slow bins below 0.3 Hz (drift, not rhythm)
    band = rates >= 0.3
    if not band.any():
        return None
    r, m = rates[band], mag[band]
    return float(r[int(np.argmax(m))])
