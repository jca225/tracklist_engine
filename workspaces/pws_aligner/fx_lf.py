from __future__ import annotations

"""FX-family operation labeling functions for the PWS aligner (lane 2).

Companion to operations.py (which holds the tempo/pitch keylock-vs-varispeed LF).
This module holds the FX/transition family — starting with echo/delay-throw, the
most common DJ transition effect.

Detector grounding (docs/dj_operation_ontology_research.md, LF B1/B2):
an echo/delay throw produces **beat-synced decaying repeats** at lags that are
rational fractions/multiples of the beat period. The tell is in the amplitude
*envelope*, not the carrier: a single dry hit has one envelope bump; an echo tail
has a decaying train of bumps spaced by the echo delay. So we autocorrelate the
onset-strength envelope and look for a peak at a beat-rational lag — a sustained
tone is self-similar at every lag in the raw signal, but its envelope is flat, so
enveloped autocorrelation is what separates echo from steady material.

This is a first detector: presence/absence of a beat-synced repeat structure,
not a full (delay-time, feedback, wet/dry) parameterization. Emit an OperationVote
so the categorical operation-lane aggregator can consume it alongside
keylock/varispeed.
"""

import librosa
import numpy as np

from .operations import OperationVote
from .votes import AbstainReason

# Beat-rational lags an echo delay typically lands on (fractions/multiples of a beat).
_BEAT_FRACTIONS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
# Normalized-autocorrelation height at a beat-rational lag above which we call it echo.
_ECHO_PEAK_THRESH: float = 0.25
_HOP: int = 512

# Noise-sweep / riser thresholds. A riser is broadband (high spectral flatness)
# with a rising spectral centroid (brightness sweeps up into the drop).
_SWEEP_FLATNESS_MIN: float = (
    0.10  # mean spectral flatness; tonal material is far below this
)
_SWEEP_CENTROID_SLOPE_MIN: float = (
    0.15  # normalized centroid rise (fraction of Nyquist per window)
)


def echo_envelope(y: np.ndarray, sr: int) -> np.ndarray:
    """Onset-strength envelope (the amplitude structure echo repeats live in)."""
    return librosa.onset.onset_strength(y=y, sr=sr, hop_length=_HOP)


def detect_echo_tail(
    y: np.ndarray, sr: int, beat_period_s: float
) -> tuple[bool, float]:
    """Detect a beat-synced decaying repeat structure (echo tail).

    Returns (has_echo, score) where score is the normalized-autocorrelation height
    at the best beat-rational lag. A single dry hit yields an envelope whose
    autocorrelation decays with no secondary peak; an echo tail yields a peak at
    the echo delay (a beat-rational lag).
    """
    env = echo_envelope(y, sr)
    if env.size < 4 or float(env.max()) <= 0.0:
        return False, 0.0
    env = env - float(env.mean())
    ac = librosa.autocorrelate(env)
    ac0 = float(ac[0])
    if ac0 <= 0.0:
        return False, 0.0
    ac = ac / ac0  # normalize so lag-0 == 1.0

    frames_per_s = sr / _HOP
    best = 0.0
    for frac in _BEAT_FRACTIONS:
        lag = int(round(beat_period_s * frac * frames_per_s))
        if lag < 1 or lag >= ac.size:
            continue
        # Local-max guard: a genuine repeat is a *peak*, not a point on a slope.
        lo = max(1, lag - 1)
        hi = min(ac.size - 1, lag + 1)
        val = float(ac[lag])
        if val >= float(ac[lo]) and val >= float(ac[hi]):
            best = max(best, val)
    return best > _ECHO_PEAK_THRESH, best


def echo_tail_vote(
    span_id: str, y: np.ndarray, sr: int, beat_period_s: float | None
) -> OperationVote:
    """Emit an OperationVote for the echo-tail LF over a span's audio."""
    lf = "echo_tail"
    if beat_period_s is None or beat_period_s <= 0.0:
        return OperationVote(lf, span_id, "none", 0.0, True, AbstainReason.NO_DATA)
    has, score = detect_echo_tail(y, sr, beat_period_s)
    if has:
        conf = float(min(1.0, score))
        return OperationVote(lf, span_id, "echo", conf, False, AbstainReason.NONE)
    return OperationVote(lf, span_id, "none", 0.0, True, AbstainReason.LOW_MARGIN)


def detect_noise_sweep(y: np.ndarray, sr: int) -> tuple[bool, float, float]:
    """Detect a noise sweep / riser: broadband noise with a rising centroid.

    Returns (has_sweep, centroid_slope, mean_flatness). The two tells:
    - **mean spectral flatness** high => broadband noise (a tonal riser or a
      melodic line has low flatness and is excluded);
    - **spectral-centroid slope** positive and large => brightness sweeps up,
      the signature of a riser climbing into a downbeat drop.
    Both must hold; steady noise (flat centroid) and tonal sweeps are rejected.
    Slope is normalized to fraction-of-Nyquist across the whole window.
    """
    if y.size < _HOP * 4:
        return False, 0.0, 0.0
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=_HOP)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=_HOP)[0]
    if centroid.size < 4:
        return False, 0.0, 0.0
    mean_flat = float(np.mean(flatness))
    # Normalized centroid trajectory (fraction of Nyquist), least-squares slope
    # expressed as total rise across the window.
    nyq = sr / 2.0
    c = centroid / nyq
    x = np.linspace(0.0, 1.0, c.size)
    slope = float(np.polyfit(x, c, 1)[0])  # rise over the [0,1] window
    has = slope >= _SWEEP_CENTROID_SLOPE_MIN and mean_flat >= _SWEEP_FLATNESS_MIN
    return has, slope, mean_flat


def noise_sweep_vote(span_id: str, y: np.ndarray, sr: int) -> OperationVote:
    """Emit an OperationVote for the noise-sweep/riser LF over a span's audio."""
    lf = "noise_sweep"
    has, slope, flat = detect_noise_sweep(y, sr)
    if has:
        # Confidence blends how strongly both tells fire (bounded to [0,1]).
        conf = float(min(1.0, 0.5 * min(1.0, slope / 0.5) + 0.5 * min(1.0, flat / 0.3)))
        return OperationVote(
            lf, span_id, "noise_sweep", conf, False, AbstainReason.NONE
        )
    return OperationVote(lf, span_id, "none", 0.0, True, AbstainReason.LOW_MARGIN)


# ---------------------------------------------------------------------------
# A2 — filter_sweep
# ---------------------------------------------------------------------------
# A filter sweep (LPF/HPF cutoff sweeping) keeps the programme's tonal content —
# low spectral flatness — while moving the spectral centroid monotonically.
# This is distinct from noise_sweep (noise_sweep REQUIRES high flatness).
# Two tells:
#   1. monotonic spectral-centroid rise (same as noise_sweep) — the cutoff moves
#   2. LOW spectral flatness — the source is tonal, not broadband noise
# Monotonicity is measured with a signed Spearman correlation of the centroid
# trajectory (a pure rise → r≈+1; a plateau → r≈0).

_FILTER_CENTROID_SLOPE_MIN: float = (
    0.12  # total normalized centroid rise (fraction of Nyquist) over window
)
_FILTER_FLATNESS_MAX: float = (
    0.05  # mean spectral flatness must be BELOW this to be tonal (not noise)
)
_FILTER_MONOTONE_R_MIN: float = (
    0.65  # Spearman r of centroid trajectory vs time must exceed this
)


def detect_filter_sweep(y: np.ndarray, sr: int) -> tuple[bool, float, float]:
    """Detect a tonal filter sweep: rising centroid + LOW spectral flatness.

    Returns (has_sweep, centroid_slope, mean_flatness).
    A filter sweep distinguishes from noise_sweep by LOW flatness: the source
    material is tonal (a chord, bassline, etc.) filtered through a moving cutoff.
    A broadband noise riser fails the flatness gate (too high).
    A steady tone fails the slope gate (no centroid movement).
    Monotonicity is tested via Spearman correlation to reject noisy trajectories
    where the centroid bounces rather than climbs steadily.
    """
    if y.size < _HOP * 4:
        return False, 0.0, 0.0
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=_HOP)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=_HOP)[0]
    if centroid.size < 4:
        return False, 0.0, 0.0
    mean_flat = float(np.mean(flatness))
    # Tonal gate: reject if broadband (would be noise_sweep territory)
    if mean_flat >= _FILTER_FLATNESS_MAX:
        return False, 0.0, mean_flat
    # Centroid slope (same normalization as noise_sweep)
    nyq = sr / 2.0
    c = centroid / nyq
    x = np.linspace(0.0, 1.0, c.size)
    slope = float(np.polyfit(x, c, 1)[0])
    if slope < _FILTER_CENTROID_SLOPE_MIN:
        return False, slope, mean_flat
    # Monotonicity: Spearman correlation of centroid vs time index
    ranks_t = np.argsort(np.argsort(np.arange(c.size))).astype(np.float64)
    ranks_c = np.argsort(np.argsort(c)).astype(np.float64)
    r = float(np.corrcoef(ranks_t, ranks_c)[0, 1])
    has = r >= _FILTER_MONOTONE_R_MIN
    return has, slope, mean_flat


def filter_sweep_vote(span_id: str, y: np.ndarray, sr: int) -> OperationVote:
    """Emit an OperationVote for the filter-sweep LF over a span's audio."""
    lf = "filter_sweep"
    has, slope, flat = detect_filter_sweep(y, sr)
    if has:
        conf = float(
            min(
                1.0,
                0.5 * min(1.0, slope / 0.3)
                + 0.5 * min(1.0, (1.0 - flat / _FILTER_FLATNESS_MAX)),
            )
        )
        return OperationVote(
            lf, span_id, "filter_sweep", conf, False, AbstainReason.NONE
        )
    return OperationVote(lf, span_id, "none", 0.0, True, AbstainReason.LOW_MARGIN)


# ---------------------------------------------------------------------------
# A1 — loop_periodicity
# ---------------------------------------------------------------------------
# A looping span tiles a short segment, so the RAW signal has near-perfect
# autocorrelation at lags equal to the tile length (i.e. beat multiples).
# Contrast: echo_tail uses the *onset-strength envelope* (decaying repeats);
# loop_periodicity uses the *raw signal* (perfect repeats of the full waveform).
# We look for a high normalized autocorrelation peak at any beat-multiple lag
# from 1× to 4× the beat period, allowing for 2-bar / 4-bar loop lengths too.

_LOOP_AC_THRESH: float = 0.65  # normalized AC height to call it a loop
_LOOP_BEAT_MULTIPLES: tuple[int, ...] = (1, 2, 4)  # 1-bar, 2-bar, 4-bar loops


def detect_loop_periodicity(
    y: np.ndarray, sr: int, beat_period_s: float
) -> tuple[bool, float]:
    """Detect a repeating loop by raw-signal autocorrelation at beat-multiple lags.

    Unlike ``detect_echo_tail`` (envelope autocorrelation, decaying repeats),
    this operates on the raw signal — a loop repeats the *entire* waveform with
    near-perfect correlation, giving a very high AC peak at the tile lag.

    Returns (has_loop, score) where score is the normalized AC at the best lag.
    """
    if y.size < sr * 1 or beat_period_s <= 0.0:
        return False, 0.0

    # Normalize to avoid amplitude dominating the correlation.
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    if rms < 1e-8:
        return False, 0.0
    yn = (y / rms).astype(np.float64)

    best = 0.0
    for mult in _LOOP_BEAT_MULTIPLES:
        lag_samples = int(round(beat_period_s * mult * sr))
        if lag_samples < 1 or lag_samples >= yn.size:
            continue
        # Pearson correlation between signal and its shifted copy.
        a = yn[lag_samples:]
        b = yn[: len(yn) - lag_samples]
        if a.size < 16:
            continue
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 0.0:
            continue
        r = float(np.dot(a, b) / denom)
        if r > best:
            best = r

    return best >= _LOOP_AC_THRESH, best


def loop_periodicity_vote(
    span_id: str, y: np.ndarray, sr: int, beat_period_s: float | None
) -> OperationVote:
    """Emit an OperationVote for the loop-periodicity LF over a span's audio."""
    lf = "loop"
    if beat_period_s is None or beat_period_s <= 0.0:
        return OperationVote(lf, span_id, "none", 0.0, True, AbstainReason.NO_DATA)
    has, score = detect_loop_periodicity(y, sr, beat_period_s)
    if has:
        conf = float(min(1.0, score))
        return OperationVote(lf, span_id, "loop", conf, False, AbstainReason.NONE)
    return OperationVote(lf, span_id, "none", 0.0, True, AbstainReason.LOW_MARGIN)


# ---------------------------------------------------------------------------
# A3 — sidechain_pump
# ---------------------------------------------------------------------------
# Sidechain compression (4-on-the-floor pump) produces periodic amplitude
# ducking on a sustained band, locked to the kick rate. The tell is in the
# RMS envelope of the signal: it has a strong periodic component at the beat
# frequency (one duck per beat). We detect this by:
#   1. Computing a short-time RMS envelope of the signal.
#   2. Taking the DFT of that envelope.
#   3. Checking whether there is a significant spectral peak at the beat
#      frequency (and its immediate neighbors for robustness).
# A flat-amplitude sustained tone has a flat RMS envelope → no beat-rate peak.
# A truly pumped signal has a large periodic dip → strong DFT peak.

_PUMP_RMS_HOP: int = 256  # short RMS hop for fine temporal resolution
_PUMP_RMS_WIN: int = 1024  # RMS window length
_PUMP_PEAK_THRESH: float = (
    0.20  # normalized DFT peak height at beat freq to call it pump
)


def detect_sidechain_pump(
    y: np.ndarray, sr: int, beat_period_s: float
) -> tuple[bool, float]:
    """Detect beat-rate amplitude modulation (sidechain pump).

    Computes a short-time RMS envelope, takes its DFT, and checks for a
    peak at the beat frequency. Returns (has_pump, score) where score is the
    relative spectral peak height at the beat frequency vs the mean spectrum.

    A flat-RMS sustained signal produces no beat-rate peak (abstain).
    A periodically ducked signal produces a strong peak (fire).
    """
    if y.size < sr or beat_period_s <= 0.0:
        return False, 0.0

    # Build short-time RMS envelope
    hop = _PUMP_RMS_HOP
    win = _PUMP_RMS_WIN
    n_frames = (y.size - win) // hop + 1
    if n_frames < 8:
        return False, 0.0
    env = np.array(
        [
            float(np.sqrt(np.mean(y[i * hop : i * hop + win].astype(np.float64) ** 2)))
            for i in range(n_frames)
        ],
        dtype=np.float64,
    )
    if float(env.max()) < 1e-9:
        return False, 0.0

    # DFT of the RMS envelope to find periodic modulation
    env_norm = env - float(env.mean())
    spectrum = np.abs(np.fft.rfft(env_norm))
    freqs_env = np.fft.rfftfreq(n_frames, d=hop / sr)  # frequency axis in Hz

    # Beat frequency in Hz
    beat_freq_hz = 1.0 / beat_period_s

    # Find the DFT bin closest to beat_freq_hz
    if freqs_env.size < 2:
        return False, 0.0
    beat_bin = int(np.argmin(np.abs(freqs_env - beat_freq_hz)))
    if beat_bin < 1 or beat_bin >= spectrum.size:
        return False, 0.0

    # Peak in a ±1-bin window around beat frequency
    lo = max(1, beat_bin - 1)
    hi = min(spectrum.size - 1, beat_bin + 1)
    peak = float(spectrum[lo : hi + 1].max())

    # Normalize by mean of rest of spectrum (excluding DC + beat region)
    mask = np.ones(spectrum.size, dtype=bool)
    mask[0] = False  # DC
    mask[lo : hi + 1] = False
    rest = spectrum[mask]
    mean_rest = float(rest.mean()) if rest.size > 0 else 1.0
    if mean_rest <= 0.0:
        return False, 0.0
    score = peak / mean_rest
    # Normalize score to [0,1] range (score > 1 means peak above mean noise floor)
    # Use a sigmoid-like normalization: score >= thresh means pump detected
    has = score >= (1.0 + _PUMP_PEAK_THRESH * 10)  # threshold in ratio units
    return has, float(min(1.0, score / 20.0))  # scale to roughly [0,1]


def sidechain_pump_vote(
    span_id: str, y: np.ndarray, sr: int, beat_period_s: float | None
) -> OperationVote:
    """Emit an OperationVote for the sidechain-pump LF over a span's audio."""
    lf = "sidechain"
    if beat_period_s is None or beat_period_s <= 0.0:
        return OperationVote(lf, span_id, "none", 0.0, True, AbstainReason.NO_DATA)
    has, score = detect_sidechain_pump(y, sr, beat_period_s)
    if has:
        conf = float(min(1.0, score))
        return OperationVote(lf, span_id, "sidechain", conf, False, AbstainReason.NONE)
    return OperationVote(lf, span_id, "none", 0.0, True, AbstainReason.LOW_MARGIN)
