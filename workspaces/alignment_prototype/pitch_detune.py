"""Calibrated micro-pitch (cents) estimator for DJ-mix detunes.

Measures how far a reference clip is pitched *in the mix* relative to its
native recording, to sub-cent precision (100 cents = 1 semitone). This is the
signal behind the ``PitchFine``/``PitchCoarse`` a labeler dials in Ableton to
make an acappella/instrumental sit in tune with the mix.

Method (validated to ~sub-cent by synthetic recovery in
``tests/test_pitch_detune.py``):

1. Take the time-averaged magnitude spectrum of the ref clip and of its aligned
   mix span, resampled onto a **log-frequency (cents) grid**. A pitch shift of
   the whole track is a rigid *translation* along that axis, identical for every
   harmonic, so averaging over time (which discards note identity and tolerates
   sub-second alignment error) preserves it.
2. **Cross-correlate** the two averaged spectra along the cents axis. The
   correlation is ``sum_f Ref(f)·Mix(f+lag)``, so it is automatically weighted
   toward bins where the *ref* has energy — bands where a co-playing track
   dominates but the ref is quiet contribute ~nothing. This is what makes the
   estimate survive layered sections.
3. Peak lag (parabolic sub-bin interpolation) = ``cents_agg`` = mix pitch minus
   ref-native pitch. Positive ⇒ the mix plays the track *sharp* of its native.

CRITICAL correctness rule (kept fixed here): the sub-semitone signal is the
**residual after removing the dialed coarse transpose**:
``residual = cents_agg - 100·pitch_coarse``. A raw estimate reads ±100¢ on every
±1-semitone clip and swamps the real detune. The search window is *centered* on
``100·pitch_coarse`` so it never saturates, whatever the coarse value.

Pure functions; audio I/O at the edges. No project imports — belongs to the
alignment prototype and is consumed by ``eda/corpus_empirics/bb_pitch_detune.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

SR = 22050
N_FFT = 8192
HOP = 2048
FMIN = 80.0  # Hz — below this is kick/bass from co-playing tracks, not the comb
FMAX = 8000.0
VOCAL_FMIN = 200.0  # vocal band: excludes the bed's bass mass so a buried
VOCAL_FMAX = 3500.0  # acappella's comb isn't dragged down by the instrumental
CENTS_PER_BIN = 2.0
MAX_CENTS = 300.0  # half-width of the search window around 100·coarse (±3 semitones:
# wide enough that a real key-transpose+detune never clips; a peak pinned at the
# edge is flagged ``saturated`` — an octave/wrong-ref artifact, not a micro-detune)


# ---------------------------------------------------------------------------
# audio → averaged log-frequency spectrum
# ---------------------------------------------------------------------------


def load_span(path: str, start_s: float, end_s: float, *, sr: int = SR) -> np.ndarray:
    """Mono audio for ``[start_s, end_s]`` of ``path`` (empty if degenerate)."""
    dur = end_s - start_s
    if dur <= 0:
        return np.zeros(0, dtype=np.float32)
    y, _ = librosa.load(path, sr=sr, mono=True, offset=max(0.0, start_s), duration=dur)
    return y.astype(np.float32)


def _cents_grid(
    fmin: float = FMIN, fmax: float = FMAX, cents_per_bin: float = CENTS_PER_BIN
) -> np.ndarray:
    """Log-frequency grid in Hz, uniform in cents."""
    total_cents = 1200.0 * np.log2(fmax / fmin)
    n = int(round(total_cents / cents_per_bin)) + 1
    return fmin * 2.0 ** (np.arange(n) * cents_per_bin / 1200.0)


def avg_logfreq_spectrum(
    y: np.ndarray,
    *,
    sr: int = SR,
    n_fft: int = N_FFT,
    hop: int = HOP,
    fmin: float = FMIN,
    fmax: float = FMAX,
    cents_per_bin: float = CENTS_PER_BIN,
) -> np.ndarray:
    """Time-averaged magnitude spectrum resampled onto the cents grid.

    Returned vector is log-compressed and z-normalised (zero mean, unit L2) so
    the downstream cross-correlation is a Pearson-style correlation robust to
    overall level. All-silence input returns zeros.
    """
    grid = _cents_grid(fmin, fmax, cents_per_bin)
    if y.size < n_fft:
        y = np.pad(y, (0, n_fft - y.size))
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann"))
    mag = S.mean(axis=1)  # time-average, linear freq
    lin_f = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    spec = np.interp(grid, lin_f, mag, left=0.0, right=0.0)
    spec = np.log1p(spec)  # tame dynamic range so no single band dominates
    spec = spec - spec.mean()
    nrm = np.linalg.norm(spec)
    return spec / nrm if nrm > 0 else spec


# ---------------------------------------------------------------------------
# cross-correlation along the cents axis
# ---------------------------------------------------------------------------


def _parabolic_peak(corr: np.ndarray, i: int) -> float:
    """Sub-bin peak offset (in bins) from a 3-point parabola around index ``i``."""
    if i <= 0 or i >= corr.size - 1:
        return float(i)
    a, b, c = corr[i - 1], corr[i], corr[i + 1]
    denom = a - 2 * b + c
    if denom == 0:
        return float(i)
    return i + 0.5 * (a - c) / denom


@dataclass(frozen=True)
class CentsResult:
    cents_agg: float  # mix pitch − ref-native pitch (total, includes coarse)
    residual: float  # cents_agg − 100·pitch_coarse (the sub-semitone detune)
    peak_corr: float  # correlation at the peak, 0..1 — a confidence proxy
    ok: bool  # both spectra had energy
    saturated: bool = False  # peak pinned at the search-window edge → unreliable


def _xcorr(
    ref_spec: np.ndarray, mix_spec: np.ndarray, center_bins: int, half_bins: int
) -> tuple[float, float]:
    """Peak (cents-lag-in-bins, corr) of ref vs shifted mix over a window.

    lag>0 means mix's comb sits ``lag`` bins *higher* than ref's, i.e. the mix
    plays the track sharp. Overlap-only dot product at each integer lag, then a
    parabolic refine of the argmax.
    """
    n = ref_spec.size
    lags = range(center_bins - half_bins, center_bins + half_bins + 1)
    corr = np.empty(len(lags), dtype=np.float64)
    for k, lag in enumerate(lags):
        if lag >= 0:
            a, b = ref_spec[: n - lag], mix_spec[lag:]
        else:
            a, b = ref_spec[-lag:], mix_spec[: n + lag]
        corr[k] = float(np.dot(a, b)) if a.size else -1.0
    i = int(np.argmax(corr))
    peak_bin = _parabolic_peak(corr, i)
    lag_bins = (center_bins - half_bins) + peak_bin
    return lag_bins, float(corr[i])


def estimate_cents(
    ref_y: np.ndarray,
    mix_y: np.ndarray,
    *,
    pitch_coarse: int = 0,
    sr: int = SR,
    cents_per_bin: float = CENTS_PER_BIN,
    max_cents: float = MAX_CENTS,
    fmin: float = FMIN,
    fmax: float = FMAX,
) -> CentsResult:
    """Measure mix-vs-ref pitch offset, centring the search on the dialed coarse.

    ``ref_y`` is the *raw* reference audio (no pitch applied); ``mix_y`` the
    aligned mix span. Passing ``pitch_coarse`` centres the ±``max_cents`` window
    on ``100·coarse`` so a whole-step transpose never saturates the window and
    the returned ``residual`` isolates the sub-semitone detune.

    ``fmin``/``fmax`` bound the correlated band. Default is full-range, correct
    when the ref is a large fraction of the mix energy (instrumentals, solo). For
    an acappella buried under a bass-heavy bed, pass the **vocal band**
    (``fmin≈200, fmax≈3500``): the full band otherwise locks the cross-correlation
    onto the bed's low-frequency mass and reports a spurious downward offset
    (validated: Drake/MIMS/Roy Orbison read −140…−290¢ full-band but ~0¢ and
    *higher* corr on the vocal band).
    """
    ref_spec = avg_logfreq_spectrum(
        ref_y, sr=sr, cents_per_bin=cents_per_bin, fmin=fmin, fmax=fmax
    )
    mix_spec = avg_logfreq_spectrum(
        mix_y, sr=sr, cents_per_bin=cents_per_bin, fmin=fmin, fmax=fmax
    )
    if np.linalg.norm(ref_spec) == 0 or np.linalg.norm(mix_spec) == 0:
        return CentsResult(float("nan"), float("nan"), 0.0, ok=False)
    center_bins = int(round(100.0 * pitch_coarse / cents_per_bin))
    half_bins = int(round(max_cents / cents_per_bin))
    lag_bins, peak = _xcorr(ref_spec, mix_spec, center_bins, half_bins)
    cents_agg = lag_bins * cents_per_bin
    saturated = abs(lag_bins - center_bins) >= half_bins - 1
    return CentsResult(
        cents_agg=cents_agg,
        residual=cents_agg - 100.0 * pitch_coarse,
        peak_corr=peak,
        ok=True,
        saturated=saturated,
    )


# ---------------------------------------------------------------------------
# synthetic helper (used by the recovery test)
# ---------------------------------------------------------------------------


def pitch_shift_cents(y: np.ndarray, cents: float, *, sr: int = SR) -> np.ndarray:
    """Phase-vocoder pitch shift by ``cents`` (keylock-style, length-preserving)."""
    return librosa.effects.pitch_shift(
        y, sr=sr, n_steps=cents, bins_per_octave=1200
    ).astype(np.float32)
