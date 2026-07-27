"""Calibration tests for the micro-pitch cents estimator.

Self-contained: builds a synthetic voice-like harmonic signal, shifts it by a
known number of cents, and checks the estimator recovers the shift. No audio
files, so these survive independent of ~/aligning state.
"""

from __future__ import annotations

import numpy as np
import pytest

from alignment.pitch_detune import (
    SR,
    estimate_cents,
    pitch_shift_cents,
)


def _voice_like(sr: int = SR, dur_s: float = 6.0, seed: int = 0) -> np.ndarray:
    """A harmonic comb with time structure: a few sustained notes, ~10 harmonics,
    light vibrato and noise — enough spectral structure for the cross-correlation
    to lock, without being a single pure tone."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * dur_s)) / sr
    notes = [196.0, 220.0, 246.94, 174.61]  # G3 A3 B3 F3
    seg = len(t) // len(notes)
    f0 = np.empty_like(t)
    for i, f in enumerate(notes):
        lo, hi = i * seg, (i + 1) * seg if i < len(notes) - 1 else len(t)
        f0[lo:hi] = f
    vibrato = 1.0 + 0.004 * np.sin(2 * np.pi * 5.0 * t)
    phase = 2 * np.pi * np.cumsum(f0 * vibrato) / sr
    y = np.zeros_like(t)
    for h in range(1, 11):
        y += (1.0 / h) * np.sin(h * phase)
    y += 0.01 * rng.standard_normal(len(t))
    return (y / np.max(np.abs(y))).astype(np.float32)


def _resample_shift(y: np.ndarray, cents: float, sr: int = SR) -> np.ndarray:
    """Exact varispeed shift by ``cents`` (resample; length changes, which the
    time-averaged estimator is invariant to). Positive ⇒ pitched up."""
    import librosa

    factor = 2.0 ** (cents / 1200.0)  # play faster ⇒ higher pitch
    return librosa.resample(y, orig_sr=int(sr * factor), target_sr=sr).astype(
        np.float32
    )


def test_zero_point():
    """Identical ref and mix read ~0 cents."""
    y = _voice_like()
    r = estimate_cents(y, y.copy())
    assert r.ok
    assert abs(r.residual) < 1.0, r


@pytest.mark.parametrize("cents", [-80, -40, -20, -10, -3, 3, 10, 20, 40, 80])
def test_synthetic_recovery_varispeed(cents):
    """Exact resample shift recovers to within ~2 cents across ±80¢."""
    y = _voice_like()
    mix = _resample_shift(y, cents)
    r = estimate_cents(y, mix)
    assert r.ok
    assert abs(r.residual - cents) < 2.5, (cents, r)


def test_sign_convention():
    """Mix pitched UP relative to ref ⇒ positive cents."""
    y = _voice_like()
    r = estimate_cents(y, _resample_shift(y, +30))
    assert r.residual > 15.0, r


def test_coarse_residual_isolation():
    """A +1-semitone + small-detune mix, with coarse=1 dialed, yields the small
    residual (not ~100¢). Guards the residual = cents_agg − 100·coarse rule."""
    y = _voice_like()
    mix = _resample_shift(y, 100 + 18)  # +1 semitone plus +18¢
    r = estimate_cents(y, mix, pitch_coarse=1)
    assert abs(r.cents_agg - 118) < 3.0, r
    assert abs(r.residual - 18) < 3.0, r


def test_phase_vocoder_helper_recovers():
    """The keylock-style helper (used only for the synthetic-audio experiments)
    also recovers a known shift, looser tolerance for vocoder smear."""
    y = _voice_like()
    mix = pitch_shift_cents(y, 35.0)
    r = estimate_cents(y, mix)
    assert abs(r.residual - 35.0) < 7.0, r
