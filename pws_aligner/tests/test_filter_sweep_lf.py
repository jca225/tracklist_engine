from __future__ import annotations

"""Tests for the filter_sweep labeling function (A2).

A filter sweep keeps tonal content (low spectral flatness) but moves the cutoff
monotonically (rising spectral centroid). This is distinct from noise_sweep
(broadband noise + rising centroid — high flatness).

Positive: a pure tone amplitude-shaped by a rising lowpass (only low freqs audible
at first, then progressively more — centroid rises, flatness stays low).
Negative (steady tone): no centroid movement.
Negative (broadband riser): high flatness => noise_sweep, NOT filter_sweep.
"""

import numpy as np

from pws_aligner.fx_lf import detect_filter_sweep, filter_sweep_vote
from pws_aligner.votes import AbstainReason

_SR = 22050


def _filter_swept_tone(duration_s: float = 2.0) -> np.ndarray:
    """Additive synthesis: partials up to 8 kHz, amplitude envelope rising from
    low to high partials over time — simulates a rising LPF sweep on a tonal signal.
    Spectral centroid rises monotonically; flatness stays low (tonal, not broadband).
    """
    n = int(duration_s * _SR)
    t = np.arange(n, dtype=np.float32) / _SR
    # time-varying gain for each harmonic: gates in from low to high over [0, duration_s]
    freqs = [110, 220, 440, 880, 1760, 3520, 7040]
    y = np.zeros(n, dtype=np.float32)
    for i, f in enumerate(freqs):
        # partial i gate-in: starts audible at t = i/len(freqs) * duration_s
        gate_time = (i / len(freqs)) * duration_s
        env = np.clip((t - gate_time) / (duration_s * 0.15), 0.0, 1.0)
        y += env * np.sin(2 * np.pi * f * t)
    # normalize
    peak = float(np.abs(y).max())
    if peak > 0:
        y /= peak
    return y


def _steady_tone(duration_s: float = 2.0) -> np.ndarray:
    """A constant pure tone — no centroid change."""
    n = int(duration_s * _SR)
    t = np.arange(n, dtype=np.float32) / _SR
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


def _broadband_riser(duration_s: float = 2.0) -> np.ndarray:
    """Broadband noise riser — high flatness + rising centroid (noise_sweep, not filter_sweep)."""
    n = int(duration_s * _SR)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    rng = np.random.RandomState(3)
    # low-freq heavy noise grows toward high-freq heavy noise
    low_n = np.cumsum(rng.randn(n))
    low_n = (low_n - low_n.mean()) / (np.abs(low_n).max() + 1e-9)
    high_n = np.diff(rng.randn(n + 1)).astype(np.float32)
    high_n = (high_n - high_n.mean()) / (np.abs(high_n).max() + 1e-9)
    y = ((1 - t) * low_n + t * high_n) * (0.3 + 0.7 * t)
    return y.astype(np.float32)


def test_filter_swept_tone_detected():
    """Rising tonal filter sweep must fire."""
    y = _filter_swept_tone()
    has, slope, flat = detect_filter_sweep(y, _SR)
    assert has, f"expected filter_sweep detected, slope={slope:.4f} flat={flat:.4f}"
    assert slope > 0.0


def test_steady_tone_not_swept():
    """A constant tone has no centroid rise — must not fire."""
    y = _steady_tone()
    has, slope, flat = detect_filter_sweep(y, _SR)
    assert not has, f"expected no filter_sweep, slope={slope:.4f} flat={flat:.4f}"


def test_broadband_riser_not_filter_sweep():
    """A noise riser is noise_sweep, NOT filter_sweep (high flatness rejects it)."""
    y = _broadband_riser()
    has, slope, flat = detect_filter_sweep(y, _SR)
    assert not has, f"broadband riser must not be filter_sweep, flat={flat:.4f}"


def test_vote_fires_on_filter_sweep():
    y = _filter_swept_tone()
    v = filter_sweep_vote("span_1", y, _SR)
    assert v.lf == "filter_sweep"
    assert v.label == "filter_sweep"
    assert not v.abstained
    assert 0.0 <= v.confidence <= 1.0


def test_vote_abstains_on_steady_tone():
    y = _steady_tone()
    v = filter_sweep_vote("span_1", y, _SR)
    assert v.abstained
    assert v.label == "none"
    assert v.reason == AbstainReason.LOW_MARGIN


def test_vote_abstains_on_broadband_riser():
    """Broadband riser must abstain from filter_sweep (not a false positive)."""
    y = _broadband_riser()
    v = filter_sweep_vote("span_1", y, _SR)
    assert v.abstained
