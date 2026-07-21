from __future__ import annotations

import numpy as np

from workspaces.pws_aligner.fx_lf import detect_noise_sweep, noise_sweep_vote
from workspaces.pws_aligner.votes import AbstainReason

_SR = 22050


def _low_noise(n: int, seed: int = 1) -> np.ndarray:
    """Low-frequency-heavy noise (brown-ish): cumulative sum of white noise."""
    w = np.random.RandomState(seed).randn(n)
    y = np.cumsum(w)
    y = y - y.mean()
    return (y / (np.abs(y).max() + 1e-9)).astype(np.float32)


def _high_noise(n: int, seed: int = 2) -> np.ndarray:
    """High-frequency-heavy noise (blue-ish): first difference of white noise."""
    w = np.random.RandomState(seed).randn(n)
    y = np.diff(w, prepend=w[0])
    return (y / (np.abs(y).max() + 1e-9)).astype(np.float32)


def _riser(seconds: float = 2.0) -> np.ndarray:
    """A noise sweep: broadband noise whose spectral centroid climbs low->high."""
    n = int(seconds * _SR)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    y = (1.0 - t) * _low_noise(n) + t * _high_noise(n)
    # crescendo into the drop
    return (y * (0.3 + 0.7 * t)).astype(np.float32)


def test_riser_detected():
    has, slope, flat = detect_noise_sweep(_riser(), _SR)
    assert has
    assert slope > 0.0


def test_steady_tone_not_swept():
    # a pure tone: tonal (low flatness), no centroid rise
    n = int(2.0 * _SR)
    t = np.arange(n) / _SR
    tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    has, _, flat = detect_noise_sweep(tone, _SR)
    assert not has


def test_steady_noise_not_swept():
    # broadband noise but NO centroid rise -> not a sweep
    n = int(2.0 * _SR)
    steady = _high_noise(n)
    has, slope, _ = detect_noise_sweep(steady, _SR)
    assert not has


def test_vote_fires_on_riser():
    v = noise_sweep_vote("span_1", _riser(), _SR)
    assert v.lf == "noise_sweep"
    assert v.label == "noise_sweep"
    assert not v.abstained
    assert 0.0 <= v.confidence <= 1.0


def test_vote_abstains_on_tone():
    n = int(2.0 * _SR)
    t = np.arange(n) / _SR
    tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    v = noise_sweep_vote("span_1", tone, _SR)
    assert v.abstained
    assert v.label == "none"
    assert v.reason == AbstainReason.LOW_MARGIN
