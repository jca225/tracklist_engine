from __future__ import annotations

"""Tests for the loop_periodicity labeling function (A1).

A looping span tiles a short segment N times; the raw signal has near-perfect
autocorrelation at a beat-multiple lag equal to the tile length. A through-composed
or noise-only signal has no such periodicity.
"""

import numpy as np

from pws_aligner.lf.fx import detect_loop_periodicity, loop_periodicity_vote
from pws_aligner.core.votes import AbstainReason

_SR = 22050


def _looped_signal(tile_s: float = 0.5, n_tiles: int = 8) -> np.ndarray:
    """Tile a short complex clip N times to produce a perfect loop."""
    rng = np.random.RandomState(42)
    tile_n = int(tile_s * _SR)
    tile = rng.randn(tile_n).astype(np.float32)
    # Add tonal content so it's not trivially silence
    t = np.arange(tile_n) / _SR
    tile = tile * 0.2 + np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return np.tile(tile, n_tiles)


def _nonrepeating_signal(duration_s: float = 4.0, seed: int = 7) -> np.ndarray:
    """White noise — no periodic structure at any lag."""
    rng = np.random.RandomState(seed)
    return rng.randn(int(duration_s * _SR)).astype(np.float32)


def _chirp_signal(duration_s: float = 4.0) -> np.ndarray:
    """Frequency sweep — through-composed, non-repeating."""
    n = int(duration_s * _SR)
    t = np.linspace(0.0, duration_s, n, dtype=np.float32)
    return np.sin(2 * np.pi * (100 + 2000 * t / duration_s) * t).astype(np.float32)


def test_looped_signal_detected():
    """A perfectly tiled loop must fire."""
    tile_s = 0.5
    y = _looped_signal(tile_s=tile_s, n_tiles=8)
    # beat period == one tile length
    beat_period_s = tile_s
    has, score = detect_loop_periodicity(y, _SR, beat_period_s)
    assert has, f"expected loop detected, score={score:.4f}"
    assert score > 0.5


def test_nonrepeating_abstains():
    """White noise must not be labelled as a loop."""
    y = _nonrepeating_signal(duration_s=4.0)
    has, score = detect_loop_periodicity(y, _SR, beat_period_s=0.5)
    assert not has, f"expected no loop, score={score:.4f}"


def test_chirp_abstains():
    """A smooth frequency sweep is not a loop."""
    y = _chirp_signal(duration_s=4.0)
    has, score = detect_loop_periodicity(y, _SR, beat_period_s=0.5)
    assert not has, f"expected no loop for chirp, score={score:.4f}"


def test_vote_fires_on_loop():
    tile_s = 0.5
    y = _looped_signal(tile_s=tile_s, n_tiles=8)
    v = loop_periodicity_vote("span_1", y, _SR, beat_period_s=tile_s)
    assert v.lf == "loop"
    assert v.label == "loop"
    assert not v.abstained
    assert 0.0 <= v.confidence <= 1.0


def test_vote_abstains_on_noise():
    y = _nonrepeating_signal(duration_s=4.0)
    v = loop_periodicity_vote("span_1", y, _SR, beat_period_s=0.5)
    assert v.abstained
    assert v.label == "none"
    assert v.reason == AbstainReason.LOW_MARGIN


def test_vote_abstains_on_missing_beat_period():
    y = _looped_signal()
    v = loop_periodicity_vote("span_1", y, _SR, beat_period_s=None)
    assert v.abstained
    assert v.reason == AbstainReason.NO_DATA
