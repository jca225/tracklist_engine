from __future__ import annotations

import numpy as np

from pws_aligner.fx_lf import detect_echo_tail, echo_tail_vote
from pws_aligner.votes import AbstainReason

_SR = 22050


def _burst(n: int, at_s: float, width_s: float = 0.04, seed: int = 0) -> np.ndarray:
    """A single windowed noise burst in an n-sample buffer (a percussive hit)."""
    rng = np.random.RandomState(seed)
    y = np.zeros(n, dtype=np.float32)
    start = int(at_s * _SR)
    w = int(width_s * _SR)
    env = np.hanning(w)
    y[start : start + w] = (rng.randn(w) * env).astype(np.float32)
    return y


def _add_echo(
    dry: np.ndarray, delay_s: float, n_rep: int = 4, decay: float = 0.6
) -> np.ndarray:
    """dry + a decaying train of repeats at `delay_s` — a beat-synced echo tail."""
    y = dry.copy()
    d = int(delay_s * _SR)
    for k in range(1, n_rep + 1):
        shifted = np.zeros_like(dry)
        if k * d < len(dry):
            shifted[k * d :] = dry[: len(dry) - k * d]
        y = y + (decay**k) * shifted
    return y.astype(np.float32)


def test_echo_detected_at_beat_fraction():
    # echo delay 0.25s == 1/4 of a 1.0s beat period (a beat-rational lag)
    dry = _burst(int(3.0 * _SR), at_s=0.5)
    echo = _add_echo(dry, delay_s=0.25)
    has, conf = detect_echo_tail(echo, _SR, beat_period_s=1.0)
    assert has
    assert conf > 0.0


def test_dry_signal_no_echo():
    dry = _burst(int(3.0 * _SR), at_s=0.5)
    has, conf = detect_echo_tail(dry, _SR, beat_period_s=1.0)
    assert not has


def test_echo_at_non_beat_lag_not_flagged():
    # echo at 0.31s, but beat_period 1.0s -> no beat-fraction lands on 0.31s;
    # the detector only scores beat-rational lags, so this should NOT flag.
    dry = _burst(int(3.0 * _SR), at_s=0.5)
    echo = _add_echo(dry, delay_s=0.31)
    has, _ = detect_echo_tail(echo, _SR, beat_period_s=1.0)
    assert not has


def test_vote_fires_on_echo():
    dry = _burst(int(3.0 * _SR), at_s=0.5)
    echo = _add_echo(dry, delay_s=0.25)
    v = echo_tail_vote("span_1", echo, _SR, beat_period_s=1.0)
    assert v.lf == "echo_tail"
    assert v.label == "echo"
    assert not v.abstained
    assert 0.0 <= v.confidence <= 1.0


def test_vote_abstains_on_dry():
    dry = _burst(int(3.0 * _SR), at_s=0.5)
    v = echo_tail_vote("span_1", dry, _SR, beat_period_s=1.0)
    assert v.abstained
    assert v.label == "none"


def test_vote_abstains_on_missing_beat_period():
    dry = _burst(int(3.0 * _SR), at_s=0.5)
    echo = _add_echo(dry, delay_s=0.25)
    v = echo_tail_vote("span_1", echo, _SR, beat_period_s=None)
    assert v.abstained
    assert v.reason == AbstainReason.NO_DATA
