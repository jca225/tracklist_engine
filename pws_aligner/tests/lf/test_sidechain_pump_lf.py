from __future__ import annotations

"""Tests for the sidechain_pump labeling function (A3).

Sidechain pumping = beat-rate amplitude ducking on a sustained band.
The RMS envelope of the signal has a periodic dip at the kick frequency (4-on-the-floor
= one duck per beat).

Positive: a sustained tone with its amplitude periodically ducked at beat rate.
Negative: same sustained tone with no ducking (flat RMS envelope).
"""

import numpy as np

import pytest

pytest.importorskip("librosa")  # the LF modules under test import librosa at module scope

from pws_aligner.lf.fx import detect_sidechain_pump, sidechain_pump_vote
from pws_aligner.core.votes import AbstainReason

_SR = 22050
_BPM = 128.0
_BEAT_S = 60.0 / _BPM  # ~0.469 s


def _sustained_tone(duration_s: float = 4.0, freq: float = 220.0) -> np.ndarray:
    """A constant-amplitude sustained tone — no pumping."""
    n = int(duration_s * _SR)
    t = np.arange(n, dtype=np.float32) / _SR
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _pumped_tone(
    duration_s: float = 4.0, freq: float = 220.0, bpm: float = _BPM, depth: float = 0.8
) -> np.ndarray:
    """Sustained tone with beat-rate amplitude ducking (sidechain pump).

    At each beat boundary the amplitude dips to (1-depth) then recovers with
    an exponential envelope — models the gain-reduction shape of a compressor
    triggered by the kick.
    """
    n = int(duration_s * _SR)
    t = np.arange(n, dtype=np.float32) / _SR
    beat_s = 60.0 / bpm
    # Build a gain envelope: starts at (1-depth) at each beat, recovers to 1.0
    tau = beat_s * 0.35  # recovery time constant (~35% of beat)
    phase = (t % beat_s) / beat_s  # 0..1 within each beat
    # Exponential rise from (1-depth) to 1.0 over one beat
    gain = 1.0 - depth * np.exp(-phase * beat_s / tau)
    tone = np.sin(2 * np.pi * freq * t).astype(np.float32)
    return (tone * gain).astype(np.float32)


def test_pumped_tone_detected():
    """A beat-ducked tone must be detected as sidechain pump."""
    y = _pumped_tone(duration_s=4.0, bpm=_BPM, depth=0.8)
    has, score = detect_sidechain_pump(y, _SR, beat_period_s=_BEAT_S)
    assert has, f"expected sidechain pump detected, score={score:.4f}"
    assert score > 0.0


def test_steady_tone_abstains():
    """An undipped sustained tone must not fire."""
    y = _sustained_tone(duration_s=4.0)
    has, score = detect_sidechain_pump(y, _SR, beat_period_s=_BEAT_S)
    assert not has, f"expected no sidechain pump for steady tone, score={score:.4f}"


def test_vote_fires_on_pumped_tone():
    y = _pumped_tone(duration_s=4.0, bpm=_BPM, depth=0.8)
    v = sidechain_pump_vote("span_1", y, _SR, beat_period_s=_BEAT_S)
    assert v.lf == "sidechain"
    assert v.label == "sidechain"
    assert not v.abstained
    assert 0.0 <= v.confidence <= 1.0


def test_vote_abstains_on_steady_tone():
    y = _sustained_tone(duration_s=4.0)
    v = sidechain_pump_vote("span_1", y, _SR, beat_period_s=_BEAT_S)
    assert v.abstained
    assert v.label == "none"
    assert v.reason == AbstainReason.LOW_MARGIN


def test_vote_abstains_on_missing_beat_period():
    y = _pumped_tone(duration_s=4.0, bpm=_BPM, depth=0.8)
    v = sidechain_pump_vote("span_1", y, _SR, beat_period_s=None)
    assert v.abstained
    assert v.reason == AbstainReason.NO_DATA
