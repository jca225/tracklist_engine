from __future__ import annotations

import librosa
import numpy as np

from workspaces.pws_aligner.operations import (
    TempoPitchLabel,
    keylock_vs_varispeed,
)

_SR = 22050


def _harmonic_tone(seconds: float = 6.0, f0: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * _SR)) / _SR
    y = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
    # amplitude modulation so time-stretch has structure to preserve
    return (y * (0.6 + 0.4 * np.sin(2 * np.pi * 2.0 * t))).astype(np.float32)


def test_varispeed_detected():
    ref = _harmonic_tone()
    r = 1.06
    mix = librosa.resample(
        ref, orig_sr=_SR, target_sr=int(_SR / r)
    )  # play fast: pitch+tempo up
    vote = keylock_vs_varispeed("s0", mix, ref, _SR, tempo_ratio=r)
    assert vote.label == TempoPitchLabel.VARISPEED.value
    assert not vote.abstained


def test_keylock_detected():
    ref = _harmonic_tone()
    r = 1.06
    mix = librosa.effects.time_stretch(ref, rate=r)  # tempo up, pitch preserved
    vote = keylock_vs_varispeed("s0", mix, ref, _SR, tempo_ratio=r)
    assert vote.label == TempoPitchLabel.KEYLOCK.value
    assert not vote.abstained


def test_keylock_small_tempo_nudge_not_varispeed():
    # r=1.025 → expected varispeed shift ≈ 12*log2(1.025) ≈ 0.43 st, below
    # _PITCH_TOL_ST=0.5.  A key-locked (time_stretch) signal has shift≈0 and
    # must be classified KEYLOCK, not VARISPEED.  Regression for the ambiguous
    # zone where the old ordering would misclassify.
    ref = _harmonic_tone()
    r = 1.025
    mix = librosa.effects.time_stretch(ref, rate=r)  # tempo up, pitch preserved
    vote = keylock_vs_varispeed("s0", mix, ref, _SR, tempo_ratio=r)
    assert vote.label == TempoPitchLabel.KEYLOCK.value
    assert not vote.abstained


def test_no_tempo_change():
    ref = _harmonic_tone()
    vote = keylock_vs_varispeed("s0", ref, ref, _SR, tempo_ratio=1.0)
    assert vote.label == TempoPitchLabel.NO_TEMPO_CHANGE.value


def test_keyshift_on_top_abstains():
    # pitch moved but matches NEITHER pattern (keylock + deliberate key-shift):
    ref = _harmonic_tone()
    mix = librosa.effects.pitch_shift(ref, sr=_SR, n_steps=3.0)
    vote = keylock_vs_varispeed("s0", mix, ref, _SR, tempo_ratio=1.06)
    assert vote.abstained
