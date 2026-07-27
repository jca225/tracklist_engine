"""fp_offset's precomputed-mix cache seam (mix_fp) must be a pure speed knob.

The ACCEPT-precision harness scores many candidates against the SAME mix span;
re-fingerprinting the full mix per candidate is the ~1 min/case perf bug. Passing
a precomputed ``mix_fp`` (symmetric with the existing ``ref_fp``) must yield the
IDENTICAL result to computing it inline — otherwise the cache would change the
sensor, not just its speed.
"""

import librosa
import numpy as np
import pytest

from alignment.landmark_fp import (
    SR,
    fingerprint_from_audio,
    fp_offset,
)


def _tone_complex(dur_s: float, freqs, sr: int = SR) -> np.ndarray:
    t = np.arange(int(dur_s * sr)) / sr
    y = np.zeros_like(t)
    for f in freqs:
        y += np.sin(2 * np.pi * f * t)
    env = ((t * 4.0) % 1.0 < 0.5).astype(np.float64)
    y = y * env
    rng = np.random.default_rng(42)
    impulse_times = np.cumsum(rng.integers(4000, 16000, size=50))
    impulse_times = impulse_times[impulse_times < len(t)]
    for idx in impulse_times:
        y[idx] += 3.0
    return y.astype(np.float32)


def _planted_mix_and_ref():
    ref = _tone_complex(20.0, [400, 900, 1700, 3100, 5200])
    lead = int(3.0 * SR)
    mix = np.zeros(lead + len(ref) + int(2.0 * SR), dtype=np.float32)
    mix[lead : lead + len(ref)] += ref
    return mix, ref


def test_mix_fp_matches_inline_computation():
    mix, ref = _planted_mix_and_ref()
    inline = fp_offset(mix, ref)
    cached = fp_offset(None, ref, mix_fp=fingerprint_from_audio(mix))
    assert cached == inline  # bit-identical: cache is a pure speed knob


def test_mix_fp_allows_none_mix_audio():
    mix, ref = _planted_mix_and_ref()
    mfp = fingerprint_from_audio(mix)
    ref_start_s, votes, stretch, sharpness = fp_offset(None, ref, mix_fp=mfp)
    assert votes > 0
    # fp_offset returns the un-negated raw offset (caller applies max(0, -off));
    # the planted 3 s lead is recovered in magnitude.
    assert abs(abs(ref_start_s) - 3.0) < 1.0


def test_requires_mix_audio_or_mix_fp():
    _, ref = _planted_mix_and_ref()
    with pytest.raises(ValueError):
        fp_offset(None, ref)  # neither mix_y nor mix_fp
