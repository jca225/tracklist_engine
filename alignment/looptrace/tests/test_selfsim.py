#!/usr/bin/env python3
"""Fixture tests for selfsim: CLONE verification must survive gain rides,
fractional-sample offsets (the resampling trap), and codec-level noise,
while unrelated content stays at the ~0 dB residual floor."""

from __future__ import annotations

import numpy as np
import pytest

from alignment.looptrace import selfsim
from alignment.looptrace.config import AUDIT_V3

SR = selfsim.SR


@pytest.fixture(scope="module")
def bed() -> np.ndarray:
    """60 s of band-limited noise — spectrally vocal-ish, self-dissimilar."""
    from scipy.signal import butter, sosfilt

    rng = np.random.default_rng(0)
    y = rng.standard_normal(60 * SR)
    sos = butter(4, [120 / (SR / 2), 4000 / (SR / 2)], btype="band", output="sos")
    return sosfilt(sos, y)


def _verify(y: np.ndarray, lag: float) -> float:
    p = selfsim.RepeatPair(10.0, 16.0, lag, 0.0)
    r, _ = selfsim.verify_pair(y, p, pad_s=1.0)
    return selfsim.residual_db(r)


def test_integer_sample_clone_is_clone(bed):
    y = bed.copy()
    y[40 * SR : 46 * SR] = bed[10 * SR : 16 * SR]
    assert _verify(y, 30.0) <= AUDIT_V3.clone_residual_db


def test_gain_scaled_clone_is_clone(bed):
    y = bed.copy()
    y[40 * SR : 46 * SR] = 0.6 * bed[10 * SR : 16 * SR]
    assert _verify(y, 30.0) <= AUDIT_V3.clone_residual_db


def test_half_sample_offset_clone_is_clone(bed):
    from scipy.signal import resample_poly

    seg = bed[10 * SR : 16 * SR]
    frac = resample_poly(seg, 2, 1)[1::2]  # half-sample shift
    y = bed.copy()
    y[40 * SR : 40 * SR + frac.size] = frac[: 6 * SR]
    assert _verify(y, 30.0) <= AUDIT_V3.clone_residual_db


def test_codec_noise_clone_is_clone(bed):
    """-20 dB additive noise ~ generous bound on AAC quantization noise."""
    rng = np.random.default_rng(1)
    seg = bed[10 * SR : 16 * SR]
    n = rng.standard_normal(seg.size)
    n *= np.linalg.norm(seg) / np.linalg.norm(n) * 10 ** (-20 / 20)
    y = bed.copy()
    y[40 * SR : 46 * SR] = seg + n
    assert _verify(y, 30.0) <= AUDIT_V3.clone_residual_db


def test_unrelated_content_is_not_clone(bed):
    assert _verify(bed, 30.0) > -1.0  # ~0 dB: no cancellation


def test_repeat_pairs_finds_planted_copy(bed):
    """The mel lag-diagonal scan must DETECT a planted 6 s copy at lag 30 s."""
    y = bed.copy()
    y[40 * SR : 46 * SR] = bed[10 * SR : 16 * SR]
    pairs = selfsim.repeat_pairs(y)
    hit = [
        p for p in pairs if abs(p.lag_s - 30.0) < 1.0 and p.a0 < 12.5 and p.a1 > 13.5
    ]
    assert hit, f"planted copy not detected; got {pairs}"


def test_repeat_pairs_clean_noise_has_no_repeats(bed):
    assert selfsim.repeat_pairs(bed) == []
