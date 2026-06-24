"""Tests for mix-side fingerprint placement curves."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("librosa")  # fingerprint pipeline computes the STFT via librosa

from workspaces.alignment_prototype.landmark_fp import fingerprint_from_audio
from workspaces.alignment_prototype.mix_fp_hits import (
    placement_curve,
    scan_band,
    score_mix_window,
)


def _tone(seconds: float = 20.0, freq: float = 440.0) -> np.ndarray:
    sr = 22050
    t = np.arange(int(sr * seconds)) / sr
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_score_mix_window_self_match() -> None:
    y = _tone(15.0)
    fp = fingerprint_from_audio(y)
    chunk = y[: 22050 * 8]
    votes, sharp, _st = score_mix_window(chunk, ref_fp=fp, ref_y=y)
    assert votes > 0
    assert sharp >= 1.0


def test_scan_band_finds_self_hit() -> None:
    rng = np.random.default_rng(2)
    y = rng.standard_normal(22050 * 30).astype(np.float32) * 0.08
    fp = fingerprint_from_audio(y)
    hits = scan_band(
        y,
        ref_fp=fp,
        ref_y=y,
        lo_s=5.0,
        hi_s=25.0,
        win_s=8.0,
        step_s=2.0,
        recording_id="rid1",
        stem="regular",
    )
    assert hits
    assert hits[0].recording_id == "rid1"
    assert hits[0].votes >= 25


def test_placement_curve_peaks_near_coarse() -> None:
    y = _tone(40.0, freq=660.0)
    fp = fingerprint_from_audio(y)
    measure_mid = np.linspace(0, 40, 80)
    curve = placement_curve(
        y,
        ref_fp=fp,
        ref_y=y,
        measure_mid_s=measure_mid,
        coarse_start_s=12.0,
        band_s=15.0,
        win_s=8.0,
    )
    from workspaces.alignment_prototype.sequence_decode import NEG

    valid = curve > NEG / 2
    assert valid.any()
    peak_t = measure_mid[int(np.argmax(curve))]
    assert abs(peak_t - 12.0) < 6.0
