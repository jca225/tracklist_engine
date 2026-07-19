"""Tests for mix-side fingerprint placement curves."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("librosa")  # fingerprint pipeline computes the STFT via librosa

from workspaces.alignment_prototype.landmark_fp import fingerprint_from_audio
from workspaces.alignment_prototype.mix_fp_hits import (
    candidate_density,
    fp_candidate_evidence,
    pick_dense_competitive,
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


def test_pick_dense_competitive_prefers_short_dense_over_long_false_diagonal() -> None:
    """bb11_34 shape: vote-argmax is a long false diagonal; GT onset is denser."""
    false_long = (3023.8, 3064.8, 403, -2848.0)  # dens ≈ 9.8/s
    true_onset = (2992.5, 3001.5, 163, -2841.0)  # dens ≈ 18.1/s
    other = (3018.1, 3066.8, 338, -2973.0)  # dens ≈ 6.9/s
    cands = [false_long, true_onset, other]
    assert candidate_density(true_onset) > candidate_density(false_long)
    picked = pick_dense_competitive(cands)
    assert picked is not None
    assert abs(picked[0] - 2992.5) < 0.1


def test_pick_dense_competitive_ignores_weak_dense_stray() -> None:
    """A tiny dense cluster below the vote floor must not beat the main diagonal."""
    main = (100.0, 140.0, 400, 0.0)  # dens = 10/s
    stray = (500.0, 501.0, 50, 0.0)  # dens = 50/s but only 12.5% of max votes
    picked = pick_dense_competitive([main, stray], frac=0.3)
    assert picked is not None
    assert abs(picked[0] - 100.0) < 0.1


def test_fp_candidate_evidence_preserves_order_and_exposes_strength() -> None:
    raw = [
        (10.0, 20.0, 100, 5.0),
        (30.0, 50.0, 40, -2.0),
    ]

    candidates = fp_candidate_evidence(raw)

    assert [candidate.rank for candidate in candidates] == [0, 1]
    assert candidates[0].set_start_s == 10.0
    assert candidates[0].votes == 100
    assert candidates[0].vote_density == 10.0
    assert candidates[0].runner_up_ratio == 2.5
    assert candidates[1].runner_up_ratio == 0.4


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
