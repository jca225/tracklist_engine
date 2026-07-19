"""Synthetic gates for HuBERT similarity → sparse segment correspondences."""

from __future__ import annotations

import numpy as np
import pytest

from workspaces.alignment_prototype.fp_segments.hubert_retrieve import (
    matches_from_similarity,
    retrieve_hubert_matches,
    sparsify_similarity,
)
from workspaces.alignment_prototype.fp_segments.run import _resolve_observation


def _planted_ridge(tm: int = 40, tr: int = 20, *, offset: int = 5) -> np.ndarray:
    """Cosine-like matrix with one clear diagonal ridge and weak noise."""
    rng = np.random.default_rng(0)
    m = rng.normal(0.05, 0.02, size=(tm, tr)).astype(np.float32)
    m = np.clip(m, 0.0, None)
    for i in range(tm):
        j = i - offset
        if 0 <= j < tr:
            m[i, j] = 0.95
            if j + 1 < tr:
                m[i, j + 1] = max(m[i, j + 1], 0.55)
            if j - 1 >= 0:
                m[i, j - 1] = max(m[i, j - 1], 0.55)
    return m


def test_sparsify_keeps_peaks_near_planted_diagonal():
    m = _planted_ridge()
    peaks = sparsify_similarity(m, peak_frac=0.5, neighborhood=3, max_peaks=64)
    assert peaks
    # Most retained peaks should sit on the planted offset≈5 diagonal.
    on_diag = sum(1 for mi, ri, _w in peaks if abs((mi - ri) - 5) <= 1)
    assert on_diag / len(peaks) >= 0.7


def test_sparsify_abstains_on_flat_noise():
    rng = np.random.default_rng(1)
    m = rng.normal(0.2, 0.01, size=(30, 20)).astype(np.float32)
    peaks = sparsify_similarity(m, peak_frac=0.9, neighborhood=3, max_peaks=64)
    assert peaks == ()


def test_matches_from_similarity_maps_bins_to_seconds():
    m = np.zeros((4, 3), dtype=np.float32)
    m[2, 1] = 1.0
    matches = matches_from_similarity(
        m,
        recording_id="r1",
        ref_stem="vocals",
        mix_channel="vocals",
        bin_s=0.5,
        peak_frac=0.5,
        neighborhood=1,
        max_peaks=8,
    )
    assert len(matches) == 1
    match = matches[0]
    assert match.mix_time_s == pytest.approx(2 * 0.5 + 0.25)
    assert match.ref_time_s == pytest.approx(1 * 0.5 + 0.25)
    assert match.weight == pytest.approx(1.0)
    assert match.hash_frequency == 1


def test_observation_defaults_and_cross_lane_rejects():
    assert _resolve_observation("vocal", None) == "hubert"
    assert _resolve_observation("instrumental", None) == "landmark"
    assert _resolve_observation("vocal", "landmark") == "landmark"
    with pytest.raises(ValueError, match="does not support"):
        _resolve_observation("instrumental", "hubert")
    with pytest.raises(ValueError, match="does not support"):
        _resolve_observation("vocal", "chroma")


def test_retrieve_hubert_matches_from_synthetic_features():
    # One shared direction creates a cosine ridge along matching bins.
    d = 8
    tm, tr = 16, 10
    mix = np.zeros((tm, d), dtype=np.float32)
    ref = np.zeros((tr, d), dtype=np.float32)
    for i in range(tm):
        mix[i, i % d] = 1.0
    for j in range(tr):
        ref[j, j % d] = 1.0
    matches = retrieve_hubert_matches(
        mix,
        ref,
        recording_id="r2",
        ref_stem="vocals",
        mix_channel="vocals",
        bin_s=0.5,
        already_pooled=True,
    )
    assert matches
    assert all(m.hash_frequency == 1 for m in matches)
    assert all(m.mix_channel == "vocals" for m in matches)
