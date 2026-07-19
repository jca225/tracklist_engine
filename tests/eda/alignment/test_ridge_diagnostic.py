from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("librosa")  # features → trajectory → recon_probe pulls librosa

from eda.alignment.ridge_diagnostic.cases import CaseRecord, _rank_candidates
from eda.alignment.ridge_diagnostic.features import (
    _align_bin_grid,
    _clamp_crop_bounds,
    cosine_sim_matrix,
)
from eda.alignment.ridge_diagnostic.ridge import gt_diagonal_mask, ridge_contrast


def test_rank_candidates_prefers_large_place_err_with_id_correct() -> None:
    rows = [
        {"id_correct": True, "place_err_s": 40.0, "slot": "a"},
        {"id_correct": True, "place_err_s": 5.0, "slot": "b"},
        {"id_correct": False, "place_err_s": 99.0, "slot": "c"},
        {"id_correct": True, "place_err_s": None, "slot": "d"},
    ]
    ranked = _rank_candidates(rows, min_place_err_s=15.0)
    assert [r["slot"] for r in ranked] == ["a"]


def test_cosine_sim_matrix_identity_has_strong_diagonal() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 8)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    m = cosine_sim_matrix(x, x)
    assert m.shape == (20, 20)
    assert float(np.mean(np.diag(m))) > 0.9


def test_align_bin_grid_crops_and_pads() -> None:
    m = np.ones((3, 4), dtype=np.float32)
    m[1, 2] = 0.5
    out = _align_bin_grid(m, (2, 5))
    assert out.shape == (2, 5)
    assert float(out[1, 2]) == 0.5
    assert float(out[0, 4]) == 0.0


def test_gt_diagonal_mask_multiseg_two_offsets() -> None:
    mask = gt_diagonal_mask((40, 40), offsets_s=(0.0, 10.0), bin_s=1.0, band_bins=0)
    assert mask[5, 5]
    assert mask[15, 5]
    assert not mask[5, 25]


def test_gt_diagonal_mask_crop_origins() -> None:
    mask = gt_diagonal_mask(
        (40, 40),
        offsets_s=(80.0,),
        bin_s=1.0,
        band_bins=0,
        mix_origin_s=100.0,
        ref_origin_s=20.0,
    )
    assert mask[5, 5]
    assert not mask[5, 6]


def test_ridge_contrast_high_on_planted_diagonal() -> None:
    m = np.zeros((30, 30), dtype=np.float32)
    for i in range(30):
        m[i, i] = 1.0
    mask = gt_diagonal_mask(m.shape, offsets_s=(0.0,), bin_s=1.0, band_bins=1)
    c = ridge_contrast(m, mask)
    assert c > 5.0


def test_clamp_crop_bounds_respects_file_duration() -> None:
    lo, hi = _clamp_crop_bounds(-5.0, 999.0, duration_s=120.0)
    assert lo == 0.0
    assert hi == 120.0
    lo, hi = _clamp_crop_bounds(100.0, 110.0, duration_s=120.0)
    assert lo == 100.0
    assert hi == 110.0
    lo, hi = _clamp_crop_bounds(115.0, 130.0, duration_s=120.0)
    assert lo == 115.0
    assert hi == 120.0
