from __future__ import annotations

import math

import pytest

from alignment.fp_segments.retrieve import retrieve_matches
from alignment.landmark_fp import (
    FHOP,
    FPS,
    SR,
    LandmarkFingerprint,
)


def _fp(hashes):
    return LandmarkFingerprint(fps=FPS, duration_s=10.0, hashes=hashes)


def test_retrieve_preserves_raw_time_correspondences():
    key = (1, 2, 3)
    matches = retrieve_matches(
        {key: (10, 20)},
        _fp({key: (30,)}),
        recording_id="r",
        ref_stem="instrumental",
        mix_channel="instrumental",
    )
    scale = FHOP / SR
    assert [(m.mix_time_s, m.ref_time_s) for m in matches] == pytest.approx(
        [(10 * scale, 30 * scale), (20 * scale, 30 * scale)]
    )
    assert all(m.hash_frequency == 2 for m in matches)
    assert all(m.weight == pytest.approx(1 / math.log2(4)) for m in matches)


def test_retrieve_caps_common_hash_cartesian_explosion():
    key = (1, 2, 3)
    matches = retrieve_matches(
        {key: tuple(range(9))},
        _fp({key: tuple(range(8))}),
        recording_id="r",
        ref_stem="instrumental",
        mix_channel="instrumental",
        pair_cap=64,
    )
    assert matches == ()  # 9 * 8 exceeds the cap


def test_retrieve_inverse_weights_more_common_keys():
    rare = (1, 2, 3)
    common = (4, 5, 6)
    matches = retrieve_matches(
        {rare: (1,), common: (2, 3)},
        _fp({rare: (4,), common: (5, 6)}),
        recording_id="r",
        ref_stem="instrumental",
        mix_channel="instrumental",
    )
    rare_weight = next(m.weight for m in matches if m.hash_frequency == 1)
    common_weight = next(m.weight for m in matches if m.hash_frequency == 4)
    assert rare_weight > common_weight


def test_retrieve_validates_pair_cap():
    with pytest.raises(ValueError, match="positive"):
        retrieve_matches(
            {},
            _fp({}),
            recording_id="r",
            ref_stem="instrumental",
            mix_channel="instrumental",
            pair_cap=0,
        )
