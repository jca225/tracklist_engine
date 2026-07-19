from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from workspaces.alignment_prototype.fp_segments.local_decode import (
    decode_constituent,
)
from workspaces.alignment_prototype.fp_segments.schema import (
    ConstituentSegment,
    LandmarkMatch,
)
from workspaces.alignment_prototype.looptrace.config import SEG_V1


CFG = replace(
    SEG_V1,
    hough_bin_s=0.4,
    min_votes=5.0,
    inlier_tol_s=0.35,
    support_sigma_s=0.35,
    grid_step_s=0.5,
    null_level=0.25,
    lam=0.4,
    min_segment_s=3.0,
    seg_cost_s=1.0,
)


def _line(
    mix_start: float,
    mix_end: float,
    *,
    intercept: float,
    slope: float = 1.0,
    seed: int = 0,
) -> list[LandmarkMatch]:
    rng = np.random.default_rng(seed)
    times = np.arange(mix_start, mix_end, 0.25)
    return [
        LandmarkMatch(
            recording_id="track-a",
            ref_stem="instrumental",
            mix_channel="instrumental",
            mix_time_s=float(t),
            ref_time_s=float(slope * t + intercept + rng.normal(0.0, 0.025)),
            weight=1.0,
        )
        for t in times
    ]


def _decode(matches, duration=30.0, slopes=(1.0,)):
    return decode_constituent(
        matches,
        mix_duration_s=duration,
        allowed_slopes=slopes,
        config=CFG,
    )


def _assert_bounds(segment, start, end, *, tol=1.0):
    assert segment.mix_start_s == pytest.approx(start, abs=tol)
    assert segment.mix_end_s == pytest.approx(end, abs=tol)


def test_schema_rejects_invalid_values():
    with pytest.raises(ValueError, match="non-negative"):
        LandmarkMatch("r", "instrumental", "instrumental", -1.0, 2.0)
    with pytest.raises(ValueError, match="after start"):
        ConstituentSegment(
            "r",
            None,
            "instrumental",
            "instrumental",
            2.0,
            2.0,
            3.0,
            4.0,
            1.0,
            1.0,
            0.5,
        )


def test_recovers_one_bounded_linear_appearance():
    segments = _decode(_line(5.0, 15.0, intercept=20.0))
    assert len(segments) == 1
    _assert_bounds(segments[0], 5.0, 15.0)
    assert segments[0].ref_start_s == pytest.approx(25.0, abs=1.0)
    assert segments[0].ref_end_s == pytest.approx(35.0, abs=1.0)


def test_preserves_null_gap_between_reentries_on_same_diagonal():
    matches = [
        *_line(2.0, 8.0, intercept=10.0, seed=1),
        *_line(18.0, 25.0, intercept=10.0, seed=2),
    ]
    segments = _decode(matches)
    assert len(segments) == 2
    _assert_bounds(segments[0], 2.0, 8.0)
    _assert_bounds(segments[1], 18.0, 25.0)
    assert segments[0].ref_start_s == pytest.approx(12.0, abs=1.0)
    assert segments[1].ref_start_s == pytest.approx(28.0, abs=1.0)


def test_recovers_reference_jump_as_two_diagonals():
    matches = [
        *_line(3.0, 10.0, intercept=8.0, seed=3),
        *_line(13.0, 22.0, intercept=42.0, seed=4),
    ]
    segments = _decode(matches)
    assert len(segments) == 2
    _assert_bounds(segments[0], 3.0, 10.0)
    _assert_bounds(segments[1], 13.0, 22.0)
    assert segments[0].ref_start_s == pytest.approx(11.0, abs=1.0)
    assert segments[1].ref_start_s == pytest.approx(55.0, abs=1.0)


def test_sustained_path_rejects_short_repeated_pattern_distractor():
    matches = [
        *_line(4.0, 18.0, intercept=15.0, seed=5),
        *_line(21.0, 22.5, intercept=70.0, seed=6),
    ]
    segments = _decode(matches)
    assert len(segments) == 1
    _assert_bounds(segments[0], 4.0, 18.0)
    assert segments[0].ref_start_s == pytest.approx(19.0, abs=1.0)


def test_selects_correct_tempo_slope():
    matches = _line(4.0, 20.0, intercept=12.0, slope=1.05, seed=7)
    segments = _decode(matches, slopes=(0.95, 1.0, 1.05))
    assert len(segments) == 1
    assert segments[0].slope == pytest.approx(1.05)
    _assert_bounds(segments[0], 4.0, 20.0)


def test_returns_empty_when_no_hough_diagonal_exists():
    matches = [
        LandmarkMatch(
            "track-a",
            "instrumental",
            "instrumental",
            float(t),
            float(r),
        )
        for t, r in ((1, 3), (4, 17), (9, 8), (13, 29))
    ]
    assert _decode(matches) == ()


def test_rejects_mixed_recording_cloud():
    matches = _line(2.0, 8.0, intercept=10.0)
    matches.append(
        LandmarkMatch("track-b", "instrumental", "instrumental", 3.0, 7.0)
    )
    with pytest.raises(ValueError, match="one recording"):
        _decode(matches)


def test_audio_landmarks_recover_jump_and_gap_end_to_end():
    """Synthetic audio -> hashes -> raw matches -> bounded segment decode."""
    from workspaces.alignment_prototype.looptrace import fixtures, landmarks

    song = fixtures.make_song(32.0, seed=21)
    sr = fixtures.SR
    rng = np.random.default_rng(22)
    mix = rng.normal(0.0, 0.002, int(30.0 * sr)).astype(np.float32)
    # Two excerpts from different reference sections, separated by ten seconds
    # of NULL/background.
    mix[int(2 * sr) : int(8 * sr)] += song[int(5 * sr) : int(11 * sr)]
    mix[int(18 * sr) : int(25 * sr)] += song[int(18 * sr) : int(25 * sr)]

    points = landmarks.match_points(
        landmarks.audio_points(mix),
        landmarks.audio_points(song),
    )
    matches = [
        LandmarkMatch(
            "track-a",
            "instrumental",
            "instrumental",
            float(mix_t),
            float(ref_t),
        )
        for mix_t, ref_t in points
    ]
    segments = _decode(matches)

    assert len(segments) == 2
    _assert_bounds(segments[0], 2.0, 8.0, tol=2.5)
    _assert_bounds(segments[1], 18.0, 25.0, tol=2.5)
    assert segments[0].ref_start_s == pytest.approx(5.0, abs=1.5)
    assert segments[1].ref_start_s == pytest.approx(18.0, abs=2.5)
