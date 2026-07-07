"""Trajectory training targets: GT rasterization + segment collapse.

The rasterizer must agree with the scorer's own interpolation
(`path_decode._gt_pieces`/`_ref_at`), mask inaudible (faded) frames as NULL,
and treat `unalignable` as a positive abstain (all placement frames ignored,
example kept). `frames_to_segments` must invert a clean per-frame decode
back into the segment triples `trajectory_acc` consumes.
"""

from __future__ import annotations

import numpy as np

from workspaces.alignment_prototype.trajectory.targets import (
    KIND_IGNORE,
    KIND_NULL,
    KIND_POSITION,
    frames_to_segments,
    gain_at,
    raster_targets,
)

BIN = 0.5


def _times(s0: float, s1: float) -> np.ndarray:
    b0 = int(round(s0 / BIN))
    n = int(round((s1 - s0) / BIN))
    return (b0 + np.arange(n) + 0.5) * BIN


class TestRasterLinear:
    def test_linear_span_follows_tempo_ratio(self) -> None:
        row = {
            "set_start_s": 100.0,
            "set_end_s": 130.0,
            "ref_start_s": 40.0,
            "tempo_ratio": 1.05,
        }
        t = raster_targets(row, _times(100.0, 130.0), ref_dur_s=200.0)
        assert (t.kind == KIND_POSITION).all()
        # ref advances at tempo_ratio per mix second from (100, 40)
        expect = 40.0 + (t.times_s - 100.0) * 1.05
        assert np.allclose(t.ref_pos_s, expect)
        assert t.span_class == "linear"

    def test_ref_overrun_is_ignored_not_clipped(self) -> None:
        row = {"set_start_s": 0.0, "set_end_s": 20.0, "ref_start_s": 190.0}
        t = raster_targets(row, _times(0.0, 20.0), ref_dur_s=200.0)
        # after 10 s the trajectory runs off the end of the ref
        assert (t.kind[:18] == KIND_POSITION).all()
        assert (t.kind[21:] == KIND_IGNORE).all()


class TestRasterMultiseg:
    def test_segments_interpolate_like_the_scorer(self) -> None:
        row = {
            "set_start_s": 60.0,
            "set_end_s": 90.0,
            "ref_start_s": 30.0,
            "tempo_ratio": 1.0,
            "ref_segments": [
                {"mix_start_s": 60.0, "ref_start_s": 30.0, "ref_end_s": 45.0},
                {"mix_start_s": 75.0, "ref_start_s": 150.0, "ref_end_s": 165.0},
            ],
        }
        t = raster_targets(row, _times(60.0, 90.0), ref_dur_s=300.0)
        # first frame sits on segment 1's diagonal, a frame after the jump on 2's
        assert abs(t.ref_pos_s[0] - 30.25) < BIN
        assert abs(t.ref_pos_s[-1] - (150.0 + (t.times_s[-1] - 75.0))) < BIN
        assert t.span_class == "multiseg"


class TestGainMasking:
    def test_faded_frames_become_null(self) -> None:
        row = {
            "set_start_s": 0.0,
            "set_end_s": 10.0,
            "ref_start_s": 0.0,
            "gain_curve": [[0.0, 1.0], [5.0, 1.0], [5.5, 0.0], [10.0, 0.0]],
        }
        t = raster_targets(row, _times(0.0, 10.0), ref_dur_s=100.0)
        assert (t.kind[:10] == KIND_POSITION).all()  # audible half
        assert (t.kind[12:] == KIND_NULL).all()  # faded-out half
        assert t.gain[0] == 1.0 and t.gain[-1] < 0.1

    def test_empty_curve_is_unity(self) -> None:
        g = gain_at((), np.array([1.0, 2.0]))
        assert (g == 1.0).all()


class TestAbstain:
    def test_unalignable_masks_everything_but_flags(self) -> None:
        row = {
            "set_start_s": 0.0,
            "set_end_s": 10.0,
            "ref_start_s": 0.0,
            "unalignable": True,
        }
        t = raster_targets(row, _times(0.0, 10.0), ref_dur_s=100.0)
        assert t.abstain
        assert (t.kind == KIND_IGNORE).all()


class TestFramesToSegments:
    def test_single_diagonal_run(self) -> None:
        ref_bin = np.arange(80, 90)  # constant offset
        segs = frames_to_segments(ref_bin, np.zeros(10, dtype=bool), BIN)
        assert segs == [(0.0, 40.0, 45.0)]

    def test_jump_splits_segments(self) -> None:
        ref_bin = np.concatenate([np.arange(80, 85), np.arange(200, 205)])
        segs = frames_to_segments(ref_bin, np.zeros(10, dtype=bool), BIN)
        assert len(segs) == 2
        assert segs[0][1] == 40.0 and segs[1][1] == 100.0
        assert segs[1][0] == 2.5  # second segment starts at frame 5

    def test_null_terminates_segment(self) -> None:
        ref_bin = np.arange(80, 90)
        null = np.zeros(10, dtype=bool)
        null[5:] = True
        segs = frames_to_segments(ref_bin, null, BIN)
        assert segs == [(0.0, 40.0, 42.5)]

    def test_small_drift_stays_one_segment(self) -> None:
        # tempo != 1 drifts the offset by 1 bin mid-run; jump_bins=2 tolerates
        ref_bin = np.array([80, 81, 82, 84, 85, 86])
        segs = frames_to_segments(ref_bin, np.zeros(6, dtype=bool), BIN)
        assert len(segs) == 1
