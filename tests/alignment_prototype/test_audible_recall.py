"""Trajectory scoring over AUDIBLE segment intervals only (WS0 de-inflation).

A multiseg/loop GT span's `[set_start_s, set_end_s]` envelope spans the SILENT
gaps between the DJ's segments (Slide: 839 s envelope = 48 s played + 791 s gap;
Good Charlotte: 206 s = 38 s played + 167 s gap). The old scorer sampled the
whole envelope and `Trajectory.from_segments` diluted each pre-gap segment's
slope to reach the next segment, so the gap both (a) inflated the "GT-seconds
lost" denominator and (b) penalized a prediction that diverged where nothing
plays. These tests pin the corrected behavior: recall is measured only over the
real played mix intervals; the gap is neither credited nor penalized; a separate
metric reports gap hallucination. Linear spans (no ref_segments) are unchanged.
"""

from __future__ import annotations

import math

from workspaces.alignment_prototype.path_decode import (
    _audible_intervals,
    audible_seconds,
    gap_hallucination_frac,
    gt_placement_onset,
    trajectory_acc,
)


def _row(segs=None, **kw):
    r = {"set_start_s": 0.0, "set_end_s": 120.0, "tempo_ratio": 1.0}
    if segs is not None:
        r["ref_segments"] = [
            {"mix_start_s": ms, "ref_start_s": rs, "ref_end_s": re}
            for ms, rs, re in segs
        ]
    r.update(kw)
    return r


# Two played segments at mix [0,10] and [100,110], a 90 s silent gap between.
_SEGS = [(0.0, 0.0, 10.0), (100.0, 100.0, 110.0)]


class TestAudibleIntervals:
    def test_segmented_row_excludes_the_gap(self):
        ivals = _audible_intervals(_row(_SEGS))
        assert ivals == [(0.0, 10.0), (100.0, 110.0)]

    def test_audible_seconds_is_played_not_envelope(self):
        row = _row(_SEGS)
        assert audible_seconds(row) == 20.0  # NOT the 120 s envelope
        assert float(row["set_end_s"]) - float(row["set_start_s"]) == 120.0

    def test_tempo_scales_played_extent(self):
        # 10 s of ref played at 2x is 5 s of mix.
        row = _row([(0.0, 0.0, 10.0)], set_end_s=100.0, tempo_ratio=2.0)
        ((lo, hi),) = _audible_intervals(row)
        assert lo == 0.0 and math.isclose(hi, 5.0)

    def test_linear_row_is_the_whole_envelope(self):
        row = _row(None, ref_start_s=30.0, set_end_s=120.0)
        assert _audible_intervals(row) == [(0.0, 120.0)]
        assert audible_seconds(row) == 120.0


class TestTrajectoryAccOverAudible:
    def test_gap_garbage_does_not_lower_score(self):
        # Prediction is exactly right on both played segments but injects a
        # wild off-diagonal segment in the silent gap. That garbage sits where
        # nothing plays, so the score must be a perfect 1.0.
        row = _row(_SEGS)
        pred = [
            (0.0, 0.0, 10.0),
            (40.0, 500.0, 510.0),  # garbage, entirely within the [10,100] gap
            (100.0, 100.0, 110.0),
        ]
        strict, _n, _f = trajectory_acc(pred, row)
        assert strict == 1.0

    def test_played_segments_use_real_slope_not_diluted(self):
        # A prediction that plays each segment at its real rate matches GT.
        row = _row(_SEGS)
        pred = [(0.0, 0.0, 10.0), (100.0, 100.0, 110.0)]
        strict, _n, _f = trajectory_acc(pred, row)
        assert strict == 1.0

    def test_wrong_second_segment_costs_its_audible_share(self):
        # Segment A right, segment B wrong (equal audible length) -> ~half.
        row = _row(_SEGS)
        pred = [(0.0, 0.0, 10.0), (100.0, 900.0, 910.0)]  # B ref off by 800 s
        strict, _n, _f = trajectory_acc(pred, row)
        assert 0.4 <= strict <= 0.6

    def test_linear_span_unchanged(self):
        # No ref_segments: identical straight diagonal must still score 1.0.
        row = _row(None, ref_start_s=0.0, ref_end_s=120.0, set_end_s=120.0)
        pred = [(0.0, 0.0, 120.0)]
        strict, _n, _f = trajectory_acc(pred, row)
        assert strict == 1.0


class TestGainEnvelopeAudibility:
    """The fader ride silences part of a span independently of the DJ's segment
    structure. A ref_segment drawn over a *faded-down* region (e.g. a track that
    fades in under a crossfade) is inaudible and must not count as played GT.
    Galantis "You" (BB12): a 64 s segment sits under a fader at 0 before the
    audible entry, so the old scorer charged the machine ~66 s of placement +
    trajectory error for content nobody hears.
    """

    # A silent lead-in segment [0,64] then the audible body [64,128]; the fader
    # is at 0 until audible_start_s=64.
    _LEADIN = [(0.0, 0.0, 64.0), (64.0, 64.0, 128.0)]

    def test_faded_lead_in_excluded_from_audible(self):
        row = _row(self._LEADIN, set_end_s=128.0, audible_start_s=64.0)
        assert _audible_intervals(row) == [(64.0, 128.0)]
        assert audible_seconds(row) == 64.0  # NOT 128

    def test_faded_out_tail_excluded_from_audible(self):
        row = _row(self._LEADIN, set_end_s=128.0, audible_end_s=100.0)
        ((lo, hi),) = _audible_intervals(row)
        assert lo == 0.0 and hi == 100.0

    def test_no_gain_fields_leaves_intervals_unclipped(self):
        # Rows without a fader ride (audible_* absent) keep the old behavior.
        row = _row(_SEGS)
        assert _audible_intervals(row) == [(0.0, 10.0), (100.0, 110.0)]

    def test_trajectory_not_penalized_for_silent_lead_in(self):
        # The machine correctly plays only the audible body and has nothing over
        # the faded-down lead-in. That silent region must not lower the score.
        row = _row(self._LEADIN, set_end_s=128.0, audible_start_s=64.0)
        pred = [(64.0, 64.0, 128.0)]
        strict, _n, _f = trajectory_acc(pred, row)
        assert strict == 1.0


class TestPlacementOnset:
    def test_uses_audible_start_when_present(self):
        row = _row(None, set_end_s=128.0, audible_start_s=64.0)
        assert gt_placement_onset(row) == 64.0

    def test_falls_back_to_set_start(self):
        row = _row(None, set_start_s=5.0, set_end_s=128.0)
        assert gt_placement_onset(row) == 5.0


class TestGapHallucination:
    def test_clean_prediction_has_zero_hallucination(self):
        row = _row(_SEGS)
        pred = [(0.0, 0.0, 10.0), (100.0, 100.0, 110.0)]
        assert gap_hallucination_frac(pred, row) == 0.0

    def test_gap_covering_segment_is_hallucination(self):
        # A 10 s pred segment sitting in the 90 s gap -> ~0.11 hallucinated.
        row = _row(_SEGS)
        pred = [
            (0.0, 0.0, 10.0),
            (40.0, 500.0, 510.0),  # covers mix [40,50] inside the gap
            (100.0, 100.0, 110.0),
        ]
        frac = gap_hallucination_frac(pred, row)
        assert math.isclose(frac, 10.0 / 90.0, abs_tol=0.02)

    def test_linear_span_has_no_gap(self):
        row = _row(None, ref_start_s=0.0, ref_end_s=120.0)
        assert gap_hallucination_frac([(0.0, 0.0, 120.0)], row) == 0.0
