"""Tests for the canonical (path_decode) transition recovery — pure part.

The audio-dependent `decode_curvature` needs a real stem + librosa, so we test
only the pure segment->curve reconstruction (`curve_from_segments`), which is
the load-bearing glue between `decode_path`'s segment list and the ride error.
"""

from __future__ import annotations

from workspaces.alignment_prototype.synthetic_mix.transition import TempoRide
from workspaces.alignment_prototype.synthetic_mix.transition_pathdecode import (
    curve_from_segments,
)


def test_empty_segments_give_no_points():
    assert curve_from_segments([], 90.0, 1.0, 2.0) == []


def test_single_linear_segment_recovers_the_diagonal():
    # one segment: at mix 0 ref starts 10, ends 100 over the 90s span -> slope 1.0
    segs = [(0.0, 10.0, 100.0)]
    pts = curve_from_segments(segs, 90.0, 1.0, 30.0)
    mix_ts = [p[0] for p in pts]
    ref_ts = [p[1] for p in pts]
    assert mix_ts == [0.0, 30.0, 60.0]
    # ref advances at 1.0 ref-s per mix-s from 10.0
    assert ref_ts == [10.0, 40.0, 70.0]


def test_recovers_a_constant_ratio_ride_exactly():
    # a flat ride (r0==r1==1.0) IS a single diagonal; a single-segment decode of
    # it must have ~zero error against the true curve.
    ride = TempoRide(r0=1.0, r1=1.0, mix_dur_s=90.0, ref_start_s=0.0)
    segs = [(0.0, 0.0, 90.0)]  # ref 0->90 over the 90s span, slope 1.0
    pts = curve_from_segments(segs, 90.0, 1.0, 5.0)
    for t, r in pts:
        assert abs(r - ride.ref_time(t)) < 1e-6


def test_two_segment_jump_reconstructs_piecewise():
    # a section jump: diagonal to mix 40, then ref resets and continues. Sample
    # strictly inside each piece (the exact boundary belongs to the first piece,
    # by value_at's first-covering-piece rule — that's the pre-jump value).
    segs = [(0.0, 0.0, 40.0), (40.0, 200.0, 250.0)]
    pts = curve_from_segments(segs, 90.0, 1.0, 50.0)  # samples mix 0 and 50
    by_mix = dict(pts)
    # first piece slope = (40-0)/(40-0) = 1.0 -> at mix 0, ref 0
    assert abs(by_mix[0.0] - 0.0) < 1e-6
    # second piece: ref 200 at mix 40, slope (250-200)/(90-40)=1.0 -> mix 50 ref 210
    assert abs(by_mix[50.0] - 210.0) < 1e-6
