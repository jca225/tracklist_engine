"""Test the ride-eval label round-trip (pure part)."""

from __future__ import annotations

from alignment.synthetic_mix.eval_transitions import (
    ride_from_label,
)


def test_ride_from_label_roundtrip():
    label = {"r0": 0.9, "r1": 1.1, "mix_dur_s": 90.0, "ref_start_s": 30.0}
    ride = ride_from_label(label)
    assert ride.r0 == 0.9
    assert ride.r1 == 1.1
    assert ride.mix_dur_s == 90.0
    assert ride.ref_start_s == 30.0
    # mean ratio 1.0 -> ref_span == mix_dur
    assert ride.ref_span_s == 90.0
