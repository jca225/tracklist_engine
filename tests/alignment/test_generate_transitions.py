"""Tests for the ride-augmented synthetic dataset generator (pure logic)."""

from __future__ import annotations

import numpy as np

from alignment.synthetic_mix.generate_transitions import (
    CURVATURE_MIX,
    build_specs,
    sample_ride,
)


def test_build_specs_count_and_round_robin():
    refs = ["a.flac", "b.flac"]
    specs = build_specs(refs, n=5, mix_dur_s=90.0, seed=0)
    assert len(specs) == 5
    assert [s.ref_path for s in specs] == ["a", "b", "a", "b", "a"] or [
        s.ref_path for s in specs
    ] == ["a.flac", "b.flac", "a.flac", "b.flac", "a.flac"]
    assert all(s.mix_dur_s == 90.0 for s in specs)


def test_sample_ride_bands_and_direction():
    rng = np.random.default_rng(1)
    halves = {c[0]: c[1] for c in CURVATURE_MIX}
    for _ in range(200):
        name, r0, r1, ref_start, direction = sample_ride(rng, 300.0, 90.0)
        half = halves[name]
        lo, hi = round(1 - half, 3), round(1 + half, 3)
        assert {round(r0, 3), round(r1, 3)} == {lo, hi}
        assert (direction == "accel") == (r0 < r1)
        assert ref_start >= 0.0


def test_curvature_distribution_weighted_gentle_medium():
    rng = np.random.default_rng(0)
    from collections import Counter

    c = Counter(sample_ride(rng, 300.0, 90.0)[0] for _ in range(2000))
    # gentle+medium dominate; steep is the tail
    assert c["gentle"] + c["medium"] > 4 * c["steep"]


def test_specs_are_deterministic_for_seed():
    refs = ["r0", "r1", "r2"]
    a = build_specs(refs, 8, 90.0, seed=42)
    b = build_specs(refs, 8, 90.0, seed=42)
    assert a == b
