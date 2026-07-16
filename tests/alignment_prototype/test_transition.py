"""Tests for the continuous tempo-ride synthetic probe."""

from __future__ import annotations

import numpy as np
import pytest

from workspaces.alignment_prototype.synthetic_mix.transition import (
    TempoRide,
    monotonic_filter,
    recovered_curve_error,
    render_ride,
)


def test_monotonic_filter_drops_outliers_not_anchor():
    # a clean rising chain with a catastrophic false peak as the FIRST point
    pts = [(0.0, 900.0), (1.0, 10.0), (2.0, 20.0), (3.0, 15.0), (4.0, 30.0)]
    kept = monotonic_filter(pts, slack_s=2.0)
    # LNDS keeps the long rising chain (10,20,30) incl. the 15 within slack? 15<20-2
    # -> 15 excluded; chain = (10,20,30), the bad first point dropped
    assert (0.0, 900.0) not in kept
    assert [r for _, r in kept] == [10.0, 20.0, 30.0]


def test_monotonic_filter_empty():
    assert monotonic_filter([]) == []


def test_constant_ride_is_linear():
    ride = TempoRide(r0=1.0, r1=1.0, mix_dur_s=10.0, ref_start_s=5.0)
    # r==1 constant -> ref_time = ref_start + t
    assert ride.ref_time(0.0) == pytest.approx(5.0)
    assert ride.ref_time(10.0) == pytest.approx(15.0)
    assert ride.ref_span_s == pytest.approx(10.0)


def test_ramp_is_quadratic():
    # ramp 1.0 -> 2.0 over 10s: ref consumed = t + 0.5*(0.1)*t^2
    ride = TempoRide(r0=1.0, r1=2.0, mix_dur_s=10.0)
    assert ride.ref_time(0.0) == pytest.approx(0.0)
    assert ride.ref_time(10.0) == pytest.approx(15.0)  # mean ratio 1.5 * 10
    # midpoint is NOT the linear-interp value (curvature): 5 + 0.05*25 = 6.25
    assert ride.ref_time(5.0) == pytest.approx(6.25)
    assert ride.mean_ratio() == pytest.approx(1.5)
    assert ride.ref_span_s == pytest.approx(15.0)


def test_render_length_and_exact_timemap():
    sr = 1000
    # ref[i] = i so out[k] should read the ref index at ref_time(t_k)*sr
    ref = np.arange(30_000, dtype=np.float32)
    ride = TempoRide(r0=0.5, r1=1.5, mix_dur_s=10.0)
    out = render_ride(ref, sr, ride)
    assert len(out) == sr * 10
    # out[k] ~= ref_time(k/sr) * sr  (linear-index ref)
    k = np.array([1000, 3000, 7000])
    want = ride.ref_time(k / sr) * sr
    assert np.allclose(out[k], want, rtol=1e-3)


def test_render_zero_fills_past_ref_tail():
    sr = 1000
    ref = np.ones(2000, dtype=np.float32)  # only 2 s of ref
    ride = TempoRide(r0=1.0, r1=1.0, mix_dur_s=5.0)  # needs 5 s of ref
    out = render_ride(ref, sr, ride)
    assert out[-1] == 0.0  # past the 2 s tail -> silence
    assert out[500] == pytest.approx(1.0)  # within the ref -> signal


def test_recovered_error_zero_for_truth():
    ride = TempoRide(r0=1.0, r1=1.3, mix_dur_s=20.0)
    truth = ride.label_points(20)
    err = recovered_curve_error(ride, truth)
    assert err["ref_mae_s"] == pytest.approx(0.0, abs=1e-9)
    assert err["mean_ratio_err"] == pytest.approx(0.0, abs=1e-6)


def test_straight_line_decoder_leaves_curvature_residual():
    # A decoder that assumes CONSTANT warp (straight line through endpoints) must
    # leave a bowed residual on a real ramp — the quantity the probe measures.
    ride = TempoRide(r0=1.0, r1=2.0, mix_dur_s=10.0)
    # straight line from (0, ref0) to (T, refT)
    r0p, rTp = ride.ref_time(0.0), ride.ref_time(10.0)
    line = [(t, r0p + (rTp - r0p) * t / 10.0) for t in np.linspace(0, 10, 21)]
    err = recovered_curve_error(ride, line)
    assert err["ref_mae_s"] > 0.3  # curvature is real, not noise
    # but the mean slope matches (endpoints pinned) -> ratio err ~ 0
    assert err["mean_ratio_err"] == pytest.approx(0.0, abs=1e-6)


def test_invalid_ride_rejected():
    with pytest.raises(ValueError):
        TempoRide(r0=0.0, r1=1.0, mix_dur_s=10.0)
    with pytest.raises(ValueError):
        TempoRide(r0=1.0, r1=1.0, mix_dur_s=0.0)
