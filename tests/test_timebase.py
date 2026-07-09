"""core.timebase — TimeMap laws (inverse round-trip, composition, monotonic guard)."""

from __future__ import annotations

import pytest

try:
    from hypothesis import given
    from hypothesis import strategies as st

    HAVE_HYPOTHESIS = True
except ImportError:  # pragma: no cover - CI installs hypothesis; dev venvs may not
    HAVE_HYPOTHESIS = False

from core.timebase import TimeMap, TimeMapError


def _mix_ref() -> TimeMap:
    # a DJ playing a ref at 1.06x for 60s, then 0.5x
    return TimeMap("mix", "ref", ((0.0, 30.0), (60.0, 93.6), (120.0, 123.6)))


def test_apply_interpolates():
    m = _mix_ref()
    assert m.apply(0.0) == pytest.approx(30.0)
    assert m.apply(30.0) == pytest.approx(61.8)
    assert m.apply(120.0) == pytest.approx(123.6)


def test_inverse_round_trip():
    m = _mix_ref()
    inv = m.inverse()
    for x in (0.0, 7.3, 59.99, 60.0, 100.0, 120.0):
        assert inv.apply(m.apply(x)) == pytest.approx(x, abs=1e-9)
    assert inv.src_domain == "ref" and inv.dst_domain == "mix"


def test_double_inverse_identity():
    m = _mix_ref()
    assert m.inverse().inverse() == m


def test_backward_jump_rejected():
    with pytest.raises(TimeMapError):
        TimeMap("mix", "ref", ((0.0, 10.0), (5.0, 8.0)))  # ref goes backward
    with pytest.raises(TimeMapError):
        TimeMap("mix", "ref", ((0.0, 10.0), (0.0, 20.0)))  # mix stalls


def test_single_anchor_rejected():
    with pytest.raises(TimeMapError):
        TimeMap("mix", "ref", ((0.0, 0.0),))


def test_compose_domain_checked():
    mix_to_beat = TimeMap.linear(
        "mix", "arr_beat", src_start=0.0, dst_start=0.0, rate=2.0, length=100.0
    )
    beat_to_ref = TimeMap.linear(
        "arr_beat", "ref", src_start=0.0, dst_start=5.0, rate=0.5, length=200.0
    )
    mix_to_ref = mix_to_beat.compose(beat_to_ref)
    assert mix_to_ref.src_domain == "mix" and mix_to_ref.dst_domain == "ref"
    assert mix_to_ref.apply(10.0) == pytest.approx(
        beat_to_ref.apply(mix_to_beat.apply(10.0))
    )
    with pytest.raises(TimeMapError):
        mix_to_beat.compose(mix_to_beat)  # arr_beat cannot feed a mix-> map


def test_compose_exact_at_interior_knots():
    a = TimeMap("mix", "beat", ((0.0, 0.0), (10.0, 20.0)))
    b = TimeMap("beat", "ref", ((0.0, 0.0), (5.0, 5.0), (20.0, 50.0)))
    c = a.compose(b)
    # b's knee at beat=5 pulls back to mix=2.5 and must be exact there
    assert c.apply(2.5) == pytest.approx(b.apply(a.apply(2.5)))
    for x in (0.0, 1.0, 2.5, 7.0, 10.0):
        assert c.apply(x) == pytest.approx(b.apply(a.apply(x)))


def test_policy_clamp_and_raise():
    m = TimeMap("mix", "ref", ((0.0, 0.0), (10.0, 10.0)), policy="clamp")
    assert m.apply(-5.0) == 0.0
    assert m.apply(15.0) == 10.0
    r = TimeMap("mix", "ref", ((0.0, 0.0), (10.0, 10.0)), policy="raise")
    with pytest.raises(TimeMapError):
        r.apply(11.0)


def test_extrapolate_uses_edge_segments():
    m = _mix_ref()  # first segment slope 1.06
    assert m.apply(-10.0) == pytest.approx(30.0 - 10.0 * 1.06)


def test_from_pairs_drops_stalls():
    # duplicated warp markers (the Aftershock class) + a DTW-style stall
    m = TimeMap.from_pairs(
        "mix", "ref", [(0.0, 0.0), (1.0, 1.0), (1.0, 1.5), (2.0, 1.5), (3.0, 3.0)]
    )
    # (1.0, 1.5) dropped (mix stalls); (2.0, 1.5) kept (advances in both
    # coords vs the last KEPT pair (1.0, 1.0))
    assert m.anchors == ((0.0, 0.0), (1.0, 1.0), (2.0, 1.5), (3.0, 3.0))


if HAVE_HYPOTHESIS:
    # well-conditioned anchors: steps in [1e-3, 1e3] so slopes stay finite
    # (adjacent-float anchors make slopes ~1e16 and overflow — real time maps
    # have sane rates; degenerate ones are construction-time concerns)
    increasing = st.lists(
        st.floats(min_value=1e-3, max_value=1e3, allow_nan=False),
        min_size=2,
        max_size=12,
    ).map(lambda steps: [sum(steps[: i + 1]) for i in range(len(steps))])

    @given(xs=increasing, ys=increasing, q=st.floats(-2e5, 2e5))
    def test_property_inverse_round_trip(xs, ys, q):
        n = min(len(xs), len(ys))
        if n < 2:
            return
        m = TimeMap("a", "b", tuple(zip(xs[:n], ys[:n])))
        assert m.inverse().apply(m.apply(q)) == pytest.approx(q, abs=1e-6, rel=1e-6)

    @given(xs=increasing, ys=increasing)
    def test_property_monotone_preserved(xs, ys):
        n = min(len(xs), len(ys))
        if n < 2:
            return
        m = TimeMap("a", "b", tuple(zip(xs[:n], ys[:n])))
        pts = [m.apply(x) for x in xs[:n]]
        assert all(p1 > p0 for p0, p1 in zip(pts, pts[1:]))
