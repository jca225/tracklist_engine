"""Tier-2 verification: the GT gain-curve merge is a well-behaved union.

_merge_curves stitches per-clip fader envelopes into one set-time-ordered curve
(slivers / loop iterations / merged slots). It is the GROUND TRUTH for audibility,
so its algebra matters: merging must be order-independent and idempotent, the
output must stay sorted and free of coincident points, and — the historical
_min_frac bug — a coincident *muted* sibling must never zero out the louder one
("keep louder" on a shared boundary).
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from labeling.export_als_to_gt import _merge_curves

# x on a 0.01 grid (gaps >> the 1e-4 dedup threshold) so only *exact* duplicate
# set-times collide — keeps the float boundary out of the generic properties.
_pt = st.tuples(
    st.integers(min_value=0, max_value=100_000).map(lambda k: k / 100.0),
    st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
)
_curve = st.lists(_pt, max_size=40).map(tuple)


def _xs(curve):
    return [x for x, _ in curve]


@given(c=_curve)
def test_output_is_sorted_by_set_time(c):
    out = _merge_curves(c)
    assert _xs(out) == sorted(_xs(out))


@given(c=_curve)
def test_no_coincident_points_in_output(c):
    out = _merge_curves(c)
    xs = _xs(out)
    for a, b in zip(xs, xs[1:]):
        assert abs(b - a) >= 1e-4


@given(c=_curve)
def test_idempotent(c):
    once = _merge_curves(c)
    assert _merge_curves(once) == once  # re-merging a merged curve is a no-op


@given(a=_curve, b=_curve)
def test_commutative(a, b):
    assert _merge_curves(a, b) == _merge_curves(b, a)


@given(a=_curve, b=_curve, c=_curve)
def test_n_ary_equals_concatenated_single(a, b, c):
    # Merging N curves == merging their concatenation (union semantics).
    assert _merge_curves(a, b, c) == _merge_curves(a + b + c)


def test_empty_and_single():
    assert _merge_curves() == ()
    assert _merge_curves(()) == ()
    assert _merge_curves(((1.0, 0.5),)) == ((1.0, 0.5),)


def test_louder_wins_on_coincident_the_min_frac_regression():
    # The _min_frac bug: a muted sibling at the same set-time zeroed the slot's
    # gain. Merge must keep the LOUDER value, not the mute.
    loud = ((10.0, 1.0),)
    muted = ((10.0, 0.0),)
    assert _merge_curves(loud, muted) == ((10.0, 1.0),)
    assert _merge_curves(muted, loud) == ((10.0, 1.0),)  # order-independent


def test_louder_wins_within_dedup_threshold():
    # Two points closer than 1e-4 but not equal -> merged, louder kept.
    out = _merge_curves(((10.0, 0.2), (10.00005, 0.9)))
    assert len(out) == 1 and out[0][1] == 0.9
