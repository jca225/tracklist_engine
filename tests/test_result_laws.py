"""Tier-2 verification: the Result monad must obey the monad + functor laws.

core/result.py is the errors-as-values substrate the whole pipeline composes on
(Result[AudioAsset, DownloadError], etc.). If flat_map isn't associative or the
identities don't hold, every `.map(...).flat_map(...)` chain is on sand. These
are property tests (Hypothesis) over the unbounded value/error domain — a search
for a counterexample to algebraic laws, with `Ok` as the monadic unit and
`flat_map` as bind.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from core.result import Err, Ok

# A family of int -> Result arrows that exercise both branches.
_F = [
    lambda x: Ok(x + 1),
    lambda x: Ok(x * 2),
    lambda x: Err(f"neg:{x}") if x < 0 else Ok(x),
    lambda x: Err("boom"),
]
_arrows = st.sampled_from(_F)
_ints = st.integers(min_value=-1000, max_value=1000)
_results = st.one_of(_ints.map(Ok), st.text(max_size=8).map(Err))


# ── monad laws (unit = Ok, bind = flat_map) ───────────────────────────────────


@given(a=_ints, f=_arrows)
def test_left_identity(a, f):
    # Ok(a) >>= f  ==  f(a)
    assert Ok(a).flat_map(f) == f(a)


@given(m=_results)
def test_right_identity(m):
    # m >>= Ok  ==  m
    assert m.flat_map(Ok) == m


@given(m=_results, f=_arrows, g=_arrows)
def test_associativity(m, f, g):
    # (m >>= f) >>= g  ==  m >>= (\x -> f x >>= g)
    left = m.flat_map(f).flat_map(g)
    right = m.flat_map(lambda x: f(x).flat_map(g))
    assert left == right


# ── functor laws (map) ────────────────────────────────────────────────────────


@given(m=_results)
def test_functor_identity(m):
    assert m.map(lambda x: x) == m


@given(m=_results)
def test_functor_composition(m):
    f = lambda x: x + 3
    g = lambda x: x * 5
    assert m.map(f).map(g) == m.map(lambda x: g(f(x)))


# ── branch absorption (the Either structure) ──────────────────────────────────


@given(a=_ints, f=_arrows)
def test_ok_absorbs_map_err(a, f):
    assert Ok(a).map_err(f) == Ok(a)


@given(e=st.text(max_size=8), f=_arrows)
def test_err_absorbs_map_and_flat_map(e, f):
    assert Err(e).map(f) == Err(e)
    assert Err(e).flat_map(f) == Err(e)


@given(a=_ints, e=st.text(max_size=8), default=_ints)
def test_unwrap_or(a, e, default):
    assert Ok(a).unwrap_or(default) == a
    assert Err(e).unwrap_or(default) == default
