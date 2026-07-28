from __future__ import annotations

"""Tests for the ordering_soft labeling function (A5).

Tracklist order today is a HARD monotonic constraint
(sequence_decode.monotonic_decode). This LF re-expresses it as a SOFT vote:
a prior that the DJ played tracks in listed order, which strong audio
evidence can outvote (reorders / medleys / B2B — the Slide −746s poisoning).

The hard decode path is untouched; this LF ships alongside it.
"""

from pws_aligner.lf.ordering import ordering_soft_vote
from pws_aligner.core.votes import AbstainReason


def test_in_order_placement_high_vote():
    """prev < own < next → high in-order confidence."""
    v = ordering_soft_vote(
        span_id="s2", own_start_s=120.0, prev_start_s=60.0, next_start_s=180.0
    )
    assert not v.abstained
    assert v.label == "in_order"
    assert v.confidence >= 0.7
    assert v.lf == "ordering_soft"
    assert v.span_id == "s2"


def test_backward_step_low_vote_not_hard_reject():
    """A backward step vs prev → LOW vote, but soft: nonzero, not abstained."""
    v = ordering_soft_vote(
        span_id="s3", own_start_s=50.0, prev_start_s=120.0, next_start_s=None
    )
    assert not v.abstained
    assert v.label == "out_of_order"
    assert 0.0 < v.confidence < 0.5


def test_larger_backward_step_scores_lower():
    """Confidence decays with the magnitude of the violation (Slide −746s ≪ −10s)."""
    small = ordering_soft_vote(
        span_id="a", own_start_s=110.0, prev_start_s=120.0, next_start_s=None
    )
    huge = ordering_soft_vote(
        span_id="b", own_start_s=120.0 - 746.0, prev_start_s=120.0, next_start_s=None
    )
    assert huge.confidence < small.confidence
    assert huge.confidence > 0.0  # never a hard reject


def test_small_jitter_within_tolerance_still_in_order():
    """A sub-tolerance backward nudge (measurement noise) is not a violation."""
    v = ordering_soft_vote(
        span_id="s4", own_start_s=119.5, prev_start_s=120.0, next_start_s=None
    )
    assert v.label == "in_order"
    assert v.confidence >= 0.7


def test_next_neighbour_violation_also_penalized():
    """own > next is the same violation seen from the other side."""
    v = ordering_soft_vote(
        span_id="s5", own_start_s=300.0, prev_start_s=None, next_start_s=200.0
    )
    assert not v.abstained
    assert v.label == "out_of_order"
    assert v.confidence < 0.5


def test_only_prev_present_in_order_fires():
    """A single neighbour is enough signal to vote."""
    v = ordering_soft_vote(
        span_id="s6", own_start_s=200.0, prev_start_s=100.0, next_start_s=None
    )
    assert not v.abstained
    assert v.label == "in_order"


def test_no_neighbours_abstains():
    """No neighbour placements → nothing to compare against → abstain NO_DATA."""
    v = ordering_soft_vote(
        span_id="s7", own_start_s=100.0, prev_start_s=None, next_start_s=None
    )
    assert v.abstained
    assert v.reason == AbstainReason.NO_DATA


def test_confidence_bounded():
    """Confidence always in (0, 1] for non-abstaining votes."""
    for own, prev, nxt in [
        (120.0, 60.0, 180.0),
        (50.0, 120.0, None),
        (0.0, 5000.0, None),
    ]:
        v = ordering_soft_vote(
            span_id="sx", own_start_s=own, prev_start_s=prev, next_start_s=nxt
        )
        assert 0.0 < v.confidence <= 1.0
