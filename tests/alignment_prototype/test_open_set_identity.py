"""Unit tests for the blind open-set identity scoring core (Phase 1B)."""

from __future__ import annotations

import numpy as np

from workspaces.alignment_prototype.open_set_identity import (
    CandidateScore,
    Decision,
    borda_fuse,
    chamfer_score,
    margin_gate,
    per_query_max_cos,
    rank_by_chamfer,
)


def _col(*vecs: list[float]) -> np.ndarray:
    """(D, n) matrix from column vectors."""
    return np.array(vecs, dtype=np.float64).T


def test_per_query_max_cos_identical_is_one():
    q = _col([1.0, 0.0], [0.0, 1.0])
    assert np.allclose(per_query_max_cos(q, q), [1.0, 1.0])


def test_per_query_max_cos_is_scale_invariant():
    q = _col([1.0, 0.0])
    ref = _col([5.0, 0.0], [0.0, 3.0])  # first ref frame is same direction, scaled
    assert np.allclose(per_query_max_cos(q, ref), [1.0])


def test_per_query_max_empty():
    assert per_query_max_cos(np.empty((2, 0)), _col([1.0, 0.0])).size == 0


def test_chamfer_unscorable_is_none_not_zero():
    # Law: unknown is None, never a fake 0.0.
    assert chamfer_score(np.empty((2, 0)), _col([1.0, 0.0])) is None


def test_chamfer_matches_better_ref_higher():
    q = _col([1.0, 0.0], [1.0, 0.0])
    good = _col([1.0, 0.0])  # aligned
    bad = _col([0.0, 1.0])  # orthogonal
    assert chamfer_score(q, good) > chamfer_score(q, bad)


def test_conditional_topk_lifts_long_contaminated_span():
    # 3 good query frames + 1 contaminated (orthogonal) frame.
    q = _col([1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0])
    ref = _col([1.0, 0.0])
    full_mean = chamfer_score(q, ref, span_s=10.0)  # short -> full mean
    topk = chamfer_score(q, ref, span_s=120.0, long_span_s=60.0, topk_frac=0.25)
    # top-25% of 4 frames = the single best (a perfect 1.0); full mean is dragged down.
    assert topk is not None and full_mean is not None
    assert topk > full_mean
    assert np.isclose(topk, 1.0)


def test_rank_by_chamfer_orders_and_trails_unscorable():
    q = _col([1.0, 0.0])
    pool = {
        "good": _col([1.0, 0.0]),
        "bad": _col([0.0, 1.0]),
        "empty": np.empty((2, 0)),
    }
    ranked = rank_by_chamfer(q, pool)
    assert ranked[0].recording_id == "good"
    assert ranked[-1].recording_id == "empty"  # None sorts last
    assert ranked[-1].score is None


def test_borda_prunes_to_agreement():
    lf1 = [CandidateScore("A", 0.9), CandidateScore("B", 0.5), CandidateScore("C", 0.1)]
    lf2 = [CandidateScore("A", 0.8), CandidateScore("C", 0.4), CandidateScore("B", 0.2)]
    fused = borda_fuse([lf1, lf2])
    assert fused[0].recording_id == "A"  # both rank A first -> top borda


def test_gate_override_when_confident_and_disagrees():
    ranked = [CandidateScore("audio_says", 0.9), CandidateScore("claim", 0.2)]
    d = margin_gate(ranked, "claim", tau=0.3, floor=0.5)
    assert d.decision == Decision.OVERRIDE
    assert d.recording_id == "audio_says"
    assert d.margin is not None and d.margin >= 0.3


def test_gate_do_no_harm_keeps_claim_below_margin():
    # top1 disagrees but only barely -> keep the claim (blind MERT is unreliable).
    ranked = [CandidateScore("audio_says", 0.55), CandidateScore("claim", 0.50)]
    d = margin_gate(ranked, "claim", tau=0.3, floor=0.5)
    assert d.decision == Decision.ACCEPT_CLAIM
    assert d.recording_id == "claim"


def test_gate_keeps_claim_when_top1_is_claim():
    ranked = [CandidateScore("claim", 0.9), CandidateScore("other", 0.3)]
    d = margin_gate(ranked, "claim", tau=0.3, floor=0.5)
    assert d.decision == Decision.ACCEPT_CLAIM
    assert d.recording_id == "claim"


def test_gate_abstains_when_no_claim_and_nothing_scorable():
    ranked = [CandidateScore("x", None), CandidateScore("y", None)]
    d = margin_gate(ranked, None, tau=0.3, floor=0.5)
    assert d.decision == Decision.ABSTAIN
    assert d.recording_id is None


def test_gate_below_floor_does_not_override():
    # top1 disagrees with a big margin but is below the confidence floor -> keep claim.
    ranked = [CandidateScore("audio_says", 0.4), CandidateScore("claim", 0.0)]
    d = margin_gate(ranked, "claim", tau=0.3, floor=0.5)
    assert d.decision == Decision.ACCEPT_CLAIM
