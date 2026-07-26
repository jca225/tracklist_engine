"""GPU-free unit tests for the Phase 1B gated identity-override decision layer."""

from __future__ import annotations

import numpy as np

from workspaces.alignment_prototype.candidate_pool import build_pool
from workspaces.alignment_prototype.identity_override import (
    decide_identity,
    resolve_identities,
    summarize,
)
from workspaces.alignment_prototype.open_set_identity import Decision

_SET_POOL = {"instrumental": ("claimed", "rival"), "acappella": ()}


def _col(*vecs: list[float]) -> np.ndarray:
    return np.array(vecs, dtype=np.float64).T


# Query points at the "rival" direction; claimed ref points orthogonal.
_RIVAL = _col([0.0, 1.0])
_CLAIMED = _col([1.0, 0.0])
_QUERY_RIVAL = _col([0.0, 1.0])
_QUERY_CLAIM = _col([1.0, 0.0])


def test_override_when_audio_confidently_disagrees():
    pool = build_pool("1", "claimed", "regular", _SET_POOL)
    r = decide_identity(
        pool,
        _QUERY_RIVAL,
        {"claimed": _CLAIMED, "rival": _RIVAL},
        tau=0.3,
        floor=0.5,
    )
    assert r.decision.decision == Decision.OVERRIDE
    assert r.decision.recording_id == "rival"
    assert r.provenance["chosen"] == "rival"
    assert r.provenance["claim"] == "claimed"
    # provenance carries the LF votes, best-first (Law 21).
    assert r.provenance["scored"][0][0] == "rival"
    assert len(r.provenance["scored"]) == 2


def test_accept_claim_when_query_agrees():
    pool = build_pool("1", "claimed", "regular", _SET_POOL)
    r = decide_identity(
        pool, _QUERY_CLAIM, {"claimed": _CLAIMED, "rival": _RIVAL}, tau=0.3, floor=0.5
    )
    assert r.decision.decision == Decision.ACCEPT_CLAIM
    assert r.decision.recording_id == "claimed"


def test_do_no_harm_when_no_query():
    pool = build_pool("1", "claimed", "regular", _SET_POOL)
    r = decide_identity(pool, None, {"claimed": _CLAIMED}, tau=0.3, floor=0.5)
    assert r.decision.decision == Decision.ACCEPT_CLAIM
    assert r.decision.recording_id == "claimed"
    assert r.provenance["reason"] == "no query"
    assert r.provenance["scored"] == []


def test_do_no_harm_when_no_ref_features():
    pool = build_pool("1", "claimed", "regular", _SET_POOL)
    r = decide_identity(pool, _QUERY_RIVAL, {}, tau=0.3, floor=0.5)
    assert r.decision.decision == Decision.ACCEPT_CLAIM
    assert r.provenance["reason"] == "no ref features"


def test_claimless_slot_abstains_without_features():
    pool = build_pool("1", None, "regular", _SET_POOL)
    r = decide_identity(pool, None, {}, tau=0.3, floor=0.5)
    assert r.decision.decision == Decision.ABSTAIN
    assert r.decision.recording_id is None


def test_unscorable_claim_is_not_overridden():
    # The claim has no features but a rival scores well -> keep the claim
    # (exercises the do-no-harm margin_gate fix through the override layer).
    pool = build_pool("1", "claimed", "regular", _SET_POOL)
    r = decide_identity(
        pool, _QUERY_RIVAL, {"rival": _RIVAL}, tau=0.3, floor=0.5
    )  # no "claimed" features
    assert r.decision.decision == Decision.ACCEPT_CLAIM
    assert r.decision.recording_id == "claimed"
    assert r.provenance["top1"] == "rival"


def test_below_floor_keeps_claim():
    # Query sits at 45deg between the refs (cos 0.707 to each) -> top1 is below a
    # high confidence floor, so no override despite a nonzero margin. (Cosine is
    # scale-invariant; "weak" means direction, not magnitude.)
    ambiguous = _col([1.0, 1.0])
    pool = build_pool("1", "claimed", "regular", _SET_POOL)
    r = decide_identity(
        pool, ambiguous, {"claimed": _CLAIMED, "rival": _RIVAL}, tau=0.05, floor=0.9
    )
    assert r.decision.decision == Decision.ACCEPT_CLAIM


def test_resolve_and_summarize_over_slots():
    pools = {
        "1": build_pool("1", "claimed", "regular", _SET_POOL),
        "2": build_pool("2", "claimed", "regular", _SET_POOL),
    }
    refs = {"claimed": _CLAIMED, "rival": _RIVAL}
    results = resolve_identities(
        pools,
        {"1": _QUERY_RIVAL, "2": _QUERY_CLAIM},
        refs,
        tau=0.3,
        floor=0.5,
    )
    counts = summarize(results)
    assert counts[Decision.OVERRIDE] == 1  # slot 1 flips to rival
    assert counts[Decision.ACCEPT_CLAIM] == 1  # slot 2 agrees
    assert counts[Decision.ABSTAIN] == 0


def test_span_s_threaded_through_to_chamfer():
    # span_s reaches chamfer_score (the conditional-top-k lift itself is proven in
    # test_open_set_identity); a clean long-span rival query still overrides.
    q = _col([0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0])  # all rival-aligned
    pool = build_pool("1", "claimed", "regular", _SET_POOL)
    r = decide_identity(
        pool,
        q,
        {"claimed": _CLAIMED, "rival": _RIVAL},
        tau=0.3,
        floor=0.5,
        span_s=120.0,
    )
    assert r.decision.decision == Decision.OVERRIDE
    assert r.decision.recording_id == "rival"
