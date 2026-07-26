"""GPU-free unit tests for the identity candidate-pool core (Phase 1B, Part A)."""

from __future__ import annotations

from workspaces.alignment_prototype.candidate_pool import (
    CLAIM_SOURCE,
    POOL_SOURCE,
    as_slot_pools,
    build_pool,
    build_pools,
    pool_stem_for,
)

_SET_POOL = {
    "acappella": ("aca1", "aca2", "aca3"),
    "instrumental": ("ins1", "ins2"),
}


def test_pool_stem_routing_two_regimes():
    # acappella identifies against vocals; regular + instrumental both against
    # the instrumental backbone (decision #22: no third regime).
    assert pool_stem_for("acappella") == "acappella"
    assert pool_stem_for("instrumental") == "instrumental"
    assert pool_stem_for("regular") == "instrumental"
    assert pool_stem_for(None) == "instrumental"
    assert pool_stem_for("weird") == "instrumental"


def test_claim_leads_and_is_present():
    p = build_pool("1", "aca2", "acappella", _SET_POOL)
    # claim first (the "no override" default) ...
    assert p.candidates[0].recording_id == "aca2"
    assert p.candidates[0].ref_source == CLAIM_SOURCE
    # ... and guaranteed present exactly once (deduped out of the ref pool).
    ids = [c.recording_id for c in p.candidates]
    assert ids.count("aca2") == 1
    assert set(ids) == {"aca1", "aca2", "aca3"}
    assert p.claim is not None and p.claim.recording_id == "aca2"


def test_routes_acappella_vs_instrumental():
    aca = build_pool("1", "x", "acappella", _SET_POOL)
    assert aca.pool_stem == "acappella"
    assert {c.recording_id for c in aca.candidates} == {"x", "aca1", "aca2", "aca3"}

    reg = build_pool("2", "y", "regular", _SET_POOL)
    assert reg.pool_stem == "instrumental"
    assert {c.recording_id for c in reg.candidates} == {"y", "ins1", "ins2"}


def test_claim_not_in_ref_pool_still_included():
    # A claim recording absent from the stem pool must still be selectable, else
    # margin_gate has no claim to fall back on ("do no harm").
    p = build_pool("1", "outsider", "acappella", _SET_POOL)
    assert p.candidates[0].recording_id == "outsider"
    assert p.size == 4  # outsider + 3 acappella refs
    # The claim keeps its declared stem even when routed to a pool of another.
    assert p.candidates[0].claimed_stem == "acappella"
    assert p.candidates[1].claimed_stem == "acappella"  # pool member takes route stem


def test_pool_members_tagged_pool_source():
    p = build_pool("1", "aca1", "acappella", _SET_POOL)
    rivals = [c for c in p.candidates if c.recording_id != "aca1"]
    assert rivals and all(c.ref_source == POOL_SOURCE for c in rivals)


def test_claimless_slot_gets_override_only_pool():
    p = build_pool("1", None, "instrumental", _SET_POOL)
    assert p.claim is None
    assert {c.recording_id for c in p.candidates} == {"ins1", "ins2"}
    assert p.is_overridable  # rivals exist, but nothing to accept-claim


def test_is_overridable_requires_a_rival():
    # Claim whose only pool is itself -> size 1 -> not overridable.
    p = build_pool("1", "solo", "instrumental", {"instrumental": ("solo",)})
    assert p.size == 1
    assert not p.is_overridable


def test_build_pools_shares_pool_across_concurrent_rows():
    rows = (
        {"slot_label": "1", "recording_id": "aca1", "claimed_stem": "acappella"},
        {"slot_label": "1", "recording_id": "aca1", "claimed_stem": "acappella"},
        {"slot_label": "2", "recording_id": "ins1", "claimed_stem": "regular"},
    )
    pools = build_pools(rows, _SET_POOL)
    assert set(pools) == {"1", "2"}
    assert pools["1"].pool_stem == "acappella"
    assert pools["2"].pool_stem == "instrumental"


def test_as_slot_pools_adapter_shape():
    pools = build_pools(
        ({"slot_label": "1", "recording_id": "aca1", "claimed_stem": "acappella"},),
        _SET_POOL,
    )
    legacy = as_slot_pools(pools)
    assert set(legacy) == {"1"}
    assert isinstance(legacy["1"], tuple)
    assert legacy["1"][0].recording_id == "aca1"
    assert legacy["1"] == pools["1"].candidates


def test_empty_set_pool_falls_back_to_claim_only():
    p = build_pool("1", "aca1", "acappella", {})
    assert p.size == 1
    assert p.source_size == 0
    assert p.candidates[0].recording_id == "aca1"
