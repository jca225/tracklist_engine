"""GPU-free unit tests for the Phase 1B extraction planner + feature bundle."""

from __future__ import annotations

import numpy as np

from workspaces.alignment_prototype.extract_stem_mert import (
    IdentityFeatureBundle,
    plan_extraction,
)
from workspaces.alignment_prototype.stem_mert import INSTRUMENTAL_LAYER, VOCAL_LAYER

_ROWS = (
    {"slot_label": "1", "claimed_stem": "acappella"},
    {"slot_label": "1", "claimed_stem": "acappella"},  # concurrent -> one query
    {"slot_label": "2", "claimed_stem": "regular"},
)
_INV = {"acappella": ("a1", "a2"), "instrumental": ("i1",)}


def test_plan_one_query_per_slot_routed_by_stem():
    plan = plan_extraction("s", _ROWS, _INV)
    queries = [it for it in plan.items if it.kind == "query"]
    assert {q.ident for q in queries} == {"1", "2"}  # concurrent rows collapsed
    by_id = {q.ident: q for q in queries}
    assert by_id["1"].pool_stem == "acappella" and by_id["1"].layer == VOCAL_LAYER
    assert (
        by_id["2"].pool_stem == "instrumental"
        and by_id["2"].layer == INSTRUMENTAL_LAYER
    )


def test_plan_covers_every_ref_with_routed_layer():
    plan = plan_extraction("s", _ROWS, _INV)
    refs = {it.ident: it for it in plan.items if it.kind == "ref"}
    assert set(refs) == {"a1", "a2", "i1"}
    assert refs["a1"].layer == VOCAL_LAYER
    assert refs["i1"].layer == INSTRUMENTAL_LAYER
    assert plan.layers == tuple(sorted({VOCAL_LAYER, INSTRUMENTAL_LAYER}))


def test_plan_persists_pool_partition():
    plan = plan_extraction("s", _ROWS, _INV, spans={"1": 90.0})
    assert plan.set_pool_by_stem == {"acappella": ("a1", "a2"), "instrumental": ("i1",)}
    assert plan.spans == {"1": 90.0}


def test_plan_ignores_unknown_stem_pools():
    plan = plan_extraction("s", _ROWS, {**_INV, "bogus": ("z1",)})
    assert "bogus" not in plan.set_pool_by_stem
    assert all(it.ident != "z1" for it in plan.items)


def test_bundle_roundtrips_through_disk(tmp_path):
    b = IdentityFeatureBundle(
        set_id="s",
        queries={"1": np.arange(6, dtype=np.float32).reshape(2, 3)},
        refs={"a1": np.ones((2, 4), dtype=np.float32)},
        set_pool_by_stem={"acappella": ("a1",), "instrumental": ()},
        spans={"1": 88.0},
    )
    b.save(tmp_path)
    loaded = IdentityFeatureBundle.load("s", tmp_path)
    assert set(loaded.queries) == {"1"}
    assert np.array_equal(loaded.queries["1"], b.queries["1"])
    assert np.array_equal(loaded.refs["a1"], b.refs["a1"])
    assert loaded.set_pool_by_stem == {"acappella": ("a1",), "instrumental": ()}
    assert loaded.spans == {"1": 88.0}


def test_bundle_load_feeds_resolve_identities(tmp_path):
    # The bundle's dicts are exactly the shapes identity_override consumes.
    from workspaces.alignment_prototype.candidate_pool import build_pools
    from workspaces.alignment_prototype.identity_override import (
        resolve_identities,
        summarize,
    )
    from workspaces.alignment_prototype.open_set_identity import Decision

    rival = np.array([[0.0], [1.0]], dtype=np.float32)  # (D=2, n=1)
    claim = np.array([[1.0], [0.0]], dtype=np.float32)
    b = IdentityFeatureBundle(
        set_id="s",
        queries={"1": rival},  # mix span looks like the rival
        refs={"claimed": claim, "rival": rival},
        set_pool_by_stem={"instrumental": ("claimed", "rival"), "acappella": ()},
        spans={},
    )
    b.save(tmp_path)
    loaded = IdentityFeatureBundle.load("s", tmp_path)

    rows = ({"slot_label": "1", "recording_id": "claimed", "claimed_stem": "regular"},)
    pools = build_pools(rows, loaded.set_pool_by_stem)
    results = resolve_identities(
        pools, loaded.queries, loaded.refs, tau=0.3, floor=0.5, spans=loaded.spans
    )
    assert summarize(results)[Decision.OVERRIDE] == 1
    assert results["1"].decision.recording_id == "rival"
