"""Stage spine: ordering, Pipeline binding, deterministic config hashing."""

from __future__ import annotations

import pytest

from alignment.pipeline.stages import (
    STAGES,
    AblationSpec,
    Grain,
    Pipeline,
)


def test_stage_order_is_the_canonical_spine():
    assert STAGES == (
        "perception",
        "identity",
        "placement",
        "bridge",
        "actor",
        "assemble",
    )


def test_pipeline_binds_one_impl_per_stage():
    p = Pipeline.of("classical", actor="viterbi", bridge="none")
    assert p.bindings["actor"] == "viterbi"
    assert p.bindings["bridge"] == "none"
    # unspecified stages fall back to the "std"/single-impl default
    assert p.bindings["perception"] == "std"
    assert set(p.bindings) == set(STAGES)


def test_pipeline_rejects_unknown_stage():
    with pytest.raises(ValueError, match="unknown stage"):
        Pipeline.of("bad", frobnicate="x")


def test_config_hash_is_deterministic_and_binding_sensitive():
    a = Pipeline.of("classical", actor="viterbi")
    b = Pipeline.of("classical-renamed", actor="viterbi")  # name must not matter
    c = Pipeline.of("ml", actor="trajectory_decoder", bridge="linear")
    assert a.config_hash() == b.config_hash()  # hash is over bindings, not name
    assert a.config_hash() != c.config_hash()


def test_ablation_spec_requires_nonempty_sets_and_actors():
    with pytest.raises(ValueError, match="at least one set"):
        AblationSpec(grain=Grain.COMPOSE, sets=(), actors=("viterbi",))
    with pytest.raises(ValueError, match="at least one actor"):
        AblationSpec(grain=Grain.ISOLATE, sets=("2nvzlh2k",), actors=())


def test_ablation_spec_hash_covers_seed_and_grain():
    base = dict(sets=("2nvzlh2k",), actors=("viterbi",))
    s0 = AblationSpec(grain=Grain.COMPOSE, seed=0, **base)
    s1 = AblationSpec(grain=Grain.COMPOSE, seed=1, **base)
    s2 = AblationSpec(grain=Grain.ISOLATE, seed=0, **base)
    assert s0.config_hash() != s1.config_hash()  # seed matters
    assert s0.config_hash() != s2.config_hash()  # grain matters
