"""Impl registry: register/lookup by (stage, name), the canned pipelines, and
the frozen-stage single-impl invariant."""

from __future__ import annotations

import pytest

from workspaces.alignment_prototype.pipeline import registry as reg


def _impl(stage: str, name: str):
    obj = type("I", (), {"stage": stage, "name": name})()
    return obj


def test_register_and_lookup_roundtrip():
    r = reg.Registry()
    a = _impl("actor", "viterbi")
    r.register(a)
    assert r.get("actor", "viterbi") is a


def test_duplicate_registration_is_rejected():
    r = reg.Registry()
    r.register(_impl("actor", "viterbi"))
    with pytest.raises(ValueError, match="already registered"):
        r.register(_impl("actor", "viterbi"))


def test_unknown_stage_registration_is_rejected():
    r = reg.Registry()
    with pytest.raises(ValueError, match="unknown stage"):
        r.register(_impl("frobnicate", "x"))


def test_lookup_missing_impl_names_available_ones():
    r = reg.Registry()
    r.register(_impl("actor", "viterbi"))
    with pytest.raises(KeyError, match="viterbi"):  # error lists what IS there
        r.get("actor", "trm")


def test_list_impls_by_stage():
    r = reg.Registry()
    r.register(_impl("actor", "viterbi"))
    r.register(_impl("actor", "trm"))
    r.register(_impl("bridge", "none"))
    assert sorted(r.names("actor")) == ["trm", "viterbi"]
    assert r.names("bridge") == ["none"]


def test_default_registry_exposes_the_four_canned_pipelines():
    names = {p.name for p in reg.canned_pipelines()}
    assert {"classical", "agentic", "ml", "trm"} <= names


def test_ml_pipeline_uses_linear_bridge_and_trajectory_decoder():
    ml = {p.name: p for p in reg.canned_pipelines()}["ml"]
    assert ml.bindings["bridge"] == "linear"
    assert ml.bindings["actor"] == "trajectory_decoder"


def test_trm_pipeline_binds_the_skeleton_actor():
    trm = {p.name: p for p in reg.canned_pipelines()}["trm"]
    assert trm.bindings["actor"] == "trm"


def test_truly_frozen_stages_have_exactly_one_impl_in_default_registry():
    # perception/identity/assemble are single-impl; placement carries the
    # agentic intervention too, so it is NOT single-impl (spec §1).
    r = reg.default_registry()
    for frozen in ("perception", "identity", "assemble"):
        assert len(r.names(frozen)) == 1, f"{frozen} should be single-impl today"
    assert r.names("placement") == ["agentic", "std"]
    # the live actor surface carries several
    assert len(r.names("actor")) >= 4  # viterbi, agentic, trajectory_decoder, trm
