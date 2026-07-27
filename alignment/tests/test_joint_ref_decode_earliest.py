"""Routing tests for the decoder-specific earliest-instance prior."""

from __future__ import annotations

from alignment import joint_ref_decode


def test_looptrace_prior_does_not_leak_into_legacy_jobs():
    assert joint_ref_decode._legacy_r0_slope("looptrace", 0.003) == 0.0


def test_legacy_decoder_receives_its_position_prior():
    assert joint_ref_decode._legacy_r0_slope("legacy", 0.003) == 0.003


def test_looptrace_prior_is_scoped_to_acappella_population():
    assert joint_ref_decode._looptrace_earliest_slope("acappella", 0.003) == 0.003
    assert joint_ref_decode._looptrace_earliest_slope("instrumental", 0.003) == 0.0
    assert joint_ref_decode._looptrace_earliest_slope("regular", 0.003) == 0.0
