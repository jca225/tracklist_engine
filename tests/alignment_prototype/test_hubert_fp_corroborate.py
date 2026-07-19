"""HuBERT↔fp corroboration + regular-stem fp loading (E1 regular-tail mass)."""

from __future__ import annotations

from workspaces.alignment_prototype.agentic.actions import DOMINANCE, REGISTRY
from workspaces.alignment_prototype.agentic.live_runners import (
    FP_HUBERT_CORROBORATE_S,
    corroborate_hubert_with_fp,
    ref_fp_for_span,
)


def test_corroborate_snaps_near_fp_peak():
    ss, rs, tag = corroborate_hubert_with_fp(120.0, 40.0, fp_ss=100.0, tol_s=30.0)
    assert ss == 100.0
    assert rs == 40.0
    assert "corroborate_fp" in tag


def test_corroborate_fp_noop_when_far_or_missing():
    assert corroborate_hubert_with_fp(200.0, 40.0, fp_ss=100.0, tol_s=30.0) == (
        200.0,
        40.0,
        "",
    )
    assert corroborate_hubert_with_fp(105.0, 40.0, fp_ss=None) == (105.0, 40.0, "")


def test_default_fp_tol_matches_constant():
    assert FP_HUBERT_CORROBORATE_S == 30.0


def test_ref_fp_for_span_skips_acappella_allows_regular_key():
    # No cache entry → None, but the stem gate must not hard-reject regular.
    assert (
        ref_fp_for_span(
            {"recording_id": "missing_rid_xyz", "claimed_stem": "acappella"}
        )
        is None
    )
    assert (
        ref_fp_for_span({"recording_id": "missing_rid_xyz", "claimed_stem": "regular"})
        is None
    )
    assert (
        ref_fp_for_span(
            {"recording_id": "missing_rid_xyz", "claimed_stem": "instrumental"}
        )
        is None
    )


def test_regular_plan_includes_fp_then_hubert():
    names = DOMINANCE["regular"]
    assert "fp" in names and "stem_hubert" in names
    assert names.index("fp") < names.index("stem_hubert")
    assert "regular" in REGISTRY["stem_hubert"].stems
