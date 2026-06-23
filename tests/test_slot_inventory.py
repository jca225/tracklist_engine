"""Tests for slot satisfaction / inventory model."""

from __future__ import annotations

import json
from pathlib import Path

from core.audio_resolve import TIER_EXACT, TIER_STEM_FALLBACK
from core.slot_inventory import (
    ActionKind,
    SatisfactionStatus,
    derive_layer_role,
    evaluate_slot,
    format_satisfaction_report,
    is_blocking,
    primary_slot_for,
    slot_claim_from_row,
    suggest_actions,
)

FIX = Path(__file__).parent / "fixtures"


def _c(taid, stem="regular", variant="regular", platform="youtube", is_reference=0):
    return dict(
        track_audio_id=taid,
        track_id="tid",
        recording_id="tid",
        stem=stem,
        variant=variant,
        platform=platform,
        is_reference=is_reference,
    )


def test_derive_layer_role_bb11():
    assert derive_layer_role("013", is_concurrent=True) == "bed"
    assert (
        derive_layer_role("013w1", is_concurrent=True, claimed_stem="regular")
        == "payload"
    )
    assert derive_layer_role("013w2", is_concurrent=True) == "constituent"
    assert derive_layer_role("013w1", claimed_stem="acappella") == "payload"
    assert derive_layer_role("005", is_concurrent=False) == "solo"


def test_primary_slot_for():
    assert primary_slot_for("013w1") == "013"
    assert primary_slot_for("013") is None


def test_slot_claim_from_row():
    claim = slot_claim_from_row(
        {
            "set_id": "2nvzlh2k",
            "slot_label": "013w1",
            "row_index": 41,
            "recording_id": "1qrzf9p",
            "track_id": "1qrzf9p",
            "claimed_stem": "regular",
            "claimed_variant": "regular",
            "full_name": "DJ Kool - Let Me Clear My Throat",
            "is_concurrent": 1,
        }
    )
    assert claim.layer_role == "payload"
    assert claim.bed_slot == "013"


def test_evaluate_satisfied_exact():
    claim = slot_claim_from_row(
        {
            "set_id": "x",
            "slot_label": "001",
            "recording_id": "abc",
            "track_id": "abc",
            "claimed_stem": "regular",
            "claimed_variant": "regular",
            "is_concurrent": 0,
        }
    )
    sat = evaluate_slot(claim, [_c(1, is_reference=1)])
    assert sat.status == SatisfactionStatus.SATISFIED
    assert sat.resolve_tier == TIER_EXACT


def test_evaluate_missing():
    claim = slot_claim_from_row(
        {
            "set_id": "x",
            "slot_label": "002",
            "recording_id": "missing",
            "track_id": "missing",
            "claimed_stem": "regular",
            "claimed_variant": "regular",
            "is_concurrent": 0,
        }
    )
    sat = evaluate_slot(claim, [])
    assert sat.status == SatisfactionStatus.MISSING
    acts = suggest_actions(sat)
    assert acts[0].kind == ActionKind.REPLACE_VERSION


def test_payload_wrong_stem_only_regular():
    claim = slot_claim_from_row(
        {
            "set_id": "2nvzlh2k",
            "slot_label": "013w1",
            "recording_id": "1qrzf9p",
            "track_id": "1qrzf9p",
            "claimed_stem": "acappella",
            "claimed_variant": "regular",
            "is_concurrent": 1,
        }
    )
    sat = evaluate_slot(claim, [_c(99, stem="regular", is_reference=1)])
    assert sat.status == SatisfactionStatus.WRONG_STEM
    assert sat.resolve_tier == TIER_STEM_FALLBACK
    kinds = {a.kind for a in suggest_actions(sat)}
    assert ActionKind.ACQUIRE_ACAPPELLA in kinds
    assert ActionKind.USE_DEMUCS_VOCALS in kinds


def test_bb11_fixture_cases():
    doc = json.loads((FIX / "bb11_slot_013.json").read_text())
    for slot in doc["slots"]:
        claim = slot_claim_from_row(
            {
                "set_id": doc["set_id"],
                "slot_label": slot["slot_label"],
                "recording_id": slot["recording_id"],
                "track_id": slot["recording_id"],
                "claimed_stem": slot["claimed_stem"],
                "claimed_variant": slot["claimed_variant"],
                "full_name": slot.get("full_name", ""),
                "is_concurrent": int(slot.get("is_concurrent", 0)),
            }
        )
        sat = evaluate_slot(claim, slot["candidates"])
        if "expect_status" in slot:
            assert sat.status.value == slot["expect_status"]
        elif slot["slot_label"] == "013":
            assert sat.status == SatisfactionStatus.SATISFIED


def test_format_report_and_blocking():
    claim = slot_claim_from_row(
        {
            "set_id": "x",
            "slot_label": "003",
            "recording_id": "z",
            "track_id": "z",
            "claimed_stem": "regular",
            "claimed_variant": "regular",
            "is_concurrent": 0,
        }
    )
    sat = evaluate_slot(claim, [])
    text = format_satisfaction_report([sat])
    assert "003" in text
    assert is_blocking(sat)
