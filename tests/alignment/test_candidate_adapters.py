from __future__ import annotations

import json

from alignment.candidate_arbiter.adapters import (
    candidates_from_timeline,
    fp_candidates_for_span,
)
from alignment.mix_fp_hits import FpCandidateEvidence


def test_serialized_proposals_become_shadow_candidates() -> None:
    timeline = {
        "set_id": "set",
        "spans": [
            {
                "slot_label": "001",
                "recording_id": "rid",
                "claimed_stem": "instrumental",
                "set_start_s": 100.0,
                "ref_start_s": 4.0,
                "start_source": "agentic:lyrics",
                "probe_proposals": {
                    "cue_prior": 98.0,
                    "mert_decode": {"set_start_s": 101.0, "confidence": 0.6},
                    "fp": 140.0,
                },
            }
        ],
    }

    candidates = candidates_from_timeline(timeline)

    assert [candidate.source for candidate in candidates] == [
        "baseline",
        "cue_prior",
        "fp",
        "mert_decode",
    ]
    by_source = {candidate.source: candidate for candidate in candidates}
    assert by_source["baseline"].set_start_s == 100.0
    assert by_source["fp"].set_start_s == 140.0
    assert by_source["fp"].evidence.baseline_delta_s == 40.0
    assert by_source["mert_decode"].native_confidence == 0.6
    assert by_source["mert_decode"].evidence.baseline_source == "agentic:lyrics"
    assert set(by_source["mert_decode"].evidence.agreeing_sources) == {
        "baseline",
        "cue_prior",
        "mert_decode",
    }


def test_shadow_extraction_does_not_mutate_timeline() -> None:
    timeline = {
        "set_id": "set",
        "spans": [
            {
                "slot_label": "1",
                "recording_id": "rid",
                "set_start_s": 10.0,
                "probe_proposals": {"fp": 12.0},
            }
        ],
    }
    before = json.dumps(timeline, sort_keys=True)

    candidates_from_timeline(timeline)

    assert json.dumps(timeline, sort_keys=True) == before


def test_missing_or_abstained_proposals_are_not_candidates() -> None:
    timeline = {
        "set_id": "set",
        "spans": [
            {
                "slot_label": "1",
                "recording_id": "rid",
                "set_start_s": 10.0,
                "probe_proposals": {
                    "fp": None,
                    "lyrics": {"set_start_s": None, "confidence": 0.0},
                },
            }
        ],
    }

    candidates = candidates_from_timeline(timeline)

    assert [candidate.source for candidate in candidates] == ["baseline"]


def test_fp_native_evidence_maps_to_candidate_contract() -> None:
    span = {
        "slot_label": "001",
        "recording_id": "rid",
        "claimed_stem": "instrumental",
        "set_start_s": 100.0,
    }
    native = (
        FpCandidateEvidence(
            rank=0,
            set_start_s=90.0,
            set_end_s=110.0,
            offset_s=4.0,
            votes=120,
            vote_density=6.0,
            runner_up_ratio=2.5,
        ),
    )

    candidates = fp_candidates_for_span("set", span, native)

    candidate = candidates[0]
    assert candidate.rank == 0
    assert candidate.ref_start_s == 94.0
    assert candidate.evidence.baseline_delta_s == -10.0
    assert candidate.evidence.vote_count == 120
    assert candidate.evidence.vote_density == 6.0
    assert candidate.evidence.runner_up_ratio == 2.5
