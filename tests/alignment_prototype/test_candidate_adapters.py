from __future__ import annotations

import json

from workspaces.alignment_prototype.candidate_arbiter.adapters import (
    candidates_from_timeline,
)


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
