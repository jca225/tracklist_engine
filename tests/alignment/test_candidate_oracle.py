from __future__ import annotations

from alignment.candidate_arbiter.oracle import (
    CandidateError,
    apply_oracle_selection,
    select_oracle_candidates,
)
from alignment.candidate_arbiter.schema import (
    CandidateEvidence,
    PlacementCandidate,
)


def _candidate(source: str, start: float) -> PlacementCandidate:
    return PlacementCandidate(
        set_id="set",
        slot_label="001",
        recording_id="rid",
        claimed_stem="regular",
        source=source,
        rank=0,
        set_start_s=start,
        ref_start_s=None,
        native_confidence=None,
        evidence=CandidateEvidence(baseline_delta_s=start - 10.0),
    )


def test_oracle_selects_lowest_error_candidate() -> None:
    baseline = _candidate("baseline", 10.0)
    fp = _candidate("fp", 18.0)
    lyrics = _candidate("lyrics", 20.5)

    selected = select_oracle_candidates(
        (
            CandidateError(baseline, error_s=10.0),
            CandidateError(fp, error_s=2.0),
            CandidateError(lyrics, error_s=0.5),
        )
    )

    assert selected[("1", "rid")].source == "lyrics"


def test_apply_oracle_selection_changes_only_placement_fields() -> None:
    timeline = {
        "set_id": "set",
        "spans": [
            {
                "slot_label": "001",
                "recording_id": "rid",
                "set_start_s": 10.0,
                "set_end_s": 30.0,
                "ref_start_s": 4.0,
                "ref_segments": [{"mix_start_s": 10.0, "ref_start_s": 4.0}],
                "name": "Track",
            }
        ],
    }

    out = apply_oracle_selection(
        timeline,
        {("1", "rid"): _candidate("fp", 20.0)},
    )

    span = out["spans"][0]
    assert span["set_start_s"] == 20.0
    assert span["set_end_s"] == 40.0
    assert span["ref_start_s"] == 4.0
    assert span["ref_segments"] == timeline["spans"][0]["ref_segments"]
    assert span["oracle_candidate_source"] == "fp"
