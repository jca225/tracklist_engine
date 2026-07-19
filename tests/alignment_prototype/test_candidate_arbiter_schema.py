from __future__ import annotations

import json

import pytest

from workspaces.alignment_prototype.candidate_arbiter.io import (
    baseline_candidates,
    read_candidate_bank,
    sha256_file,
    write_baseline_bank,
    write_candidate_bank,
)
from workspaces.alignment_prototype.candidate_arbiter.schema import (
    SCHEMA_VERSION,
    CandidateBankMetadata,
    CandidateEvidence,
    PlacementCandidate,
)


def _candidate(**overrides) -> PlacementCandidate:
    values = {
        "set_id": "set",
        "slot_label": "001w2",
        "recording_id": "rid",
        "claimed_stem": "instrumental",
        "source": "baseline",
        "rank": 0,
        "set_start_s": 12.5,
        "ref_start_s": None,
        "native_confidence": None,
        "evidence": CandidateEvidence(baseline_delta_s=0.0),
    }
    values.update(overrides)
    return PlacementCandidate(**values)


def test_candidate_key_normalizes_slot() -> None:
    assert _candidate().key == ("set", "1w2", "rid", "baseline", 0)


def test_candidate_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="set_start_s"):
        _candidate(set_start_s=float("nan"))
    with pytest.raises(ValueError, match="ridge_peak"):
        _candidate(
            evidence=CandidateEvidence(
                baseline_delta_s=0.0, ridge_peak=float("inf")
            )
        )


def test_candidate_bank_jsonl_round_trip(tmp_path) -> None:
    path = tmp_path / "bank.jsonl"
    metadata = CandidateBankMetadata(producer_sha="abc123")
    candidates = (
        _candidate(),
        _candidate(source="fp", rank=1, native_confidence=0.8),
    )

    write_candidate_bank(path, metadata, candidates)
    got_metadata, got_candidates = read_candidate_bank(path)

    assert got_metadata == metadata
    assert got_metadata.schema_version == SCHEMA_VERSION
    assert got_candidates == candidates
    assert json.loads(path.read_text().splitlines()[0])["record_type"] == "metadata"


def test_baseline_extraction_is_read_only_and_complete() -> None:
    timeline = {
        "set_id": "set",
        "spans": [
            {
                "slot_label": "001",
                "recording_id": "a",
                "claimed_stem": "regular",
                "set_start_s": 10.0,
                "ref_start_s": 2.0,
                "name": "A",
            },
            {
                "slot_label": "002w1",
                "recording_id": "b",
                "claimed_stem": "acappella",
                "set_start_s": 20.0,
                "name": "B",
            },
        ],
    }
    before = json.dumps(timeline, sort_keys=True)

    candidates = baseline_candidates(timeline)

    assert len(candidates) == len(timeline["spans"])
    assert [c.source for c in candidates] == ["baseline", "baseline"]
    assert [c.rank for c in candidates] == [0, 0]
    assert [c.evidence.baseline_delta_s for c in candidates] == [0.0, 0.0]
    assert json.dumps(timeline, sort_keys=True) == before


def test_candidate_bank_rejects_duplicate_stable_keys(tmp_path) -> None:
    path = tmp_path / "bank.jsonl"
    c = _candidate()
    with pytest.raises(ValueError, match="duplicate candidate key"):
        write_candidate_bank(
            path,
            CandidateBankMetadata(producer_sha="abc123"),
            (c, c),
        )


def test_write_baseline_bank_stamps_source_hash(tmp_path) -> None:
    timeline_path = tmp_path / "timeline.json"
    timeline_path.write_text(
        json.dumps(
            {
                "set_id": "set",
                "spans": [
                    {
                        "slot_label": "1",
                        "recording_id": "rid",
                        "claimed_stem": "regular",
                        "set_start_s": 1.0,
                    }
                ],
            }
        )
    )
    output = tmp_path / "bank.jsonl"

    write_baseline_bank(timeline_path, output, producer_sha="deadbeef")
    metadata, candidates = read_candidate_bank(output)

    assert metadata.producer_sha == "deadbeef"
    assert metadata.source_timeline_sha256 == sha256_file(timeline_path)
    assert len(candidates) == 1
