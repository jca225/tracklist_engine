"""TDD tests for decode_bridge.posterior_to_placement."""

from __future__ import annotations

import json

import pytest

from workspaces.pws_aligner.hypotheses import Hypothesis
from workspaces.pws_aligner.decode_bridge import posterior_to_placement


def test_map_hypothesis_becomes_placement():
    post = {
        Hypothesis(None, 0): 0.1,
        Hypothesis("r1", 60): 0.7,
        Hypothesis("r2", 150): 0.2,
    }
    span = posterior_to_placement("span0", post, bin_s=2.0)
    assert span["recording_id"] == "r1"
    assert span["offset_s"] == 120.0  # bin 60 * 2.0
    assert abs(span["confidence"] - 0.7) < 1e-9
    assert span["abstain"] is False


def test_null_winner_abstains():
    post = {Hypothesis(None, 0): 0.6, Hypothesis("r1", 60): 0.4}
    span = posterior_to_placement("span0", post)
    assert span["abstain"] is True


def test_span_id_in_output():
    post = {Hypothesis("r1", 5): 1.0}
    span = posterior_to_placement("spanXYZ", post, bin_s=2.0)
    assert span["span_id"] == "spanXYZ"


# ---------------------------------------------------------------------------
# Votes-file -> timeline path (synthetic, no audio)
# ---------------------------------------------------------------------------


def _make_votes_file(tmp_path, set_id: str = "test_set") -> tuple:
    """Write a minimal probe_votes JSON and return (path, expected recording_id)."""
    votes_path = tmp_path / f"{set_id}_probe_votes.json"
    payload = [
        {
            "span_id": "1",
            "slot_label": "1",
            "recording_id": "r1",
            "claimed_stem": "regular",
            "set_start_s": 10.0,
            "set_end_s": 50.0,
            "ref_start_s": 0.0,
            "ref_end_s": 40.0,
            "confidence": 0.0,
            "name": "Track 1",
            "probes": [
                {
                    "probe": "fp",
                    "recording_id": "r1",
                    "offset_s": 8.0,
                    "confidence": 0.8,
                    "abstain": False,
                    "features": [0.8, 1.2, 0.5],
                },
                {
                    "probe": "mert",
                    "recording_id": "r1",
                    "offset_s": 10.0,
                    "confidence": 0.6,
                    "abstain": False,
                    "features": [0.6, 0.9, 0.4],
                },
            ],
        },
        {
            "span_id": "2",
            "slot_label": "2",
            "recording_id": "r2",
            "claimed_stem": "regular",
            "set_start_s": 50.0,
            "set_end_s": 90.0,
            "ref_start_s": 0.0,
            "ref_end_s": 40.0,
            "confidence": 0.0,
            "name": "Track 2",
            "probes": [
                {
                    "probe": "fp",
                    "recording_id": "r2",
                    "offset_s": 50.0,
                    "confidence": 0.9,
                    "abstain": False,
                    "features": [0.9, 2.0, 0.7],
                },
            ],
        },
    ]
    votes_path.write_text(json.dumps(payload))
    return votes_path, "r1"


def test_votes_file_to_timeline(tmp_path):
    """End-to-end: votes file -> aggregation -> pws_timeline.json."""
    from workspaces.pws_aligner.run_phase1 import run_phase1

    set_id = "test_set"
    votes_path, _ = _make_votes_file(tmp_path, set_id)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    tl_path = run_phase1(
        set_id,
        votes_path=votes_path,
        out_dir=out_dir,
    )

    assert tl_path.exists(), "timeline file must be written"
    doc = json.loads(tl_path.read_text())
    assert doc["set_id"] == set_id
    spans = doc["spans"]
    assert len(spans) == 2

    # Each span has required fields
    for s in spans:
        assert "recording_id" in s
        assert "set_start_s" in s
        assert "ref_start_s" in s
        assert "abstain" in s

    # span 1: majority of evidence for r1 → should pick r1 (not abstain)
    s1 = next(s for s in spans if s["slot_label"] == "1")
    assert s1["abstain"] is False
    assert s1["recording_id"] == "r1"
