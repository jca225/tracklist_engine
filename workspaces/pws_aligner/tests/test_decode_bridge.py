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
        bin_s=2.0,
    )

    assert tl_path.exists(), "timeline file must be written"
    doc = json.loads(tl_path.read_text())
    assert doc["set_id"] == set_id
    spans = doc["spans"]
    assert len(spans) == 2

    # Each span has required fields, including slot_label on every span
    for s in spans:
        assert "recording_id" in s
        assert "set_start_s" in s
        assert "ref_start_s" in s
        assert "abstain" in s
        assert "slot_label" in s

    # span 1: majority of evidence for r1 -> should pick r1 (not abstain)
    # fixture probes: fp votes offset_s=8.0 (bin 4, bin*2.0=8.0s),
    #                 mert votes offset_s=10.0 (bin 5, bin*2.0=10.0s)
    # MAP bin * 2.0 gives the winning offset; ref_start_s = set_start_s + offset_s.
    s1 = next(s for s in spans if s["slot_label"] == "1")
    assert s1["abstain"] is False
    assert s1["recording_id"] == "r1"
    # ref_start_s must be set_start_s (10.0) + winning offset_s from MAP bin.
    # fp wins the MAP: span 2 corroborates fp as the reliable probe, so EM tips
    # span 1 to fp's bin 4 -> offset 8.0 -> 10.0 + 8.0 = 18.0 exactly.
    assert s1["ref_start_s"] == 18.0
    assert s1["confidence"] > 0.0


def _make_all_abstain_votes_file(tmp_path, set_id: str = "abstain_set"):
    """Write a probe_votes file where every probe on the single span abstains."""
    votes_path = tmp_path / f"{set_id}_probe_votes.json"
    payload = [
        {
            "span_id": "1",
            "slot_label": "1",
            "recording_id": "r1",
            "claimed_stem": "regular",
            "set_start_s": 20.0,
            "set_end_s": 60.0,
            "ref_start_s": 5.0,
            "ref_end_s": 45.0,
            "confidence": 0.0,
            "name": "Track A",
            "probes": [
                {
                    "probe": "fp",
                    "recording_id": None,
                    "offset_s": 0.0,
                    "confidence": 0.0,
                    "abstain": True,
                    "features": [],
                },
                {
                    "probe": "mert",
                    "recording_id": None,
                    "offset_s": 0.0,
                    "confidence": 0.0,
                    "abstain": True,
                    "features": [],
                },
            ],
        },
    ]
    votes_path.write_text(json.dumps(payload))
    return votes_path


def test_abstain_span_roundtrips_msgspec(tmp_path):
    """Regression test for the Critical: abstain spans must not emit recording_id=null.

    TimelineSpan.recording_id is a required `str` field in PredictedTimeline;
    a JSON null raises msgspec.ValidationError before any span is scored.
    The fix writes "" as the empty-string sentinel so the JSON is scorer-loadable
    (scores as an identity miss, which is honest semantics for "no answer").
    """
    import msgspec

    from core.contracts.timeline import PredictedTimeline
    from workspaces.pws_aligner.run_phase1 import run_phase1

    set_id = "abstain_set"
    votes_path = _make_all_abstain_votes_file(tmp_path, set_id)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    tl_path = run_phase1(set_id, votes_path=votes_path, out_dir=out_dir)
    assert tl_path.exists(), "pipeline must complete even when all probes abstain"

    doc = json.loads(tl_path.read_text())
    spans = doc["spans"]
    assert len(spans) == 1

    abstain_span = spans[0]
    assert abstain_span["abstain"] is True, "abstaining span must have abstain=True"
    assert abstain_span["recording_id"] == "", (
        "abstaining span must use empty-string sentinel, not null"
    )

    # The regression test: this decoder call must NOT raise ValidationError.
    # On pre-fix code it raises:
    #   ValidationError("Expected `str`, got `null` - at `$.spans[0].recording_id`")
    decoder = msgspec.json.Decoder(PredictedTimeline)
    decoded = decoder.decode(tl_path.read_bytes())
    assert decoded.set_id == set_id
    assert len(decoded.spans) == 1
    assert decoded.spans[0].recording_id == ""
