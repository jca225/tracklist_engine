from __future__ import annotations

"""Tests for the cue-time placement labeling function (tracklist_lf.py).

Written TDD-first: all tests should be RED before tracklist_lf.py exists.
"""

import pytest
from pws_aligner.tracklist_lf import (
    CUE_BIAS_S,
    CUE_INLIER_RATE,
    PlacementVote,
    cue_placement_vote,
    cue_placement_votes,
)


# ---------------------------------------------------------------------------
# Unit tests for cue_placement_vote
# ---------------------------------------------------------------------------


def test_bias_correction() -> None:
    """A cue at 100.0s should produce set_start_s == 100.0 - CUE_BIAS_S (bias correction)."""
    vote = cue_placement_vote(
        set_id="BB11",
        recording_id="rec_abc",
        slot_label="slot_1",
        cue_seconds=100.0,
    )
    assert not vote.abstained
    assert vote.set_start_s == pytest.approx(100.0 - CUE_BIAS_S)
    assert vote.recording_id == "rec_abc"
    assert vote.lf == "cue_time"


def test_abstain_on_missing_cue() -> None:
    """cue_seconds=None must produce an abstained vote with reason='no_cue'."""
    vote = cue_placement_vote(
        set_id="BB11",
        recording_id="rec_abc",
        slot_label="slot_1",
        cue_seconds=None,
    )
    assert vote.abstained
    assert vote.reason == "no_cue"
    assert vote.set_start_s is None


def test_abstain_on_zero_sentinel() -> None:
    """cue_seconds=0.0 is the 'no cue / mix start' sentinel and must abstain."""
    vote = cue_placement_vote(
        set_id="BB11",
        recording_id="rec_abc",
        slot_label="slot_1",
        cue_seconds=0.0,
    )
    assert vote.abstained
    assert vote.reason == "no_cue"
    assert vote.set_start_s is None


def test_confidence_is_inlier_prior() -> None:
    """A fired (non-abstained) cue's confidence must equal CUE_INLIER_RATE and be in [0,1]."""
    vote = cue_placement_vote(
        set_id="BB12",
        recording_id="rec_xyz",
        slot_label="slot_5",
        cue_seconds=42.5,
    )
    assert not vote.abstained
    assert vote.confidence == pytest.approx(CUE_INLIER_RATE)
    assert 0.0 <= vote.confidence <= 1.0


def test_keyed_on_recording_id_not_slot() -> None:
    """Two rows with the SAME slot_label but DIFFERENT recording_id must produce
    two distinct votes each carrying their own recording_id.

    This proves that recording_id is the join key, not the slot index.
    The BB12 slot-numbering trap: timeline slots max at 42 while GT slots max at 155;
    joining on slot_label would silently corrupt multiseg sets.
    """
    vote_a = cue_placement_vote(
        set_id="BB12",
        recording_id="rec_001",
        slot_label="slot_3",
        cue_seconds=60.0,
    )
    vote_b = cue_placement_vote(
        set_id="BB12",
        recording_id="rec_002",
        slot_label="slot_3",  # same slot_label as vote_a
        cue_seconds=120.0,
    )
    # They must be distinct votes
    assert vote_a.recording_id == "rec_001"
    assert vote_b.recording_id == "rec_002"
    # slot_label is merely stored for traceability
    assert vote_a.slot_label == "slot_3"
    assert vote_b.slot_label == "slot_3"
    # The votes are genuinely different
    assert vote_a != vote_b


# ---------------------------------------------------------------------------
# Batch function tests
# ---------------------------------------------------------------------------


def test_batch_skips_rows_without_recording_id() -> None:
    """cue_placement_votes must skip rows where recording_id is None (can't key them)."""
    rows = [
        {
            "set_id": "BB11",
            "recording_id": None,
            "slot_label": "slot_0",
            "cue_seconds": 10.0,
        },
        {
            "set_id": "BB11",
            "recording_id": "rec_good",
            "slot_label": "slot_1",
            "cue_seconds": 50.0,
        },
    ]
    votes = cue_placement_votes(rows)
    assert len(votes) == 1
    assert votes[0].recording_id == "rec_good"


def test_batch_maps_all_valid_rows() -> None:
    """3 valid rows produce 3 votes with correct order and keys."""
    rows = [
        {
            "set_id": "BB12",
            "recording_id": "rec_1",
            "slot_label": "slot_0",
            "cue_seconds": 10.0,
        },
        {
            "set_id": "BB12",
            "recording_id": "rec_2",
            "slot_label": "slot_1",
            "cue_seconds": 80.0,
        },
        {
            "set_id": "BB12",
            "recording_id": "rec_3",
            "slot_label": "slot_2",
            "cue_seconds": 200.0,
        },
    ]
    votes = cue_placement_votes(rows)
    assert len(votes) == 3
    for i, (vote, row) in enumerate(zip(votes, rows)):
        assert vote.recording_id == row["recording_id"]
        assert vote.set_id == row["set_id"]
        assert vote.set_start_s == pytest.approx(row["cue_seconds"] - CUE_BIAS_S)
