from __future__ import annotations

"""Tests for the claimed_axes_identity labeling function (A4).

Pure metadata: endorses the claimed identity (claimed_version + claimed_stem)
with a fixed prior confidence. Abstains when both claims are absent.
"""

from pws_aligner.identity_lf import claimed_identity_vote
from pws_aligner.votes import AbstainReason


def test_vote_endorses_when_version_present():
    """claimed_version present → non-abstaining vote endorsing the recording."""
    v = claimed_identity_vote(
        span_id="span_1",
        recording_id="rec_abc",
        claimed_version="remix",
        claimed_stem=None,
    )
    assert not v.abstained
    assert v.label == "identity_endorsed"
    assert v.lf == "claimed_axes_identity"
    assert v.span_id == "span_1"
    assert 0.0 < v.confidence <= 1.0


def test_vote_endorses_when_stem_present():
    """claimed_stem present (version absent) → endorsing vote."""
    v = claimed_identity_vote(
        span_id="span_2",
        recording_id="rec_xyz",
        claimed_version=None,
        claimed_stem="acappella",
    )
    assert not v.abstained
    assert v.label == "identity_endorsed"


def test_vote_endorses_when_both_present():
    """Both claims present → single endorsing vote (not doubled)."""
    v = claimed_identity_vote(
        span_id="span_3",
        recording_id="rec_123",
        claimed_version="original",
        claimed_stem="instrumental",
    )
    assert not v.abstained
    assert v.label == "identity_endorsed"
    assert 0.0 < v.confidence <= 1.0


def test_vote_abstains_when_both_absent():
    """Both claims absent → abstain (no signal to endorse)."""
    v = claimed_identity_vote(
        span_id="span_4",
        recording_id="rec_000",
        claimed_version=None,
        claimed_stem=None,
    )
    assert v.abstained
    assert v.reason == AbstainReason.NO_DATA


def test_vote_abstains_on_empty_strings():
    """Empty strings treated as absent → abstain."""
    v = claimed_identity_vote(
        span_id="span_5",
        recording_id="rec_000",
        claimed_version="",
        claimed_stem="",
    )
    assert v.abstained
    assert v.reason == AbstainReason.NO_DATA


def test_vote_carries_recording_id():
    """The endorsing vote must carry the recording_id for the label model to key on."""
    v = claimed_identity_vote(
        span_id="span_6",
        recording_id="rec_99",
        claimed_version="remix",
        claimed_stem=None,
    )
    assert v.recording_id == "rec_99"


def test_confidence_is_fixed_prior():
    """Confidence is a fixed prior (all present-claim votes share the same value)."""
    v1 = claimed_identity_vote("s1", "r1", "remix", None)
    v2 = claimed_identity_vote("s2", "r2", "original", "acappella")
    assert v1.confidence == v2.confidence
