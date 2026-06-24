"""merge(): independent corroboration raises confidence; nothing-commits abstains.
This is the deterministic driver's fusion rule — the payoff of a shared contract."""

from __future__ import annotations

from workspaces.alignment_prototype.harness import AlignmentResult, merge


def _r(rec, off, conf, *, source="p", abstain=False):
    if abstain:
        return AlignmentResult.abstained(source=source, recording_id=rec)
    return AlignmentResult(
        recording_id=rec, offset_s=off, confidence=conf, source=source
    )


def test_no_live_results_abstains():
    assert merge(()).abstain
    assert merge((_r("a", 0, 0, abstain=True), _r("b", 0, 0, abstain=True))).abstain


def test_single_result_passthrough():
    out = merge((_r("rec1", 10.0, 0.7, source="fp"),))
    assert not out.abstain
    assert out.recording_id == "rec1" and out.offset_s == 10.0
    assert out.confidence == 0.7 and out.source == "merge(fp)"


def test_agreement_boosts_confidence():
    # Two probes, same recording, offsets within tolerance -> corroboration.
    out = merge(
        (_r("rec1", 10.0, 0.6, source="fp"), _r("rec1", 10.5, 0.5, source="chroma")),
        offset_tol_s=2.0,
        agreement_bonus=0.1,
    )
    assert out.confidence == 0.7  # winner 0.6 + 0.1 corroboration
    assert out.source == "merge(chroma+fp)"


def test_agreement_capped_at_one():
    rs = tuple(_r("rec1", 10.0 + i * 0.1, 0.95, source=f"p{i}") for i in range(5))
    assert merge(rs, agreement_bonus=0.5).confidence == 1.0


def test_disagreement_no_boost_winner_wins():
    # Same recording but placements disagree -> no corroboration; higher conf wins.
    out = merge(
        (_r("rec1", 10.0, 0.6, source="fp"), _r("rec1", 90.0, 0.8, source="chroma")),
        offset_tol_s=2.0,
    )
    assert out.offset_s == 90.0 and out.confidence == 0.8  # no bonus
    assert out.source == "merge(chroma)"


def test_different_recordings_do_not_corroborate():
    out = merge(
        (_r("recA", 10.0, 0.7, source="fp"), _r("recB", 10.0, 0.6, source="chroma")),
    )
    assert out.recording_id == "recA" and out.confidence == 0.7  # no cross-rec boost


def test_min_confidence_gate_abstains():
    out = merge((_r("rec1", 10.0, 0.3, source="fp"),), min_confidence=0.5)
    assert out.abstain and out.recording_id == "rec1"
