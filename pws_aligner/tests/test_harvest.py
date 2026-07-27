"""Harvest executor: turn ACCEPTed proposals into a training-data ledger.

The executor is the flywheel's write-side. It is gated by the ACCEPT-precision
certification: only ACCEPTs whose stem is in ``allowed_stems`` (default
``{"regular"}`` — the only axis certified poison-free, 2026-07-18) become
harvested records. instrumental/acappella ACCEPTs are structurally excluded
until re-certified. It writes ONLY to a harvest-ledger JSONL (training data) and
keeps corrections PROPOSED — the seam's zero-canonical-mutation invariant holds.
"""

from __future__ import annotations

import json

from core.acquisition_case import TrainingSignal
from pws_aligner.cotrain_seam import (
    AgreementReport,
    Band,
    CandidateProposal,
    MixSpan,
    ProposedCorrection,
    RefCandidate,
)
from pws_aligner.harvest import (
    HarvestRecord,
    harvest,
    record_from_proposal,
    write_ledger,
)


def _accept_prop(
    *, stem: str = "regular", slot: str = "001", start: float = 100.0
) -> CandidateProposal:
    span = MixSpan(set_id="s", slot_label=slot, set_start_s=start, span_dur_s=40.0)
    cand = RefCandidate(
        recording_id="R" + slot,
        source_url="yt://x",
        source_path=f"/tmp/{slot}.flac",
        stem=stem,
    )
    span_key = f"s:{slot}@{start:.1f}s"
    report = AgreementReport(
        band=Band.ACCEPT,
        fired=(),
        consensus_offset_s=12.0,
        n_agreeing=2,
        agree_tol_s=1.0,
        mean_confidence=0.72,
        detail="2 channels agree",
    )
    signal = TrainingSignal(preference_pairs=((cand.source_path, span_key),))
    return CandidateProposal(
        candidate=cand, span=span, agreement=report, training_signal=signal
    )


def _band_prop(band: Band, *, stem: str = "regular") -> CandidateProposal:
    span = MixSpan(set_id="s", slot_label="009", set_start_s=200.0, span_dur_s=40.0)
    cand = RefCandidate(recording_id="R9", source_url="yt://y", stem=stem)
    report = AgreementReport(
        band=band,
        fired=(),
        consensus_offset_s=5.0,
        n_agreeing=1,
        agree_tol_s=3.0,
        mean_confidence=0.5,
        detail="",
    )
    return CandidateProposal(candidate=cand, span=span, agreement=report)


# ── record_from_proposal: the axis + band gate ───────────────────────────────


def test_regular_accept_is_harvested():
    rec = record_from_proposal(_accept_prop(stem="regular"))
    assert isinstance(rec, HarvestRecord)
    assert rec.stem == "regular"
    assert rec.span_key == "s:001@100.0s"
    assert rec.winner == "/tmp/001.flac"
    assert rec.consensus_offset_s == 12.0
    assert rec.preference_pairs == (("/tmp/001.flac", "s:001@100.0s"),)


def test_instrumental_accept_is_NOT_harvested_by_default():
    # instrumental is NOT certified poison-free (2026-07-18) → excluded.
    assert record_from_proposal(_accept_prop(stem="instrumental")) is None


def test_instrumental_harvested_when_explicitly_allowed():
    rec = record_from_proposal(
        _accept_prop(stem="instrumental"),
        allowed_stems=frozenset({"regular", "instrumental"}),
    )
    assert rec is not None and rec.stem == "instrumental"


def test_review_is_not_harvested():
    assert record_from_proposal(_band_prop(Band.REVIEW)) is None


def test_abstain_is_not_harvested():
    assert record_from_proposal(_band_prop(Band.ABSTAIN)) is None


# ── write_ledger: JSONL shape + idempotent dedup by span_key ──────────────────


def test_write_ledger_jsonl_shape(tmp_path):
    ledger = tmp_path / "harvest.jsonl"
    n = write_ledger([record_from_proposal(_accept_prop())], ledger)
    assert n == 1
    lines = ledger.read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    for key in ("span_key", "set_id", "slot_label", "recording_id", "winner", "stem"):
        assert key in row


def test_write_ledger_dedups_by_span_key(tmp_path):
    ledger = tmp_path / "harvest.jsonl"
    rec = record_from_proposal(_accept_prop())
    assert write_ledger([rec], ledger) == 1
    # re-writing the SAME span_key appends nothing (idempotent harvest)
    assert write_ledger([rec], ledger) == 0
    assert len(ledger.read_text().strip().splitlines()) == 1


# ── harvest: end-to-end over mixed proposals ─────────────────────────────────


def _agree(rid, sources, off=12.0):
    from alignment.harness.contract import AlignmentResult

    return [
        AlignmentResult(
            recording_id=rid, offset_s=off + 0.1 * i, confidence=0.8, source=s
        )
        for i, s in enumerate(sources)
    ]


def test_harvest_regular_kept_on_two_channel_agreement():
    # regular is certified at 2-channel agreement (fp+chroma).
    span = MixSpan("s", "001", 100.0, 40.0)
    reg = RefCandidate(
        recording_id="Rreg", source_url="a", source_path="/r.flac", stem="regular"
    )
    scorer = lambda cand, sp: _agree("Rreg", ("fp", "chroma"))
    records = harvest([(reg, span, None)], scorer)
    assert [r.recording_id for r in records] == ["Rreg"]


def test_harvest_instrumental_requires_unanimous_three_channels():
    # CERTIFIED_POLICY bands instrumental at min_agreeing=3 (the poison-free band,
    # 2026-07-18). Two agreeing channels is NOT enough; three is.
    span = MixSpan("s", "002", 200.0, 40.0)
    ins = RefCandidate(
        recording_id="Rins", source_url="b", source_path="/i.flac", stem="instrumental"
    )

    two = lambda cand, sp: _agree("Rins", ("fp", "chroma"))
    assert harvest([(ins, span, None)], two) == []  # 2/3 → REVIEW, not harvested

    three = lambda cand, sp: _agree("Rins", ("fp", "chroma", "continuity"))
    recs = harvest([(ins, span, None)], three)
    assert [r.recording_id for r in recs] == ["Rins"]


def test_harvest_drops_uncertified_stem():
    # acappella is not in CERTIFIED_POLICY → never harvested, even if it agrees.
    span = MixSpan("s", "003", 300.0, 40.0)
    aca = RefCandidate(
        recording_id="Raca", source_url="c", source_path="/a.flac", stem="acappella"
    )
    scorer = lambda cand, sp: _agree("Raca", ("hubert", "chroma", "continuity"))
    assert harvest([(aca, span, None)], scorer) == []
