"""Tests for the co-training seam (dry-run skeleton).

Covers, offline with a FAKE scorer:
  1. Seam 1 — suspect → AcquisitionCase production (verdict → problem class,
     clean slots skipped, stem/variant problems surfaced).
  2. Seam 2 — accept/review/abstain banding (Fellegi–Sunter), including the
     single-channel-can't-ACCEPT confirmation-drift guard.
  3. TrainingSignal + PROPOSED-correction construction on ACCEPT only.
  4. Zero canonical mutation — the seam returns proposals; nothing is written.
"""

from __future__ import annotations

from pathlib import Path

from core.acquisition_case import (
    AcquisitionCase,
    CaseStatus,
    ProblemClass,
    TrainingSignal,
)
from workspaces.alignment_prototype.harness.contract import AlignmentResult
from workspaces.pws_aligner.cotrain_seam import (
    Band,
    BandThresholds,
    CandidateProposal,
    GtCandidateCase,
    MixSpan,
    ProposedCorrection,
    RefCandidate,
    band_agreement,
    build_acquisition_case,
    build_acquisition_cases,
    evaluate_banding,
    propose_from_candidate,
    real_probe_scorer,
    slot_is_suspect,
)
from workspaces.pws_aligner.cotrain_seam import SlotSignal


# ── Seam 1: suspect → AcquisitionCase ─────────────────────────────────────────


def _sig(**kw) -> SlotSignal:
    base = dict(slot_label="027", recording_id="tlp1", verdict="NO_REFERENCE")
    base.update(kw)
    return SlotSignal(**base)


def test_clean_verdict_is_not_suspect():
    assert not slot_is_suspect(_sig(verdict="OK"))
    assert not slot_is_suspect(_sig(verdict="SKIPPED"))
    assert build_acquisition_case("2nvzlh2k", _sig(verdict="OK")) is None


def test_wrong_song_opens_case_with_problem_class():
    case = build_acquisition_case("2nvzlh2k", _sig(verdict="WRONG_SONG"))
    assert isinstance(case, AcquisitionCase)
    assert case.set_id == "2nvzlh2k"
    assert case.status is CaseStatus.OPEN
    assert ProblemClass.WRONG_SONG in case.problem_classes
    # a single provenance attempt is recorded (no search executed)
    assert len(case.attempts) == 1
    assert case.attempts[0].checks["identity_gate_verdict"] == "WRONG_SONG"


def test_fallback_to_original_maps_to_wrong_version():
    case = build_acquisition_case("2nvzlh2k", _sig(verdict="FALLBACK_TO_ORIGINAL"))
    assert case is not None
    assert ProblemClass.WRONG_VERSION in case.problem_classes


def test_no_reference_maps_to_missing_asset():
    case = build_acquisition_case("2nvzlh2k", _sig(verdict="NO_REFERENCE"))
    assert case is not None
    assert ProblemClass.MISSING_ASSET in case.problem_classes


def test_acappella_claim_surfaces_wrong_stem():
    case = build_acquisition_case(
        "2nvzlh2k",
        _sig(slot_label="027w1", claimed_stem="acappella", verdict="NO_REFERENCE"),
    )
    assert case is not None
    assert ProblemClass.WRONG_STEM in case.problem_classes
    # acappella w-slot derives the payload layer role
    assert case.layer_role == "payload"


def test_extended_variant_surfaces_wrong_variant():
    case = build_acquisition_case(
        "2nvzlh2k", _sig(claimed_variant="extended", verdict="NO_REFERENCE")
    )
    assert case is not None
    assert ProblemClass.WRONG_VARIANT in case.problem_classes


def test_build_batch_skips_clean_slots():
    sigs = [
        _sig(slot_label="001", verdict="OK"),
        _sig(slot_label="002", recording_id="tlp2", verdict="WRONG_SONG"),
        _sig(slot_label="003", recording_id="tlp3", verdict="SKIPPED"),
        _sig(slot_label="004", recording_id="tlp4", verdict="NO_REFERENCE"),
    ]
    cases = build_acquisition_cases("2nvzlh2k", sigs)
    assert [c.slot_label for c in cases] == ["002", "004"]


# ── Seam 2: banding ───────────────────────────────────────────────────────────


def _span() -> MixSpan:
    return MixSpan(
        set_id="2nvzlh2k", slot_label="027", set_start_s=800.0, span_dur_s=40.0
    )


def _cand(**kw) -> RefCandidate:
    base = dict(recording_id="tlp1", source_url="yt://x")
    base.update(kw)
    return RefCandidate(**base)


def test_two_agreeing_channels_accept():
    results = [
        AlignmentResult(recording_id="r", offset_s=12.0, confidence=0.8, source="fp"),
        AlignmentResult(
            recording_id="r", offset_s=12.3, confidence=0.7, source="hubert"
        ),
    ]
    report = band_agreement(results)
    assert report.band is Band.ACCEPT
    assert report.n_agreeing == 2


def test_single_channel_cannot_accept_only_review():
    # one confident channel: confirmation-drift guard forbids ACCEPT.
    results = [
        AlignmentResult(recording_id="r", offset_s=12.0, confidence=0.95, source="fp"),
    ]
    report = band_agreement(results)
    assert report.band is Band.REVIEW


def test_disagreeing_channels_abstain():
    results = [
        AlignmentResult(recording_id="r", offset_s=12.0, confidence=0.6, source="fp"),
        AlignmentResult(
            recording_id="r", offset_s=99.0, confidence=0.6, source="hubert"
        ),
    ]
    report = band_agreement(results)
    assert report.band is Band.ABSTAIN


def test_all_abstained_probes_abstain():
    results = [
        AlignmentResult.abstained(source="fp"),
        AlignmentResult.abstained(source="hubert"),
    ]
    report = band_agreement(results)
    assert report.band is Band.ABSTAIN
    assert report.detail == "no probe fired"


def test_same_source_twice_counts_as_one_channel():
    # two fp results must NOT count as two agreeing channels → no ACCEPT.
    results = [
        AlignmentResult(recording_id="r", offset_s=12.0, confidence=0.9, source="fp"),
        AlignmentResult(recording_id="r", offset_s=12.1, confidence=0.9, source="fp"),
    ]
    report = band_agreement(results)
    assert report.band is Band.REVIEW  # one strong channel, not two


def test_low_confidence_agreement_does_not_accept():
    results = [
        AlignmentResult(recording_id="r", offset_s=12.0, confidence=0.3, source="fp"),
        AlignmentResult(
            recording_id="r", offset_s=12.2, confidence=0.3, source="hubert"
        ),
    ]
    report = band_agreement(results)
    assert report.band is not Band.ACCEPT


# ── Seam 2: TrainingSignal + proposed correction ──────────────────────────────


def _accept_scorer(candidate: RefCandidate, span: MixSpan):
    return [
        AlignmentResult(
            recording_id=candidate.recording_id,
            offset_s=10.0,
            confidence=0.8,
            source="fp",
        ),
        AlignmentResult(
            recording_id=candidate.recording_id,
            offset_s=10.2,
            confidence=0.7,
            source="hubert",
        ),
    ]


def _abstain_scorer(candidate: RefCandidate, span: MixSpan):
    return [
        AlignmentResult(
            recording_id=candidate.recording_id,
            offset_s=10.0,
            confidence=0.6,
            source="fp",
        ),
        AlignmentResult(
            recording_id=candidate.recording_id,
            offset_s=90.0,
            confidence=0.6,
            source="hubert",
        ),
    ]


def test_accept_builds_training_signal():
    prop = propose_from_candidate(_cand(), _span(), _accept_scorer)
    assert prop.agreement.band is Band.ACCEPT
    assert isinstance(prop.training_signal, TrainingSignal)
    assert len(prop.training_signal.preference_pairs) == 1
    winner, span_key = prop.training_signal.preference_pairs[0]
    assert winner == "yt://x"
    assert span_key.startswith("2nvzlh2k:027@")
    assert prop.resolution is not None
    assert prop.resolution.ref_source == "online_candidate"
    assert prop.resolution.gt_confirmed is False


def test_abstain_yields_no_training_signal():
    prop = propose_from_candidate(_cand(), _span(), _abstain_scorer)
    assert prop.agreement.band is Band.ABSTAIN
    assert prop.training_signal is None
    assert prop.correction is None
    assert prop.resolution is None


def test_correction_proposed_when_candidate_differs_from_claim():
    cand = _cand(version="remix")
    claim_axes = {"version": "original", "stem": "regular", "variant": "regular"}
    prop = propose_from_candidate(cand, _span(), _accept_scorer, claim_axes=claim_axes)
    assert prop.agreement.band is Band.ACCEPT
    assert isinstance(prop.correction, ProposedCorrection)
    assert prop.correction.axis == "version"
    assert prop.correction.old_value == "original"
    assert prop.correction.new_value == "remix"
    assert prop.correction.dry_run is True


def test_no_correction_when_candidate_matches_claim():
    cand = _cand(version="original", stem="regular", variant="regular")
    claim_axes = {"version": "original", "stem": "regular", "variant": "regular"}
    prop = propose_from_candidate(cand, _span(), _accept_scorer, claim_axes=claim_axes)
    assert prop.agreement.band is Band.ACCEPT
    assert prop.correction is None  # a missing-ref fill, not a correction


# ── dry-run evaluation on GT ──────────────────────────────────────────────────


def test_evaluate_banding_precision_recall():
    span = _span()
    cases = [
        GtCandidateCase(_cand(recording_id="good"), span, candidate_is_correct=True),
        GtCandidateCase(_cand(recording_id="bad"), span, candidate_is_correct=False),
    ]

    def scorer(candidate: RefCandidate, s: MixSpan):
        if candidate.recording_id == "good":
            return _accept_scorer(candidate, s)
        return _abstain_scorer(candidate, s)

    score, proposals = evaluate_banding(cases, scorer)
    # ACCEPT the good one, ABSTAIN the bad one → perfect precision & recall.
    assert score.accept_correct == 1
    assert score.accept_wrong == 0
    assert score.abstain_correct == 1
    assert score.precision == 1.0
    assert score.recall == 1.0
    assert len(proposals) == 2


def test_evaluate_flags_accept_wrong():
    # a wrong candidate that the probes (spuriously) agree on → accept_wrong.
    span = _span()
    cases = [
        GtCandidateCase(_cand(recording_id="bad"), span, candidate_is_correct=False)
    ]
    score, _ = evaluate_banding(cases, _accept_scorer)
    assert score.accept_wrong == 1
    assert score.precision == 0.0


# ── zero canonical mutation ───────────────────────────────────────────────────


def test_dry_run_never_writes_canonical(tmp_path, monkeypatch):
    """The seam produces proposals only — assert no JSONL / DB write occurs.

    We spy on the acquisition-case ``save_cases`` sink: producing cases and
    proposals must never call it.
    """
    import core.acquisition_case as ac

    called = {"n": 0}

    def _boom(*a, **k):
        called["n"] += 1
        raise AssertionError("cotrain_seam must not write canonical state")

    monkeypatch.setattr(ac, "save_cases", _boom)

    sigs = [SlotSignal(slot_label="027", recording_id="tlp1", verdict="WRONG_SONG")]
    cases = build_acquisition_cases("2nvzlh2k", sigs)
    assert len(cases) == 1

    prop = propose_from_candidate(_cand(), _span(), _accept_scorer)
    assert isinstance(prop, CandidateProposal)

    assert called["n"] == 0  # nothing was written


def test_proposed_correction_is_flagged_dry_run():
    corr = ProposedCorrection(
        set_id="s",
        slot_label="027",
        recording_id="r",
        axis="version",
        old_value="original",
        new_value="remix",
        source_url="yt://x",
        reason="test",
    )
    assert corr.dry_run is True


# ── real-probe scorer is a safe no-op when audio absent ───────────────────────


def test_real_probe_scorer_abstains_without_audio(tmp_path):
    scorer = real_probe_scorer(
        mix_audio_path=tmp_path / "missing_mix.wav",
        ref_audio_root=tmp_path / "refs",
    )
    results = scorer(_cand(), _span())
    assert all(r.abstain for r in results)
    # and the banding of an all-abstain result is ABSTAIN (no signal)
    assert band_agreement(results).band is Band.ABSTAIN
