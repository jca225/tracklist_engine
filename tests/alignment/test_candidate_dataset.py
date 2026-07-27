from __future__ import annotations

from alignment.candidate_arbiter.dataset import (
    CandidateLabel,
    build_examples_from_errors,
)
from alignment.candidate_arbiter.oracle import CandidateError
from alignment.candidate_arbiter.schema import (
    CandidateEvidence,
    PlacementCandidate,
)


def _candidate(source: str, start: float) -> PlacementCandidate:
    return PlacementCandidate(
        set_id="set",
        slot_label="1",
        recording_id="rid",
        claimed_stem="regular",
        source=source,
        rank=0,
        set_start_s=start,
        ref_start_s=None,
        native_confidence=None,
        evidence=CandidateEvidence(baseline_delta_s=start - 10.0),
    )


def test_candidate_label_is_relative_to_baseline() -> None:
    examples = build_examples_from_errors(
        (
            CandidateError(_candidate("baseline", 10.0), error_s=8.0),
            CandidateError(_candidate("fp", 16.0), error_s=2.0),
            CandidateError(_candidate("mert_decode", 30.0), error_s=12.0),
        ),
        min_improvement_s=0.5,
    )

    assert len(examples) == 2
    by_source = {example.candidate.source: example for example in examples}
    assert by_source["fp"].label == CandidateLabel(
        candidate_error_s=2.0,
        baseline_error_s=8.0,
        improvement_s=6.0,
        improves_baseline=True,
        within_accept_tolerance=True,
    )
    assert by_source["mert_decode"].label.improves_baseline is False
    assert by_source["mert_decode"].label.improvement_s == -4.0


def test_tied_candidate_is_not_labeled_as_improvement() -> None:
    examples = build_examples_from_errors(
        (
            CandidateError(_candidate("baseline", 10.0), error_s=2.0),
            CandidateError(_candidate("fp", 11.0), error_s=2.0),
        ),
        min_improvement_s=0.0,
    )

    assert examples[0].label.improves_baseline is False


def test_span_without_baseline_is_excluded() -> None:
    examples = build_examples_from_errors(
        (CandidateError(_candidate("fp", 11.0), error_s=1.0),)
    )
    assert examples == ()
