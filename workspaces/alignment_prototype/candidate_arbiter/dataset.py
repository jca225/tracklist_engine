"""Baseline-relative labels and examples for candidate-critic training."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from workspaces.alignment_prototype.drivers.base import norm_slot

from .oracle import CandidateError
from .schema import PlacementCandidate


@dataclass(frozen=True)
class CandidateLabel:
    candidate_error_s: float
    baseline_error_s: float
    improvement_s: float
    improves_baseline: bool
    within_accept_tolerance: bool


@dataclass(frozen=True)
class CandidateExample:
    candidate: PlacementCandidate
    label: CandidateLabel

    @property
    def group_key(self) -> tuple[str, str, str]:
        return (
            self.candidate.set_id,
            norm_slot(self.candidate.slot_label),
            self.candidate.recording_id,
        )


def build_examples_from_errors(
    rows: tuple[CandidateError, ...],
    *,
    min_improvement_s: float = 0.0,
    accept_tolerance_s: float = 5.0,
) -> tuple[CandidateExample, ...]:
    """Label non-baseline candidates against their same-span baseline error."""
    grouped: dict[tuple[str, str, str], list[CandidateError]] = defaultdict(list)
    for row in rows:
        candidate = row.candidate
        grouped[
            (
                candidate.set_id,
                norm_slot(candidate.slot_label),
                candidate.recording_id,
            )
        ].append(row)

    examples: list[CandidateExample] = []
    for options in grouped.values():
        baseline_rows = [
            option for option in options if option.candidate.source == "baseline"
        ]
        if len(baseline_rows) != 1:
            continue
        baseline_error = baseline_rows[0].error_s
        for option in options:
            if option.candidate.source == "baseline":
                continue
            improvement = baseline_error - option.error_s
            examples.append(
                CandidateExample(
                    candidate=option.candidate,
                    label=CandidateLabel(
                        candidate_error_s=option.error_s,
                        baseline_error_s=baseline_error,
                        improvement_s=improvement,
                        improves_baseline=improvement > min_improvement_s,
                        within_accept_tolerance=option.error_s <= accept_tolerance_s,
                    ),
                )
            )
    return tuple(examples)
