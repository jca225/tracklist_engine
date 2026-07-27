"""Provenance adapter for placement/structure probe output (Phase 2).

The existing harness probes emit one :class:`AlignmentResult` containing both
where a reference was placed and which piecewise ref trajectory was decoded.
This adapter keeps those axes independent in the provenance spine:

* PLACEMENT candidate = the accepted mix/ref start coordinate;
* STRUCTURE candidate = the accepted piecewise ref-segment sequence;
* an abstaining probe records evidence plus two empty-posterior beliefs;
* probabilities are injected explicitly by a calibrator.  Raw probe margins or
  Viterbi scores are never silently treated as posteriors.

The adapter is intentionally off the shipped inference path.  It proves the
Phase-2 contract over existing probe output without changing decode math.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from core.provenance import (
    Axis,
    AxisBelief,
    DecisionRule,
    EvidenceDirection,
    ProvenanceRepository,
    Run,
    SubjectRef,
)
from workspaces.alignment_prototype.harness.contract import AlignmentResult


@dataclass(frozen=True)
class CalibratedAxisProbability:
    """One axis posterior supplied by a named, versioned calibrator."""

    axis: Axis
    probability: float
    calibration_id: str

    def __post_init__(self) -> None:
        if self.axis not in (Axis.PLACEMENT, Axis.STRUCTURE):
            raise ValueError(f"unsupported probe axis: {self.axis}")
        if not (0.0 <= self.probability <= 1.0):
            raise ValueError(f"probability must be in [0,1], got {self.probability}")
        if not self.calibration_id:
            raise ValueError("calibration_id is required")


@dataclass(frozen=True)
class PlacementStructureBeliefs:
    placement: AxisBelief
    structure: AxisBelief


def _placement_candidate(set_start_s: float) -> str:
    return json.dumps(
        {
            "schema": "placement-v1",
            "set_start_s": set_start_s,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _structure_candidate(
    result: AlignmentResult, *, mix_time_origin_s: float
) -> str:
    if not result.segments:
        raise ValueError("structure belief requires an explicit segment sequence")
    return json.dumps(
        {
            "schema": "structure-v1",
            "mix_time_frame": "set_absolute_s",
            "recording_id": result.recording_id,
            "segments": [
                {
                    "mix_start_s": mix_time_origin_s + segment.mix_start_s,
                    "ref_start_s": segment.ref_start_s,
                    "ref_end_s": segment.ref_end_s,
                }
                for segment in result.segments
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def record_placement_structure_beliefs(
    *,
    repo: ProvenanceRepository,
    run: Run,
    subject: SubjectRef,
    result: AlignmentResult,
    placement_rule: DecisionRule,
    structure_rule: DecisionRule,
    set_start_s: float | None = None,
    segment_mix_time_origin_s: float | None = None,
    placement_probability: CalibratedAxisProbability | None = None,
    structure_probability: CalibratedAxisProbability | None = None,
) -> PlacementStructureBeliefs:
    """Persist separate placement and structure belief chains for one probe.

    A committed placement requires an explicit mix-time ``set_start_s`` plus a
    calibrated placement probability.  ``AlignmentResult.offset_s`` is
    reference-time for path decode and MUST NOT be reused as mix-time placement.
    Structure commits only when an explicit segment sequence, its mix-time
    origin, and a calibrated structure probability are present. Missing axes
    persist as unresolved.
    """
    if placement_rule.axis is not Axis.PLACEMENT:
        raise ValueError("placement_rule must target Axis.PLACEMENT")
    if structure_rule.axis is not Axis.STRUCTURE:
        raise ValueError("structure_rule must target Axis.STRUCTURE")

    calibrated = {
        Axis.PLACEMENT: placement_probability,
        Axis.STRUCTURE: structure_probability,
    }
    if result.abstain:
        if set_start_s is not None or segment_mix_time_origin_s is not None or any(
            value is not None for value in calibrated.values()
        ):
            raise ValueError("an abstaining result cannot carry axis probabilities")
    else:
        if set_start_s is None or placement_probability is None:
            raise ValueError(
                "committed results require explicit set_start_s and calibrated "
                "placement probability"
            )
        if placement_probability.axis is not Axis.PLACEMENT:
            raise ValueError("placement_probability must target Axis.PLACEMENT")
        if (
            structure_probability is not None
            and structure_probability.axis is not Axis.STRUCTURE
        ):
            raise ValueError("structure_probability must target Axis.STRUCTURE")
        if structure_probability is not None and not result.segments:
            raise ValueError(
                "structure_probability requires an explicit segment sequence"
            )
        if structure_probability is not None and segment_mix_time_origin_s is None:
            raise ValueError(
                "structure_probability requires segment_mix_time_origin_s"
            )

    candidate_ids = {
        Axis.PLACEMENT: (
            None
            if result.abstain or set_start_s is None
            else _placement_candidate(set_start_s)
        ),
        Axis.STRUCTURE: (
            None
            if result.abstain
            or structure_probability is None
            or segment_mix_time_origin_s is None
            or not result.segments
            else _structure_candidate(
                result, mix_time_origin_s=segment_mix_time_origin_s
            )
        ),
    }
    rules = {
        Axis.PLACEMENT: placement_rule,
        Axis.STRUCTURE: structure_rule,
    }
    beliefs: dict[Axis, AxisBelief] = {}

    for axis in (Axis.PLACEMENT, Axis.STRUCTURE):
        candidate_id = candidate_ids[axis]
        probability = calibrated[axis]
        claim = repo.record_claim(
            axis=axis,
            subject=subject,
            predicate=(
                "has_ref_start"
                if axis is Axis.PLACEMENT
                else "has_ref_segment_sequence"
            ),
            candidate=(
                SubjectRef(f"{axis.value}-hypothesis", candidate_id)
                if candidate_id is not None
                else None
            ),
            run=run,
            context={"probe_source": result.source},
        )
        evidence = repo.record_evidence(
            claim=claim,
            source_family=result.source or "alignment_probe",
            direction=(
                EvidenceDirection.ABSTAINS
                if result.abstain
                else EvidenceDirection.SUPPORTS
            ),
            run=run,
            native_score={
                "probe_confidence": result.confidence,
                "axis_probability": (
                    probability.probability if probability is not None else None
                ),
            },
            uncertainty={
                "calibration_id": (
                    probability.calibration_id if probability is not None else None
                )
            },
        )
        posterior = (
            {}
            if candidate_id is None or probability is None
            else {candidate_id: probability.probability}
        )
        beliefs[axis] = repo.record_belief(
            subject=subject,
            axis=axis,
            posterior=posterior,
            rule=rules[axis],
            run=run,
            contributing_evidence_ids=[evidence.evidence_id],
        )

    return PlacementStructureBeliefs(
        placement=beliefs[Axis.PLACEMENT],
        structure=beliefs[Axis.STRUCTURE],
    )
