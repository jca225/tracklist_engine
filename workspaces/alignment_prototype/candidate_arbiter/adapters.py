"""Read-only adapters from serialized probe proposals to candidate contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from workspaces.alignment_prototype.agentic.belief import (
    CLUSTER_TOL_S,
    INDEPENDENCE_GROUP,
)

from .io import baseline_candidates
from .schema import CandidateEvidence, PlacementCandidate


def _proposal_value(value: object) -> tuple[float, float | None, float | None] | None:
    """Normalize legacy float or structured serialized proposal values."""
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value), None, None
    if not isinstance(value, Mapping):
        raise TypeError(f"unsupported probe proposal {type(value).__name__}")
    set_start = value.get("set_start_s")
    if set_start is None:
        return None
    ref_start = value.get("ref_start_s")
    confidence = value.get("confidence")
    return (
        float(set_start),
        float(ref_start) if ref_start is not None else None,
        float(confidence) if confidence is not None else None,
    )


def _groups(sources: tuple[str, ...]) -> int:
    return len({INDEPENDENCE_GROUP.get(source, source) for source in sources})


def candidates_from_timeline(timeline: dict) -> tuple[PlacementCandidate, ...]:
    """Extract baseline + serialized probe candidates without changing decisions."""
    set_id = str(timeline.get("set_id") or "")
    baselines = {
        candidate.slot_label: candidate for candidate in baseline_candidates(timeline)
    }
    out: list[PlacementCandidate] = []

    for span in timeline["spans"]:
        slot = str(span["slot_label"])
        baseline = baselines[slot]
        out.append(baseline)
        raw_proposals: dict[str, Any] = dict(span.get("probe_proposals") or {})
        proposals = {
            source: normalized
            for source, value in raw_proposals.items()
            if (normalized := _proposal_value(value)) is not None
        }
        positions = {
            "baseline": baseline.set_start_s,
            **{source: proposal[0] for source, proposal in proposals.items()},
        }
        cue = proposals.get("cue_prior")
        mert = proposals.get("mert_decode")

        for source in sorted(proposals):
            set_start, ref_start, confidence = proposals[source]
            agreeing = tuple(
                sorted(
                    other
                    for other, position in positions.items()
                    if abs(position - set_start) <= CLUSTER_TOL_S
                )
            )
            out.append(
                PlacementCandidate(
                    set_id=set_id,
                    slot_label=slot,
                    recording_id=str(span["recording_id"]),
                    claimed_stem=str(span.get("claimed_stem") or "regular"),
                    source=source,
                    rank=0,
                    set_start_s=set_start,
                    ref_start_s=ref_start,
                    native_confidence=confidence,
                    evidence=CandidateEvidence(
                        baseline_delta_s=set_start - baseline.set_start_s,
                        agreeing_sources=agreeing,
                        independent_groups=_groups(agreeing),
                        cue_delta_s=(set_start - cue[0]) if cue else None,
                        mert_delta_s=(set_start - mert[0]) if mert else None,
                    ),
                )
            )
    return tuple(out)
