from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Sequence
from alignment.harness.contract import AlignmentResult


class AbstainReason(Enum):
    NONE = "none"
    NO_DATA = "no_data"
    LOW_MARGIN = "low_margin"
    OUT_OF_DOMAIN = "out_of_domain"


@dataclass(frozen=True)
class Vote:
    probe: str
    span_id: str
    recording_id: str | None
    offset_s: float
    confidence: float
    abstained: bool
    reason: AbstainReason
    features: tuple[float, ...]


def collect_votes(
    span_id: str,
    results: Sequence[tuple[AlignmentResult, tuple[float, ...], AbstainReason]],
) -> tuple[Vote, ...]:
    return tuple(
        Vote(
            probe=res.source,
            span_id=span_id,
            recording_id=None if res.abstain else res.recording_id,
            offset_s=res.offset_s,
            confidence=res.confidence,
            abstained=res.abstain,
            reason=reason,
            features=features,
        )
        for res, features, reason in results
    )
