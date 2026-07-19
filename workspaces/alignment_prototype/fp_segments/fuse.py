"""Path-level corroboration across independently decoded audio channels."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

from .schema import ConstituentSegment


@dataclass(frozen=True)
class CorroboratedSegment:
    segment: ConstituentSegment
    agreeing_channels: tuple[str, ...]
    contradicting_channels: tuple[str, ...]
    missing_channels: tuple[str, ...]
    corroborated_confidence: float


def _time_overlap(a: ConstituentSegment, b: ConstituentSegment) -> float:
    return max(0.0, min(a.mix_end_s, b.mix_end_s) - max(a.mix_start_s, b.mix_start_s))


def paths_compatible(
    primary: ConstituentSegment,
    other: ConstituentSegment,
    *,
    min_overlap_s: float = 2.0,
    ref_tolerance_s: float = 2.0,
) -> bool:
    """Whether two paths explain the same mix interval and reference position."""
    overlap = _time_overlap(primary, other)
    if overlap < min_overlap_s:
        return False
    t = max(primary.mix_start_s, other.mix_start_s) + overlap / 2
    p_ref = primary.ref_start_s + (t - primary.mix_start_s) * primary.slope
    o_ref = other.ref_start_s + (t - other.mix_start_s) * other.slope
    return abs(p_ref - o_ref) <= ref_tolerance_s


def corroborate_segments(
    primary: Sequence[ConstituentSegment],
    channels: Mapping[str, Sequence[ConstituentSegment] | None],
    *,
    agreement_boost: float = 0.2,
    contradiction_penalty: float = 0.2,
) -> tuple[CorroboratedSegment, ...]:
    """Score channel agreement without moving the primary path boundaries.

    ``None`` means the channel was unavailable and is never negative evidence.
    An empty sequence means the channel was available but decoded no path.
    """
    out = []
    for segment in primary:
        agreeing: list[str] = []
        contradicting: list[str] = []
        missing: list[str] = []
        for name, paths in channels.items():
            if paths is None:
                missing.append(name)
            elif any(paths_compatible(segment, other) for other in paths):
                agreeing.append(name)
            else:
                contradicting.append(name)
        confidence = segment.confidence
        confidence += agreement_boost * len(agreeing)
        confidence -= contradiction_penalty * len(contradicting)
        out.append(
            CorroboratedSegment(
                segment=segment,
                agreeing_channels=tuple(sorted(agreeing)),
                contradicting_channels=tuple(sorted(contradicting)),
                missing_channels=tuple(sorted(missing)),
                corroborated_confidence=min(1.0, max(0.0, confidence)),
            )
        )
    return tuple(out)
