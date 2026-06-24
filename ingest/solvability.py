"""Metadata-only solvability tier for stem acquisition routing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class SolvabilityTier(IntEnum):
    UNKNOWN = 0
    OFFICIAL_LIKELY = 1
    COMMUNITY = 2
    SEPARATION_ONLY = 3


@dataclass(frozen=True)
class SolvabilityResult:
    tier: SolvabilityTier
    detail: str = ""


def classify_metadata(
    *,
    full_name: str | None,
    version: str | None,
    claimed_stem: str = "regular",
) -> SolvabilityResult:
    """Heuristic tier without external API (Discogs/MB stub for future)."""
    name = (full_name or "").lower()
    if claimed_stem in ("acappella", "instrumental"):
        if "acap" in name or "instrumental" in name:
            return SolvabilityResult(SolvabilityTier.COMMUNITY, "explicit stem in name")
        return SolvabilityResult(
            SolvabilityTier.COMMUNITY, "stem slot — search community"
        )
    if version in ("mashup", "bootleg"):
        return SolvabilityResult(
            SolvabilityTier.COMMUNITY, "fan edit — community/sc link"
        )
    if version in ("remix", "rework"):
        return SolvabilityResult(
            SolvabilityTier.OFFICIAL_LIKELY, "named remix — YT Music/Topic"
        )
    return SolvabilityResult(
        SolvabilityTier.OFFICIAL_LIKELY, "original — Topic/default"
    )
