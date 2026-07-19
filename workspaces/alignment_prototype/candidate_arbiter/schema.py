"""Immutable, versioned contracts for placement candidates and their evidence."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any

from workspaces.alignment_prototype.drivers.base import norm_slot

SCHEMA_VERSION = 1
STEMS = frozenset({"regular", "acappella", "instrumental"})


def _validate_optional_floats(value: object, *, owner: str) -> None:
    for field in fields(value):
        item = getattr(value, field.name)
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"{owner}.{field.name} must be finite, got {item!r}")


@dataclass(frozen=True)
class CandidateEvidence:
    baseline_delta_s: float
    ridge_peak: float | None = None
    ridge_second: float | None = None
    ridge_margin: float | None = None
    ridge_prominence: float | None = None
    ridge_support_s: float | None = None
    vote_count: int | None = None
    vote_density: float | None = None
    runner_up_ratio: float | None = None
    agreeing_sources: tuple[str, ...] = ()
    independent_groups: int = 0
    cue_delta_s: float | None = None
    mert_delta_s: float | None = None
    neighbor_gap_left_s: float | None = None
    neighbor_gap_right_s: float | None = None
    active_layer_count: int | None = None
    repeat_ambiguity: float | None = None

    def __post_init__(self) -> None:
        _validate_optional_floats(self, owner=type(self).__name__)
        for name in ("vote_count", "independent_groups", "active_layer_count"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if not isinstance(self.agreeing_sources, tuple):
            object.__setattr__(self, "agreeing_sources", tuple(self.agreeing_sources))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CandidateEvidence":
        payload = dict(value)
        payload["agreeing_sources"] = tuple(payload.get("agreeing_sources") or ())
        return cls(**payload)


@dataclass(frozen=True)
class PlacementCandidate:
    set_id: str
    slot_label: str
    recording_id: str
    claimed_stem: str
    source: str
    rank: int
    set_start_s: float
    ref_start_s: float | None
    native_confidence: float | None
    evidence: CandidateEvidence

    def __post_init__(self) -> None:
        if not self.set_id or not self.slot_label or not self.recording_id:
            raise ValueError("set_id, slot_label and recording_id are required")
        if not self.source:
            raise ValueError("source is required")
        if self.claimed_stem not in STEMS:
            raise ValueError(f"unknown claimed_stem {self.claimed_stem!r}")
        if self.rank < 0:
            raise ValueError("rank must be non-negative")
        _validate_optional_floats(self, owner=type(self).__name__)
        if self.native_confidence is not None and not (
            0.0 <= self.native_confidence <= 1.0
        ):
            raise ValueError("native_confidence must be in [0, 1]")

    @property
    def key(self) -> tuple[str, str, str, str, int]:
        return (
            self.set_id,
            norm_slot(self.slot_label),
            self.recording_id,
            self.source,
            self.rank,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence"] = self.evidence.to_dict()
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PlacementCandidate":
        payload = dict(value)
        payload["evidence"] = CandidateEvidence.from_dict(payload["evidence"])
        return cls(**payload)


@dataclass(frozen=True)
class CandidateBankMetadata:
    producer_sha: str
    schema_version: int = SCHEMA_VERSION
    source_timeline_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.producer_sha:
            raise ValueError("producer_sha is required")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version={self.schema_version}; "
                f"expected {SCHEMA_VERSION}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CandidateBankMetadata":
        return cls(**value)
