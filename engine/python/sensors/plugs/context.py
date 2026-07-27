"""Shared Propose context (image ids, OD queries, free-form metadata)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID


def _empty_uuids() -> list[UUID]:
    return []


def _empty_strs() -> list[str]:
    return []


def _empty_meta() -> dict[str, Any]:
    return {}


@dataclass
class EvidenceContext:
    """Inputs available to a gatherer for one Propose pass."""

    image_artifact_ids: list[UUID] = field(default_factory=_empty_uuids)
    od_queries: list[str] = field(default_factory=_empty_strs)
    metadata: dict[str, Any] = field(default_factory=_empty_meta)
