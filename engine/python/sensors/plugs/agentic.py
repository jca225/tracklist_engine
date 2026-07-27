"""Agentic Propose plug — tool/LLM agent emits votes or abstains.

Decide (including any future LLM judge) lives in Rust. This module only
proposes ``EvidenceEmission`` rows (family=``agent``). Default is offline
abstain so CI needs no model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol
from uuid import UUID, uuid4

from sensors.lfs.types import Emission, NativeScore
from sensors.plugs.context import EvidenceContext

SOURCE_SPEC_ID = "00000000-0000-4000-8000-000000000011"

ProposeFn = Callable[[Any, EvidenceContext], tuple[Any | None, float, str]]


class UnitRef(Protocol):
    @property
    def unit_id(self) -> str | UUID: ...


class AgenticSearchGatherer:
    """LLM/tool agent that proposes identity/placement votes or abstains."""

    def __init__(
        self,
        *,
        name: str = "agentic_search",
        source_family: str = "agent",
        source_spec_id: str | None = None,
        propose: ProposeFn | None = None,
    ) -> None:
        self._name = name
        self._family = source_family
        self._spec_id = source_spec_id or SOURCE_SPEC_ID
        self._propose = propose

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_family(self) -> str:
        return self._family

    def gather(
        self,
        units: Sequence[UnitRef],
        context: EvidenceContext,
        run_id: str | None = None,
    ) -> list[Emission]:
        producer = run_id or str(uuid4())
        out: list[Emission] = []
        for unit in units:
            if self._propose is None:
                value, conf = None, 0.0
            else:
                value, conf, _note = self._propose(unit, context)
            abstained = value is None
            out.append(
                Emission(
                    emission_id=str(uuid4()),
                    unit_id=str(unit.unit_id),
                    source_spec_id=self._spec_id,
                    source_name=self._name,
                    source_family=self._family,
                    producer_run_id=producer,
                    value=None if abstained else str(value),
                    abstained=abstained,
                    native_score=NativeScore(
                        family=self._family,
                        value=conf,
                        scale="abstain" if abstained else "agent_confidence",
                        support=None if abstained else 1,
                    ),
                )
            )
        return out


def unresolved_search_tasks(
    *,
    unit_ids: Sequence[str],
    axis: str = "identity",
    decision: str = "unresolved",
) -> list[dict[str, Any]]:
    """Example Act payload: units that still need an online search."""
    return [
        {
            "unit_id": uid,
            "axis": axis,
            "decision": decision,
            "task": "search_missing_recording",
        }
        for uid in unit_ids
    ]
