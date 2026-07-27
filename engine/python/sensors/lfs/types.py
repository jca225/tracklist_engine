"""Shared LF emission / unit shapes for the Rust JSON boundary."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class NativeScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family: str
    value: float
    scale: str
    support: int | None = None


class Emission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    emission_id: str
    unit_id: str
    source_spec_id: str
    source_name: str
    source_family: str
    producer_run_id: str
    value: str | None = None
    abstained: bool
    native_score: NativeScore
    evidence_ids: list[str] = Field(default_factory=list)


class InferenceUnitOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unit_id: str
    set_id: str
    axis: str
    subject_type: str
    subject_id: str
    generation_parameters_hash: str
    split: str
