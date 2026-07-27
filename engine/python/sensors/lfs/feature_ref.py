"""LF that votes using a registered feature ArtifactId (not a filesystem path).

Demonstrates storage rule #3: emissions carry ``feature_artifact_id`` in
``evidence_ids`` / metadata. The symbolic vote here is still claimed_stem when
present; the feature id is attached so the cortex can join embeddings later.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sensors.schemas import validate_payload

SOURCE_NAME = "feature_ref_lf"
SOURCE_FAMILY = "signal"


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


def emit_with_feature_ref(
    *,
    unit_id: str,
    feature_artifact_id: str,
    vote: str | None,
    producer_run_id: str | None = None,
) -> dict[str, Any]:
    """Build one EvidenceEmission that cites a feature ArtifactId."""
    run_id = producer_run_id or str(uuid.uuid4())
    abstained = vote is None
    em = Emission(
        emission_id=str(uuid.uuid4()),
        unit_id=unit_id,
        source_spec_id="00000000-0000-4000-8000-000000000002",
        source_name=SOURCE_NAME,
        source_family=SOURCE_FAMILY,
        producer_run_id=run_id,
        value=None if abstained else vote,
        abstained=abstained,
        native_score=NativeScore(
            family=SOURCE_FAMILY,
            value=0.0 if abstained else 1.0,
            scale="abstain" if abstained else "count",
            support=None if abstained else 1,
        ),
        # ArtifactId is a UUID string in our store — treat as evidence ref.
        evidence_ids=[feature_artifact_id],
    )
    payload = em.model_dump()
    validate_payload("evidence_emission", payload)
    payload["feature_artifact_id"] = feature_artifact_id
    return payload


def write_feature_ref_demo(
    feature_register_json: Path,
    out_path: Path,
    *,
    unit_id: str | None = None,
) -> Path:
    """Read feature-register report and emit one LF row referencing the feature id."""
    report = json.loads(feature_register_json.read_text(encoding="utf-8"))
    fid = report["feature_artifact_id"]
    uid = unit_id or str(uuid.uuid4())
    emission = emit_with_feature_ref(
        unit_id=uid,
        feature_artifact_id=fid,
        vote="acappella",  # demo vote; real LF would use the embedding
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"emissions": [emission]}, indent=2), encoding="utf-8")
    return out_path
