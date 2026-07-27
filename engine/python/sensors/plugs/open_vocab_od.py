"""Open-vocabulary object detection as a Propose gatherer.

Detector output cites image / DetectionSet FeatureBlob artifact ids. Emission
``value`` is the winning open-vocab label. Missing image or no hit → abstain.
OD does not mint ``RecordingId``. Decide/Promote remain in Rust.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, TypeAdapter

from sensors.lfs.types import Emission, NativeScore
from sensors.plugs.context import EvidenceContext

SOURCE_FAMILY = "vision"
DEFAULT_NAME = "open_vocab_od"
SOURCE_SPEC_ID = "00000000-0000-4000-8000-000000000010"

_STR_LIST = TypeAdapter(list[str])
_UNIT_QUERY_MAP = TypeAdapter(dict[str, list[str]])

ABLETON_QUERIES: tuple[str, ...] = (
    "audio clip",
    "automation lane",
    "cue marker",
    "track header",
    "acappella clip",
    "return track",
)
TRACKLIST_PAGE_QUERIES: tuple[str, ...] = (
    "tracklist row",
    "artist name",
    "track title",
    "timestamp",
    "ID badge",
)
SPECTROGRAM_QUERIES: tuple[str, ...] = (
    "vocal onset",
    "drop",
    "breakdown",
    "kick drum burst",
    "silence gap",
)


class UnitRef(Protocol):
    @property
    def unit_id(self) -> str | UUID: ...


class BoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x0: float
    y0: float
    x1: float
    y1: float
    pixel_space: bool = False

    def validate_geometry(self) -> None:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("inverted bounding box")


class Detection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    label: str
    score: float
    box: BoundingBox
    mix_start_s: float | None = None
    mix_end_s: float | None = None


class DetectionSet(BaseModel):
    """One detector pass — deposit as FeatureBlob bytes under a ProcessSpec."""

    model_config = ConfigDict(extra="forbid")

    image_artifact_id: UUID
    model_card_id: UUID | None = None
    process_spec_id: UUID | None = None
    queries: list[str]
    detections: list[Detection]
    image_width: int | None = None
    image_height: int | None = None


DetectFn = Callable[[UUID, list[str], EvidenceContext], DetectionSet]


def _as_str_list(raw: object) -> list[str] | None:
    try:
        parsed = _STR_LIST.validate_python(raw)
    except (TypeError, ValueError):
        return None
    return parsed if parsed else None


def _as_unit_query_map(raw: object) -> dict[str, list[str]]:
    if raw is None:
        return {}
    try:
        return _UNIT_QUERY_MAP.validate_python(raw)
    except (TypeError, ValueError):
        return {}


def _uid(unit: UnitRef) -> str:
    return str(unit.unit_id)


def stub_detect(
    image_artifact_id: UUID,
    queries: list[str],
    context: EvidenceContext,
) -> DetectionSet:
    """Deterministic offline detector: one mid-frame box per query (score 0.5)."""
    _ = context
    detections = [
        Detection(
            query=q,
            label=q,
            score=0.5,
            box=BoundingBox(x0=0.25, y0=0.25, x1=0.75, y1=0.75),
        )
        for q in queries
    ]
    return DetectionSet(
        image_artifact_id=image_artifact_id,
        queries=list(queries),
        detections=detections,
    )


class OpenVocabOdGatherer:
    """Open-vocab OD → emissions (family=vision) for Rust Decide."""

    def __init__(
        self,
        *,
        name: str = DEFAULT_NAME,
        source_family: str = SOURCE_FAMILY,
        source_spec_id: str | None = None,
        default_queries: tuple[str, ...] | list[str] = ABLETON_QUERIES,
        detect: DetectFn | None = None,
        min_score: float = 0.2,
    ) -> None:
        self._name = name
        self._family = source_family
        self._spec_id = source_spec_id or SOURCE_SPEC_ID
        self._default_queries = tuple(default_queries)
        self._detect = detect or stub_detect
        self._min_score = min_score

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
        image_raw: object | None = context.metadata.get("image_artifact_id")
        if image_raw is None and context.image_artifact_ids:
            image_raw = context.image_artifact_ids[0]
        if image_raw is None:
            return [self._abstain(_uid(u), producer) for u in units]

        image_id = image_raw if isinstance(image_raw, UUID) else UUID(str(image_raw))
        parsed_queries = _as_str_list(context.metadata.get("od_queries"))
        if parsed_queries is None and context.od_queries:
            parsed_queries = list(context.od_queries)
        queries: list[str] = (
            parsed_queries if parsed_queries is not None else list(self._default_queries)
        )

        threshold_raw = context.metadata.get("score_threshold", self._min_score)
        threshold = (
            float(threshold_raw) if isinstance(threshold_raw, (int, float)) else self._min_score
        )
        unit_query_map = _as_unit_query_map(context.metadata.get("unit_query_map"))
        det_feature = context.metadata.get("detection_feature_artifact_id")
        evidence_ids: list[str] = [str(image_id)]
        if det_feature is not None:
            evidence_ids.append(str(det_feature))

        needed: list[str] = list(queries)
        for qs in unit_query_map.values():
            for q in qs:
                if q not in needed:
                    needed.append(q)

        det_set = self._detect(image_id, needed, context)
        for d in det_set.detections:
            d.box.validate_geometry()

        out: list[Emission] = []
        for unit in units:
            uid = _uid(unit)
            uqs = unit_query_map.get(uid, queries)
            hits = [d for d in det_set.detections if d.query in uqs and d.score >= threshold]
            if not hits:
                out.append(self._abstain(uid, producer))
                continue
            best = max(hits, key=lambda d: d.score)
            out.append(
                Emission(
                    emission_id=str(uuid4()),
                    unit_id=uid,
                    source_spec_id=self._spec_id,
                    source_name=self._name,
                    source_family=self._family,
                    producer_run_id=producer,
                    value=best.label,
                    abstained=False,
                    native_score=NativeScore(
                        family=self._family,
                        value=best.score,
                        scale="od_confidence",
                        support=len(hits),
                    ),
                    evidence_ids=list(evidence_ids),
                )
            )
        return out

    def _abstain(self, unit_id: str, run_id: str) -> Emission:
        return Emission(
            emission_id=str(uuid4()),
            unit_id=unit_id,
            source_spec_id=self._spec_id,
            source_name=self._name,
            source_family=self._family,
            producer_run_id=run_id,
            value=None,
            abstained=True,
            native_score=NativeScore(family=self._family, value=0.0, scale="abstain", support=None),
        )


def detections_to_agent_tasks(det_set: DetectionSet) -> list[dict[str, Any]]:
    """Hand boxes to AgenticSearchGatherer or a human review queue."""
    return [
        {
            "task": "inspect_detection",
            "query": d.query,
            "label": d.label,
            "score": d.score,
            "box": d.box.model_dump(),
            "image_artifact_id": str(det_set.image_artifact_id),
        }
        for d in det_set.detections
    ]
