"""Tests for experimental Propose plugs (OD + agentic)."""

from dataclasses import dataclass
from uuid import uuid4

from sensors.plugs import (
    ABLETON_QUERIES,
    AgenticSearchGatherer,
    BoundingBox,
    Detection,
    DetectionSet,
    EvidenceContext,
    OpenVocabOdGatherer,
    unresolved_search_tasks,
)


@dataclass(frozen=True)
class _Unit:
    unit_id: str


def test_od_abstains_without_image() -> None:
    unit = _Unit(unit_id=str(uuid4()))
    ems = OpenVocabOdGatherer().gather([unit], EvidenceContext())
    assert len(ems) == 1
    assert ems[0].abstained
    assert ems[0].source_family == "vision"


def test_od_stub_emits_boxes_citing_image_artifact() -> None:
    unit = _Unit(unit_id=str(uuid4()))
    image_id = uuid4()
    g = OpenVocabOdGatherer(default_queries=("audio clip",))
    ctx = EvidenceContext(image_artifact_ids=[image_id], od_queries=["audio clip"])
    ems = g.gather([unit], ctx)
    assert len(ems) == 1
    assert not ems[0].abstained
    assert ems[0].evidence_ids == [str(image_id)]
    assert ems[0].value == "audio clip"
    assert ems[0].native_score.scale == "od_confidence"


def test_od_custom_detect_respects_threshold() -> None:
    unit = _Unit(unit_id=str(uuid4()))
    image_id = uuid4()

    def detect(img: object, queries: list[str], _ctx: EvidenceContext) -> DetectionSet:
        from uuid import UUID

        return DetectionSet(
            image_artifact_id=img if isinstance(img, UUID) else UUID(str(img)),
            queries=queries,
            detections=[
                Detection(
                    query="cue marker",
                    label="cue marker",
                    score=0.1,
                    box=BoundingBox(x0=0.0, y0=0.0, x1=0.1, y1=0.1),
                )
            ],
        )

    g = OpenVocabOdGatherer(detect=detect, min_score=0.5, default_queries=("cue marker",))
    ems = g.gather(
        [unit],
        EvidenceContext(metadata={"image_artifact_id": image_id, "od_queries": ["cue marker"]}),
    )
    assert ems[0].abstained


def test_ableton_query_pack_nonempty() -> None:
    assert "audio clip" in ABLETON_QUERIES


def test_agentic_default_abstains() -> None:
    unit = _Unit(unit_id=str(uuid4()))
    ems = AgenticSearchGatherer().gather([unit], EvidenceContext())
    assert ems[0].abstained
    assert ems[0].source_family == "agent"


def test_agentic_wired_propose_votes() -> None:
    unit = _Unit(unit_id=str(uuid4()))

    def propose(_u: object, _ctx: EvidenceContext) -> tuple[str | None, float, str]:
        return "acappella", 0.9, "ok"

    ems = AgenticSearchGatherer(propose=propose).gather([unit], EvidenceContext())
    assert not ems[0].abstained
    assert ems[0].value == "acappella"


def test_unresolved_search_tasks() -> None:
    tasks = unresolved_search_tasks(unit_ids=["a", "b"])
    assert len(tasks) == 2
    assert tasks[0]["task"] == "search_missing_recording"
