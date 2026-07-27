"""Experimental Propose plugs harvested from archived ``python_kernel``.

These emit ``EvidenceEmission``-shaped rows for the Rust Decide/Promote/Act
spine. They do **not** implement Decide (majority / LLM judge) or Promote —
that stays in ``dj_kernel``.
"""

from sensors.plugs.agentic import AgenticSearchGatherer, unresolved_search_tasks
from sensors.plugs.context import EvidenceContext
from sensors.plugs.open_vocab_od import (
    ABLETON_QUERIES,
    SPECTROGRAM_QUERIES,
    TRACKLIST_PAGE_QUERIES,
    BoundingBox,
    Detection,
    DetectionSet,
    OpenVocabOdGatherer,
    detections_to_agent_tasks,
    stub_detect,
)

__all__ = [
    "ABLETON_QUERIES",
    "AgenticSearchGatherer",
    "BoundingBox",
    "Detection",
    "DetectionSet",
    "EvidenceContext",
    "OpenVocabOdGatherer",
    "SPECTROGRAM_QUERIES",
    "TRACKLIST_PAGE_QUERIES",
    "detections_to_agent_tasks",
    "stub_detect",
    "unresolved_search_tasks",
]
