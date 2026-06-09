"""Typed records for the span aligner prototype."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlotCandidate:
    """One library candidate the aligner may select for a layer."""
    recording_id: str
    claimed_stem: str
    ref_source: str = "reference"


@dataclass(frozen=True)
class SpanTarget:
    """Supervision target for one mix play span (matches GroundTruthTrack)."""
    slot_label: str
    recording_id: str | None
    claimed_stem: str
    set_start_s: float
    set_end_s: float
    ref_start_s: float
    ref_end_s: float | None
    tempo_ratio: float | None
    pitch_shift_semi: int
    label: str


@dataclass(frozen=True)
class SpanPrediction:
    """Model output for one predicted layer."""
    slot_label: str
    recording_id: str | None
    claimed_stem: str
    set_start_s: float
    set_end_s: float
    ref_start_s: float
    ref_end_s: float | None
    confidence: float = 0.0
