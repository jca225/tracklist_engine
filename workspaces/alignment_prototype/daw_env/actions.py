"""Frozen DAW action API (v1) — ALS-first Ableton ReAct harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

SoloBus = str  # "mix" | "mix_instrumental" | "mix_vocals" | slot_label


@dataclass(frozen=True)
class PlaceSpan:
    slot: str
    set_start_s: float
    set_end_s: float
    ref_start_s: float


@dataclass(frozen=True)
class NudgeSetStart:
    slot: str
    delta_s: float


@dataclass(frozen=True)
class NudgeRefStart:
    slot: str
    delta_s: float


@dataclass(frozen=True)
class SoloLayer:
    bus: SoloBus


@dataclass(frozen=True)
class RenderListen:
    t0_s: float
    t1_s: float


@dataclass(frozen=True)
class CommitSpan:
    slot: str


@dataclass(frozen=True)
class EscalateHuman:
    slot: str


DawAction = Union[
    PlaceSpan,
    NudgeSetStart,
    NudgeRefStart,
    SoloLayer,
    RenderListen,
    CommitSpan,
    EscalateHuman,
]
