"""Typed contracts for sparse fingerprint correspondences and decoded runs."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LandmarkMatch:
    recording_id: str
    ref_stem: str
    mix_channel: str
    mix_time_s: float
    ref_time_s: float
    weight: float = 1.0
    hash_frequency: int = 1

    def __post_init__(self) -> None:
        if not self.recording_id or not self.ref_stem or not self.mix_channel:
            raise ValueError("recording_id, ref_stem and mix_channel are required")
        for name in ("mix_time_s", "ref_time_s", "weight"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.mix_time_s < 0 or self.ref_time_s < 0:
            raise ValueError("landmark times must be non-negative")
        if self.weight <= 0:
            raise ValueError("weight must be positive")
        if self.hash_frequency < 1:
            raise ValueError("hash_frequency must be positive")


@dataclass(frozen=True)
class ConstituentSegment:
    recording_id: str
    slot_label: str | None
    ref_stem: str
    mix_channel: str
    mix_start_s: float
    mix_end_s: float
    ref_start_s: float
    ref_end_s: float
    slope: float
    evidence: float
    confidence: float

    def __post_init__(self) -> None:
        if not self.recording_id or not self.ref_stem or not self.mix_channel:
            raise ValueError("recording_id, ref_stem and mix_channel are required")
        for name in (
            "mix_start_s",
            "mix_end_s",
            "ref_start_s",
            "ref_end_s",
            "slope",
            "evidence",
            "confidence",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.mix_start_s < 0 or self.ref_start_s < 0:
            raise ValueError("segment times must be non-negative")
        if self.mix_end_s <= self.mix_start_s or self.ref_end_s <= self.ref_start_s:
            raise ValueError("segment end must be after start")
        if self.slope <= 0:
            raise ValueError("slope must be positive")
        if self.evidence < 0:
            raise ValueError("evidence must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
