"""Immutable records passed between stages. Frozen per the repo's Rust-flavoured
style — construct new values, don't mutate."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class SourceTrack:
    """One candidate original song, with whatever stem audio we have for it.

    `vocals` and `instrumental` are Demucs stems of the source *file* (which —
    per the BB12 audit — is usually a full track even when the label says
    'acappella'), so both are typically real signal and both are tried."""

    sid: str                       # stable id (recording_id or sanitized name)
    name: str
    vocals: Optional[Path] = None
    instrumental: Optional[Path] = None
    full: Optional[Path] = None    # full mixdown of the source, if present
    bpm: Optional[float] = None    # source tempo, if known (for stretch estimate)
    recording_id: Optional[str] = None

    def channels(self) -> list[tuple[str, Path]]:
        """(channel_name, path) pairs that actually exist on disk."""
        out: list[tuple[str, Path]] = []
        if self.vocals and self.vocals.is_file():
            out.append(("vocals", self.vocals))
        if self.instrumental and self.instrumental.is_file():
            out.append(("instrumental", self.instrumental))
        if not out and self.full and self.full.is_file():
            out.append(("full", self.full))
        return out


@dataclass(frozen=True)
class MixInput:
    """The mix to search, resolved to its stems + (optional) beat grid."""

    set_id: str
    name: str
    vocals: Optional[Path] = None
    instrumental: Optional[Path] = None
    full: Optional[Path] = None
    beat_times: Optional[tuple[float, ...]] = None  # seconds; None -> estimate

    def channel_path(self, channel: str) -> Optional[Path]:
        return {"vocals": self.vocals, "instrumental": self.instrumental,
                "full": self.full}.get(channel)


@dataclass(frozen=True)
class Detection:
    """One (source plays here in the mix) hit."""

    sid: str
    name: str
    channel: str                 # vocals | instrumental | full
    mix_start_s: float
    mix_end_s: float
    ref_start_s: float
    ref_end_s: float
    pitch_shift_semi: int
    time_stretch: float          # mix seconds per source second (>1 == sped up)
    confidence: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("extra")
        d["mix_start_s"] = round(self.mix_start_s, 3)
        d["mix_end_s"] = round(self.mix_end_s, 3)
        d["ref_start_s"] = round(self.ref_start_s, 3)
        d["ref_end_s"] = round(self.ref_end_s, 3)
        d["time_stretch"] = round(self.time_stretch, 4)
        d["confidence"] = round(self.confidence, 4)
        return d
