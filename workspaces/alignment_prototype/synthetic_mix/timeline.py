"""Timeline datatypes for BB12-realistic synthetic windows."""

from __future__ import annotations

from dataclasses import dataclass

from .catalog import BedEntry, PayloadEntry, RegularEntry


@dataclass(frozen=True)
class MixSlice:
    """One contiguous slice of source audio placed on the mix timeline."""

    mix_start_s: float
    mix_end_s: float
    ref_start_s: float
    ref_end_s: float


@dataclass(frozen=True)
class InstrumentalBlock:
    bed: BedEntry
    mix_start_s: float
    mix_end_s: float
    slices: tuple[MixSlice, ...]
    slot_label: str
    pitch_shift_semi: int = 0


@dataclass(frozen=True)
class AcappellaSpan:
    payload: PayloadEntry
    mix_start_s: float
    mix_end_s: float
    host_bpm: float
    ref_start_s: float
    ref_end_s: float
    slices: tuple[MixSlice, ...]
    is_loop: bool
    slot_label: str
    parent_slot: str
    tempo_ratio: float
    pitch_shift_semi: int
    gain_curve: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class RegularSpan:
    """A full-song play (instrumental + vocals summed)."""

    regular: RegularEntry
    mix_start_s: float
    mix_end_s: float
    host_bpm: float
    ref_start_s: float
    ref_end_s: float
    slot_label: str
    parent_slot: str
    tempo_ratio: float
    pitch_shift_semi: int
    gain_curve: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class MashupWindowV2:
    mix_id: str
    window_duration_s: float
    instrumentals: tuple[InstrumentalBlock, ...]
    acappellas: tuple[AcappellaSpan, ...]
    curriculum: str
    regulars: tuple[RegularSpan, ...] = ()
