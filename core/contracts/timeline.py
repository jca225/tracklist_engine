"""PredictedTimeline — the typed record for `out/<set>_predicted_timeline*.json`.

Writer: `workspaces/alignment_prototype/infer.py` (spans), mutated in place by
`joint_ref_decode` (ref_segments). Until those write through this module, the
decoder is TOLERANT of unknown span fields (infer adds provenance fields
freely) but STRICT about the fields consumers key on: a missing/mistyped
`set_id`, `slot_label`, `recording_id`, or span timing fails at load, not
three functions later inside a scorer.

`load_timeline()` is the only sanctioned reader — raw ``json.loads`` of a
timeline is a ratcheted guardrail violation class.
"""

from __future__ import annotations

from pathlib import Path

import msgspec

from core.contracts.ids import (
    RecordingId,
    SetId,
    SlotLabel,
    normalize_slot_label,
)


class RefSegment(msgspec.Struct, frozen=True):
    """One straight piece of a span's mix→ref mapping (loops/multiseg)."""

    mix_start_s: float
    ref_start_s: float
    ref_end_s: float | None = None
    dur_s: float | None = None  # legacy pre-ref_end_s writer


class TimelineSpan(msgspec.Struct, frozen=True):
    slot_label: str
    recording_id: str
    set_start_s: float
    set_end_s: float
    ref_start_s: float | None = None
    ref_end_s: float | None = None
    claimed_stem: str = "regular"
    confidence: float | None = None
    cue_anchor_s: float | None = None
    name: str = ""
    start_source: str = ""
    ref_stretch: float | None = None
    ref_segments: tuple[RefSegment, ...] = ()
    is_loop: bool = False

    @property
    def slot(self) -> SlotLabel | None:
        return normalize_slot_label(self.slot_label)

    @property
    def recording(self) -> RecordingId:
        return RecordingId(self.recording_id)


class SkippedSlot(msgspec.Struct, frozen=True):
    slot_label: str
    name: str = ""


class PredictedTimeline(msgspec.Struct, frozen=True):
    set_id: str
    spans: tuple[TimelineSpan, ...]
    train_set_id: str | None = None
    band_s: float | None = None
    ref_decode: str | None = None
    skipped: tuple[SkippedSlot, ...] = ()

    @property
    def sid(self) -> SetId:
        return SetId(self.set_id)


_DECODER = msgspec.json.Decoder(PredictedTimeline)


def load_timeline(path: Path | str) -> PredictedTimeline:
    """Load + validate a predicted-timeline JSON.

    Raises ``msgspec.ValidationError`` with field-level detail on any missing
    or mistyped consumed field; unknown extra fields are ignored (writer is
    not yet migrated). The returned record carries its ``set_id`` — pass it
    through ``core.contracts.guard.join_guard`` before combining with any
    other set-scoped artifact.
    """
    raw = Path(path).read_bytes()
    tl = _DECODER.decode(raw)
    if not tl.set_id:
        raise msgspec.ValidationError(f"{path}: timeline has empty set_id")
    return tl
