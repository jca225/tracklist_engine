"""Canonical 1001tracklists data model — one home for every struct.

Two tiers:
  * PERSISTED table rows (SetTrackSlotRow, TrackSuggestionRow): field order is
    the SQL INSERT column order, so materialize writes by name, never by a
    hand-maintained positional tuple.
  * IN-MEMORY parse objects (TrackRow, SuggestionRow, NoticeRow, IDTrack):
    re-exported from their parsers so callers import "the model" from one place.

Behavior-preserving: these mirror the CURRENT schema. Widening (new columns,
set_notices, ID-lens) is a separate plan — do not add fields here for it.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

# --- re-exported in-memory parse structs (single import surface) ---
from tokenizer.track_tokenizer import TrackRow
from tokenizer.suggestion_tokenizer import SuggestionRow
from tokenizer.text_tokenizer import TextRowToken as NoticeRow
from tokenizer.id_tokenizer import IDTrack

__all__ = [
    "SetTrackSlotRow",
    "TrackSuggestionRow",
    "columns",
    "as_row",
    "TrackRow",
    "SuggestionRow",
    "NoticeRow",
    "IDTrack",
]


@dataclass(frozen=True)
class SetTrackSlotRow:
    """One row of set_track_slots. Field order == INSERT column order."""

    set_id: str
    row_index: int
    tlp_id: int | None
    recording_id: str | None
    track_id: str
    source: str
    slot_label: str | None
    is_concurrent: int
    cue_seconds: int | None
    cue_time_seconds: int | None
    claimed_version: str | None
    claimed_stem: str
    claimed_variant: str
    full_name: str | None
    title: str | None
    artists_json: str | None
    duration_seconds: int | None
    layer_role: str | None
    constituents_json: str | None


@dataclass(frozen=True)
class TrackSuggestionRow:
    """One row of track_suggestions. Field order == INSERT column order."""

    sug_id: int | None
    set_id: str
    tlp_id: int | None
    pos: int | None
    track_slug: str | None
    track_display: str | None
    artist_title: str | None
    suggester_user_id: int | None
    suggester_name: str | None
    suggestion_timestamp: str | None
    is_remix: int | None
    has_youtube: int | None
    has_soundcloud: int | None
    has_spotify: int | None


def columns(row_cls) -> tuple[str, ...]:
    """The SQL column names for a row dataclass, in INSERT order."""
    return tuple(f.name for f in fields(row_cls))


def as_row(instance) -> tuple:
    """Positional value tuple in field order, for executemany()."""
    return tuple(getattr(instance, f.name) for f in fields(instance))
