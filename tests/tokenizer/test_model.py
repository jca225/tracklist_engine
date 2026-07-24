from dataclasses import fields
from tokenizer import model

SLOT_COLS = (
    "set_id",
    "row_index",
    "tlp_id",
    "recording_id",
    "track_id",
    "source",
    "slot_label",
    "is_concurrent",
    "cue_seconds",
    "cue_time_seconds",
    "claimed_version",
    "claimed_stem",
    "claimed_variant",
    "full_name",
    "title",
    "artists_json",
    "duration_seconds",
    "layer_role",
    "constituents_json",
)
SUG_COLS = (
    "sug_id",
    "set_id",
    "tlp_id",
    "pos",
    "track_slug",
    "track_display",
    "artist_title",
    "suggester_user_id",
    "suggester_name",
    "suggestion_timestamp",
    "is_remix",
    "has_youtube",
    "has_soundcloud",
    "has_spotify",
)


def test_slot_row_field_order_matches_insert():
    assert tuple(f.name for f in fields(model.SetTrackSlotRow)) == SLOT_COLS
    assert model.columns(model.SetTrackSlotRow) == SLOT_COLS


def test_suggestion_row_field_order_matches_insert():
    assert tuple(f.name for f in fields(model.TrackSuggestionRow)) == SUG_COLS


def test_as_row_returns_positional_tuple_in_field_order():
    r = model.SetTrackSlotRow(*range(len(SLOT_COLS)))
    assert model.as_row(r) == tuple(range(len(SLOT_COLS)))


def test_parse_structs_reexported():
    for name in ("TrackRow", "SuggestionRow", "NoticeRow", "IDTrack"):
        assert hasattr(model, name)
