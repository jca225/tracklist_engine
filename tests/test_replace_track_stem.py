"""Tests for stem-aware replace_track_audio helpers and pull suffixes."""
from __future__ import annotations

from labeling.pull_set_for_alignment import _qualifier_suffix
from scripts.replace_track_audio import (
    _resolve_axis_for_ledger,
    _resolve_stem_for_replace,
)


def test_resolve_stem_cli_overrides_old():
    old = {"stem": "acappella", "platform": "youtube_music"}
    assert _resolve_stem_for_replace("instrumental", old) == "instrumental"


def test_resolve_stem_inherits_from_retired_row():
    old = {"stem": "acappella", "platform": "manual"}
    assert _resolve_stem_for_replace(None, old) == "acappella"


def test_resolve_stem_defaults_regular():
    assert _resolve_stem_for_replace(None, None) == "regular"


def test_resolve_axis_auto_stem_for_acappella():
    assert _resolve_axis_for_ledger("version", "acappella") == "stem"
    assert _resolve_axis_for_ledger("version", "regular") == "version"
    assert _resolve_axis_for_ledger("variant", "acappella") == "variant"


def test_qualifier_suffix_compound_remix_and_acappella():
    suffix = _qualifier_suffix(
        "Artist - Title (Syn Cole Remix)",
        "remix",
        stem="acappella",
        variant="regular",
    )
    assert suffix == " (Syn Cole Remix) (Acappella)"


def test_qualifier_suffix_stem_only():
    assert _qualifier_suffix("Artist - Title", "original", stem="instrumental") == " (Instrumental)"


def test_qualifier_suffix_extended_with_remix():
    suffix = _qualifier_suffix(
        "Artist - Title (Syn Cole Remix)",
        "remix",
        stem="regular",
        variant="extended",
    )
    assert suffix == " (Syn Cole Remix) (Extended Mix)"
