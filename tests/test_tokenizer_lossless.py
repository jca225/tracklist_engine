"""Phase 1 of the tokenizer split: the parse is lossless (full_name keeps its
vocal/remixer qualifiers, is_instrumental is captured), and the download-side
search-query projection reproduces the old collapsed query.
"""
from __future__ import annotations

import pytest

from tokenizer.track_tokenizer import parse_track_row
from ingest.search_query import to_search_query


def _row(meta_name: str, *, track_value: str | None = None) -> str:
    """Minimal tlpItem row carrying a schema.org name meta tag."""
    tv = track_value or meta_name
    return (
        '<div class="tlpTog bItm tlpItem" data-trackid="abc123">'
        '  <div itemtype="http://schema.org/MusicRecording">'
        f'    <meta itemprop="name" content="{meta_name}"/>'
        f'    <span class="trackValue">{tv}</span>'
        '  </div>'
        '</div>'
    )


# ---- parse stays lossless ---------------------------------------------------

def test_full_name_keeps_instrumental_and_remixer_qualifiers() -> None:
    name = "Martin Garrix & Troye Sivan - There For You (Madison Mars Remix) (Instrumental)"
    tr = parse_track_row(_row(name, track_value=name))
    # The canonical record is verbatim — both the remixer and the (Instrumental)
    # qualifier survive (the latter used to be stripped at parse time).
    assert tr.full_name == name
    assert tr.is_instrumental is True


def test_acappella_is_stem_not_version_tag() -> None:
    name = "Some Artist - A Song (Acappella)"
    tr = parse_track_row(_row(name))
    assert tr.full_name == name
    assert tr.claimed_stem == "acappella"
    assert tr.version_tag is None
    assert tr.is_instrumental is False


def test_instrumental_claimed_stem() -> None:
    name = "Martin Garrix & Troye Sivan - There For You (Madison Mars Remix) (Instrumental)"
    tr = parse_track_row(_row(name, track_value=name))
    assert tr.claimed_stem == "instrumental"
    assert tr.is_instrumental is True
    assert tr.version_tag == "Remix"


def test_plain_track_has_no_instrumental_flag() -> None:
    name = "Daft Punk - Around the World"
    tr = parse_track_row(_row(name))
    assert tr.full_name == name
    assert tr.is_instrumental is False


# ---- download projection reproduces the old collapsed query -----------------

def test_search_query_strips_instrumental_keeps_remixer() -> None:
    q = to_search_query(
        "Martin Garrix & Troye Sivan - There For You (Madison Mars Remix) (Instrumental)",
        "Martin Garrix, Troye Sivan",
        "There For You",
    )
    assert q == "Martin Garrix & Troye Sivan - There For You (Madison Mars Remix)"


def test_search_query_strips_acappella() -> None:
    assert to_search_query("Some Artist - A Song (Acappella)", "Some Artist", "A Song") \
        == "Some Artist - A Song"


def test_search_query_falls_back_on_id_placeholder() -> None:
    # A literal "ID" in the query derails YT Music search → bare Artist - Title.
    assert to_search_query("Artist - ID (ID Remix)", "Artist", "ID") == "Artist - ID"


def test_search_query_falls_back_without_full_name() -> None:
    assert to_search_query(None, "Daft Punk", "One More Time") == "Daft Punk - One More Time"


def test_search_query_plain_full_name_unchanged() -> None:
    assert to_search_query("Daft Punk - Around the World", "Daft Punk", "Around the World") \
        == "Daft Punk - Around the World"
