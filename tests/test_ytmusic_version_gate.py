"""Regression guard for the YT Music rescue version-match gate.

Bug: the rescue searched by 'Artist - Title (Remixer Remix)' but installed
hits[0] unconditionally — so a query for a named remix/bootleg silently
resolved to the official original (correct title, WRONG audio). The gate
refuses a download when no 'songs' hit carries the query's version qualifier.
"""

from __future__ import annotations

from ingest.adapters.ytmusic_adapter import (
    YTMSearchHit,
    _expected_version_tokens,
    _select_hit,
    pick_search_hit,
)


def _h(title: str) -> YTMSearchHit:
    return YTMSearchHit(video_id="x", title=title, artists=(), duration_s=None)


def test_tokens_named_remixer():
    assert _expected_version_tokens("Adele - Someone Like You (Vicetone Remix)") == [
        "vicetone"
    ]
    assert _expected_version_tokens(
        "Backstreet Boys - Everybody (Oski & Apashe & Lennon Bootleg)"
    ) == ["oski", "apashe", "lennon"]


def test_tokens_bare_version_and_original():
    assert _expected_version_tokens("Lorde - Team (Remix)") == ["remix"]
    assert _expected_version_tokens("Daft Punk - Around The World") == []  # original
    # vocal-axis qualifier is not a version qualifier
    assert _expected_version_tokens("X - Y (Instrumental)") == []


def test_refuses_original_when_remix_requested():
    # only the original is available -> refuse (None), do NOT install it
    hits = (_h("Got The Love"), _h("Got The Love (Extended Mix)"))
    assert _select_hit("X - Got The Love (Vanze Bootleg)", hits) is None


def test_picks_named_version_when_present():
    hits = (_h("Got The Love"), _h("Got The Love (Vanze Bootleg)"))
    sel = _select_hit("X - Got The Love (Vanze Bootleg)", hits)
    assert sel is not None and "vanze" in sel.title.lower()


def test_original_query_accepts_top_hit():
    hits = (_h("Around The World"), _h("Around The World (Radio Edit)"))
    sel = _select_hit("Daft Punk - Around The World", hits)
    assert sel is hits[0]


def test_pick_search_hit_filters_duration():
    hits = (
        YTMSearchHit(
            video_id="long",
            title="Wrong Song",
            artists=(),
            duration_s=2000.0,
        ),
        _h("Got The Love (Vanze Bootleg)"),
    )
    sel = pick_search_hit("X - Got The Love (Vanze Bootleg)", hits)
    assert sel is not None and "vanze" in sel.title.lower()


def test_search_and_pick_refuses_named_remix_without_match():
    """Integration-style: mock search unavailable; unit test pick path via pick_search_hit."""
    hits = (_h("Someone Like You"),)
    assert pick_search_hit("Adele - Someone Like You (Vicetone Remix)", hits) is None
