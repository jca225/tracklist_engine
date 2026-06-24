"""Tests for role-aware search queries."""

from __future__ import annotations

from ingest.search_query import (
    TrackSearchMeta,
    to_search_query_for_claim,
    to_search_query_for_meta,
)


def test_bed_preserves_mashup_name():
    q = to_search_query_for_claim(
        full_name="Quintino vs DJ Kool - Let Me Clear My Scorpion (Henry Fong Mashup)",
        artists_csv="Quintino",
        title="Let Me Clear My Scorpion",
        layer_role="bed",
        version="mashup",
    )
    assert "Henry Fong" in q
    assert "Mashup" in q


def test_payload_adds_acapella_suffix():
    q = to_search_query_for_claim(
        full_name="DJ Kool - Let Me Clear My Throat",
        artists_csv="DJ Kool",
        title="Let Me Clear My Throat",
        layer_role="payload",
        claimed_stem="regular",
    )
    assert "acapella" in q.lower()


def test_meta_helper():
    meta = TrackSearchMeta(
        full_name="Artist - Title (Remix)",
        artists_csv="Artist",
        title="Title",
        version="remix",
    )
    q = to_search_query_for_meta(meta, layer_role="solo")
    assert "Remix" in q
