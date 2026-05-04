"""Tests for the pure-function parts of audio_pipeline.models."""
from __future__ import annotations

import pytest

from audio_pipeline.models import (
    MediaSource, Track,
    youtube_url, soundcloud_api_url, spotify_track_url,
    DOWNLOAD_PLATFORM_PRIORITY,
)


# ---------- URL builders -----------------------------------------------------

def test_youtube_url_uses_watch_endpoint():
    assert youtube_url("7Iqgcfwl8WE") == "https://www.youtube.com/watch?v=7Iqgcfwl8WE"


def test_soundcloud_api_url_is_numeric_tracks_endpoint():
    # yt-dlp accepts this form for numeric SC IDs — verified against a real Big Bootie row.
    assert soundcloud_api_url("187673842") == "https://api.soundcloud.com/tracks/187673842"


def test_spotify_track_url_uses_open_spotify():
    assert spotify_track_url("1hKCTTdqf6cR5SJ8EhK5v0") == \
        "https://open.spotify.com/track/1hKCTTdqf6cR5SJ8EhK5v0"


# ---------- Track.source_for -------------------------------------------------

def _mk_sources(*platforms: str) -> tuple[MediaSource, ...]:
    return tuple(
        MediaSource(platform=p, player_id=f"id-{p}", url=f"https://x/{p}")
        for p in platforms
    )


def test_source_for_returns_matching_platform():
    t = Track("T1", ("tlp1",), _mk_sources("youtube", "spotify"))
    yt = t.source_for("youtube")
    assert yt is not None and yt.player_id == "id-youtube"


def test_source_for_returns_none_when_missing():
    t = Track("T1", ("tlp1",), _mk_sources("spotify"))
    assert t.source_for("youtube") is None


def test_source_for_on_empty_sources():
    t = Track("T1", ("tlp1",), ())
    assert t.source_for("youtube") is None


# ---------- platform priority ------------------------------------------------

def test_download_platform_priority_prefers_youtube():
    """We verified in the EDA that 99% of canonical Big Bootie tracks have a
    YouTube link whereas only ~2% have SoundCloud, so YT is the primary target."""
    assert DOWNLOAD_PLATFORM_PRIORITY[0] == "youtube"
    assert "soundcloud" in DOWNLOAD_PLATFORM_PRIORITY
