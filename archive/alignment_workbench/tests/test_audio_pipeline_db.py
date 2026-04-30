"""Tests for the DB adapter using an in-memory SQLite with the real schema."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from archive.audio_pipeline.adapters import db as db_adapter
from archive.audio_pipeline.models import AudioAsset
from archive.audio_pipeline.result import Ok, Err


_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "web_crawler" / "database" / "schema.sql"


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Build a real on-disk SQLite from schema.sql and seed minimal fixtures.
    The adapter uses its own connections per call, so a tmp file is cleaner
    than sharing an in-memory conn."""
    path = tmp_path / "test.db"
    schema = _SCHEMA_PATH.read_text()
    conn = sqlite3.connect(path)
    conn.executescript(schema)

    conn.execute(
        "INSERT INTO dj_sets (set_id, set_url, title) VALUES (?, ?, ?)",
        ("S_TEST", "https://x/S_TEST", "Two Friends - Big Bootie Mix Vol. 99"),
    )
    # Canonical track T1 with both a YouTube and a Spotify link;
    # T2 with only Spotify (forces spotdl fallback);
    # T3 with a NULL track_id (must be filtered out by load_set_tracks).
    conn.executemany(
        """
        INSERT INTO dj_set_track_media_links
        (set_id, tlp_id, track_id, platform, player_id)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            ("S_TEST", "tlp-1", "T1", "youtube", "vid-abc"),
            ("S_TEST", "tlp-1", "T1", "spotify", "sp-xyz"),
            ("S_TEST", "tlp-2", "T2", "spotify", "sp-only"),
            ("S_TEST", "tlp-3", None,  "youtube", "vid-orphan"),
        ],
    )
    conn.commit()
    conn.close()
    return path


# ---------- load_set_tracks --------------------------------------------------

def test_load_set_tracks_groups_sources_by_canonical_id(db_path: Path):
    r = db_adapter.load_set_tracks(db_path, "S_TEST")
    assert isinstance(r, Ok)

    by_id = {t.track_id: t for t in r.value}
    assert set(by_id.keys()) == {"T1", "T2"}    # T3 filtered: null track_id

    t1 = by_id["T1"]
    platforms = sorted(s.platform for s in t1.sources)
    assert platforms == ["spotify", "youtube"]
    yt = next(s for s in t1.sources if s.platform == "youtube")
    assert yt.url == "https://www.youtube.com/watch?v=vid-abc"


def test_load_set_tracks_unknown_set_returns_empty_ok(db_path: Path):
    r = db_adapter.load_set_tracks(db_path, "NOPE")
    assert isinstance(r, Ok) and r.value == ()


# ---------- insert_audio + already_downloaded round-trip --------------------

def test_insert_then_already_downloaded_round_trip(db_path: Path):
    asset = AudioAsset(
        track_audio_id=None,
        track_id="T1", platform="youtube",
        source_url="https://www.youtube.com/watch?v=vid-abc",
        player_id="vid-abc",
        path="/tmp/fake.m4a", sha256="deadbeef",
        duration_s=210.5, sample_rate=44100,
        codec="m4a", bitrate_kbps=128,
    )

    first = db_adapter.insert_audio(db_path, asset)
    assert isinstance(first, Ok) and first.value > 0

    # already_downloaded now returns True for the same (track_id, platform, player_id)
    present = db_adapter.already_downloaded(db_path, "T1", "youtube", "vid-abc")
    assert isinstance(present, Ok) and present.value is True

    # A different player_id is still considered not-yet-downloaded.
    absent = db_adapter.already_downloaded(db_path, "T1", "youtube", "other-vid")
    assert isinstance(absent, Ok) and absent.value is False

    # Re-insert is idempotent — returns Ok with the same row id, doesn't raise.
    second = db_adapter.insert_audio(db_path, asset)
    assert isinstance(second, Ok) and second.value == first.value
