# tests/test_soundcloud_store_init.py
"""Tests for soundcloud.store schema init."""

from __future__ import annotations

from pathlib import Path

from soundcloud.store import connect, init_db


def _tables(db_path: Path) -> set[str]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    return {r["name"] for r in rows}


def test_init_creates_all_tables(tmp_path):
    db = tmp_path / "sc_lake.db"
    init_db(db)
    names = _tables(db)
    assert {
        "sc_users",
        "sc_tracks",
        "sc_playlists",
        "sc_likes",
        "sc_reposts",
        "sc_follows",
        "sc_playlist_tracks",
        "sc_recording_map",
        "crawl_checkpoints",
    } <= names


def test_init_is_idempotent(tmp_path):
    db = tmp_path / "sc_lake.db"
    init_db(db)
    init_db(db)  # second run must not raise
    assert "sc_users" in _tables(db)


def test_wal_and_busy_timeout(tmp_path):
    db = tmp_path / "sc_lake.db"
    init_db(db)
    with connect(db) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 60000
