# tests/test_soundcloud_first_look.py
"""Tests for soundcloud.analysis.first_look over a seeded lake."""

from __future__ import annotations

from soundcloud import store
from soundcloud.analysis import first_look
from soundcloud.records import ScTrack, ScUser

FA = "2026-07-18T00:00:00Z"


def _seed(db):
    store.init_db(db)
    with store.connect(db) as conn:
        store.upsert_user(
            conn, ScUser(7, "me", "Me", None, None, False, None, None, None, FA, "r")
        )
        store.upsert_track(
            conn,
            ScTrack(100, "A", 42, "house", None, None, None, None, None, None, FA, "r"),
        )
        store.upsert_track(
            conn,
            ScTrack(101, "B", 42, "house", None, None, None, None, None, None, FA, "r"),
        )
        store.upsert_track(
            conn,
            ScTrack(
                102, "C", 99, "techno", None, None, None, None, None, None, FA, "r"
            ),
        )
        store.add_like(conn, 7, 100, None)
        store.add_like(conn, 7, 101, None)
        store.add_like(conn, 7, 102, None)
        store.add_follow(conn, 7, 42)  # follows owner 42 (of liked tracks)
        conn.commit()


def test_report_fields(tmp_path):
    db = tmp_path / "sc.db"
    _seed(db)
    r = first_look.report(db)
    assert r["genre_distribution"] == {"house": 2, "techno": 1}
    assert r["top_liked_owners"][0] == (42, 2)
    assert r["following_like_overlap"] == 1
