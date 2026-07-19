# tests/test_soundcloud_store_upsert.py
"""Tests for soundcloud.store parse + upsert idempotency."""

from __future__ import annotations

from soundcloud import store
from soundcloud.records import ScTrack, ScUser

RAW_USER = {
    "id": 327506308,
    "permalink": "user-327506308",
    "username": "John",
    "followers_count": 12,
    "followings_count": 30,
    "verified": False,
    "city": "NYC",
    "country_code": "US",
    "description": "bio",
}
RAW_TRACK = {
    "id": 555,
    "title": "Heavy Thunder",
    "user": {"id": 42},
    "genre": "bass",
    "tag_list": "trap",
    "duration": 210000,
    "playback_count": 9,
    "likes_count": 3,
    "created_at": "2020/01/01 00:00:00 +0000",
    "permalink": "heavy-thunder",
}
FA = "2026-07-18T00:00:00Z"


def test_parse_user_maps_country_code():
    u = store.parse_user(RAW_USER, fetched_at=FA, raw_ref="r")
    assert isinstance(u, ScUser)
    assert u.sc_user_id == 327506308
    assert u.country == "US"
    assert u.verified is False


def test_parse_track_maps_owner_and_duration():
    t = store.parse_track(RAW_TRACK, fetched_at=FA, raw_ref="r")
    assert isinstance(t, ScTrack)
    assert t.owner_sc_user_id == 42
    assert t.duration_ms == 210000


def test_upsert_user_is_idempotent(tmp_path):
    db = tmp_path / "sc.db"
    store.init_db(db)
    u = store.parse_user(RAW_USER, fetched_at=FA, raw_ref="r")
    with store.connect(db) as conn:
        store.upsert_user(conn, u)
        store.upsert_user(conn, u)  # second time must not duplicate
        conn.commit()
        n = conn.execute("SELECT COUNT(*) FROM sc_users").fetchone()[0]
    assert n == 1


def test_edges_and_checkpoint(tmp_path):
    db = tmp_path / "sc.db"
    store.init_db(db)
    with store.connect(db) as conn:
        store.add_like(conn, 327506308, 555, FA)
        store.add_like(conn, 327506308, 555, FA)  # dedup
        store.add_follow(conn, 327506308, 42)
        assert store.is_done(conn, 327506308, "likes") is False
        store.mark_done(conn, 327506308, "likes", FA)
        conn.commit()
        assert store.is_done(conn, 327506308, "likes") is True
        assert conn.execute("SELECT COUNT(*) FROM sc_likes").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM sc_follows").fetchone()[0] == 1
