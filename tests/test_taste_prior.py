"""Tests for taste_prior warehouse (no network)."""
from __future__ import annotations

from pathlib import Path

from personalization.collect import make_user_id
from personalization.persistence import connect, init_db, insert_comments, insert_likes, status_counts, upsert_listener
from personalization.records import ListenerRow, ScLikeRow, ScMixCommentRow


def test_make_user_id_stable():
    assert make_user_id("soundcloud", "foo") == make_user_id("soundcloud", "foo")
    assert make_user_id("soundcloud", "Foo") == make_user_id("soundcloud", "foo")


def test_warehouse_roundtrip(tmp_path: Path):
    db = tmp_path / "t.db"
    init_db(db)
    with connect(db) as conn:
        upsert_listener(
            conn,
            ListenerRow(
                user_id=make_user_id("soundcloud", "testuser"),
                platform="soundcloud",
                handle="testuser",
                mix_id="1fsnxchk",
                sc_user_id=123,
                first_seen_at="2026-01-01T00:00:00+00:00",
            ),
        )
        conn.commit()
        n = insert_likes(
            conn,
            (
                ScLikeRow(
                    user_id=make_user_id("soundcloud", "testuser"),
                    mix_id="1fsnxchk",
                    sc_user_id=123,
                    liked_at="2026-01-02T00:00:00+00:00",
                    track_id=999,
                    track_title="Test Track",
                    track_permalink="t",
                    track_artist_username="artist",
                    track_genre="Electronic",
                    raw_json="{}",
                ),
            ),
        )
        assert n == 1
        stats = status_counts(conn)
        assert stats["listeners_by_mix"]["1fsnxchk"] == 1
        assert stats["likes_by_mix"]["1fsnxchk"] == 1


def test_mix_comment_roundtrip(tmp_path: Path):
    db = tmp_path / "t.db"
    init_db(db)
    uid = make_user_id("soundcloud", "commenter")
    with connect(db) as conn:
        upsert_listener(
            conn,
            ListenerRow(
                user_id=uid,
                platform="soundcloud",
                handle="commenter",
                mix_id="1fsnxchk",
                sc_user_id=42,
                first_seen_at="2026-01-01T00:00:00+00:00",
            ),
        )
        conn.commit()
        n = insert_comments(
            conn,
            (
                ScMixCommentRow(
                    user_id=uid,
                    mix_id="1fsnxchk",
                    sc_user_id=42,
                    sc_track_id=317238901,
                    comment_id=123,
                    commented_at="2026-04-17T21:22:26Z",
                    mix_position_ms=746635,
                    body="This is still the best one",
                    raw_json="{}",
                ),
            ),
        )
        assert n == 1
        stats = status_counts(conn)
        assert stats["comments_by_mix"]["1fsnxchk"] == 1
        assert stats["comments_with_mix_position_ms"]["1fsnxchk"] == 1
