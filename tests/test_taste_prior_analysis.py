"""Tests for bot scoring and comment heatmap."""
from __future__ import annotations

import json
from pathlib import Path

from workspaces.taste_prior.bot_heuristics import score_mix_listeners
from workspaces.taste_prior.collect import make_user_id
from workspaces.taste_prior.comment_heatmap import build_heatmap
from workspaces.taste_prior.persistence import (
    connect,
    init_db,
    insert_comments,
    insert_likes,
    insert_playlists,
    migrate_db,
    upsert_listener,
)
from workspaces.taste_prior.records import ListenerRow, ScLikeRow, ScMixCommentRow, ScPlaylistRow
from workspaces.taste_prior.taste_cluster import cluster_mix


def test_bot_flags_generic_user(tmp_path: Path):
    db = tmp_path / "t.db"
    init_db(db)
    uid = make_user_id("soundcloud", "user-283964440")
    with connect(db) as conn:
        upsert_listener(
            conn,
            ListenerRow(
                user_id=uid,
                platform="soundcloud",
                handle="user-283964440",
                mix_id="2nvzlh2k",
                sc_user_id=99,
                first_seen_at="2026-01-01T00:00:00+00:00",
                username="user283964440",
                followers_count=0,
                followings_count=0,
            ),
        )
        conn.commit()
        for i in range(300):
            insert_likes(
                conn,
                (
                    ScLikeRow(
                        user_id=uid,
                        mix_id="2nvzlh2k",
                        sc_user_id=99,
                        liked_at=f"2024-01-01T{i % 24:02d}:00:00+00:00",
                        track_id=1000 + i,
                        track_title="t",
                        track_permalink="t",
                        track_artist_username="a",
                        track_genre="",
                        raw_json="{}",
                    ),
                ),
            )
        summary = score_mix_listeners(conn, "2nvzlh2k")
        assert summary["bots_flagged"] >= 1
        row = conn.execute(
            "SELECT bot_score, is_bot FROM listener_bot_scores WHERE user_id = ?", (uid,)
        ).fetchone()
        assert row["is_bot"] == 1
        assert row["bot_score"] >= 0.55


def test_comment_heatmap_bins(tmp_path: Path):
    db = tmp_path / "t.db"
    init_db(db)
    uid = make_user_id("soundcloud", "c")
    with connect(db) as conn:
        upsert_listener(
            conn,
            ListenerRow(
                user_id=uid,
                platform="soundcloud",
                handle="c",
                mix_id="2nvzlh2k",
                sc_user_id=1,
                first_seen_at="2026-01-01T00:00:00+00:00",
            ),
        )
        conn.commit()
        insert_comments(
            conn,
            (
                ScMixCommentRow(
                    user_id=uid,
                    mix_id="2nvzlh2k",
                    sc_user_id=1,
                    sc_track_id=1,
                    comment_id=1,
                    commented_at="2026-01-01T00:00:00+00:00",
                    mix_position_ms=60_000,
                    body="wow",
                    raw_json="{}",
                ),
                ScMixCommentRow(
                    user_id=uid,
                    mix_id="2nvzlh2k",
                    sc_user_id=1,
                    sc_track_id=1,
                    comment_id=2,
                    commented_at="2026-01-01T00:01:00+00:00",
                    mix_position_ms=90_000,
                    body="nice",
                    raw_json="{}",
                ),
            ),
        )
        hm = build_heatmap(conn, "2nvzlh2k", bin_width_s=30.0, mix_duration_s=120.0)
        assert hm.n_with_position == 2
        assert sum(hm.bin_counts) == 2


def test_taste_cluster_synthetic(tmp_path: Path):
    db = tmp_path / "t.db"
    init_db(db)
    users = []
    with connect(db) as conn:
        migrate_db(conn)
        for i in range(40):
            uid = make_user_id("soundcloud", f"u{i}")
            users.append(uid)
            upsert_listener(
                conn,
                ListenerRow(
                    user_id=uid,
                    platform="soundcloud",
                    handle=f"u{i}",
                    mix_id="2nvzlh2k",
                    sc_user_id=1000 + i,
                    first_seen_at="2026-01-01T00:00:00+00:00",
                ),
            )
        conn.commit()
        likes = []
        for i, uid in enumerate(users):
            for tid in range(20 + (i % 5)):
                likes.append(
                    ScLikeRow(
                        user_id=uid,
                        mix_id="2nvzlh2k",
                        sc_user_id=1000 + i,
                        liked_at="2026-01-01T00:00:00+00:00",
                        track_id=tid,
                        track_title="t",
                        track_permalink="t",
                        track_artist_username="a",
                        track_genre="",
                        raw_json="{}",
                    )
                )
        insert_likes(conn, tuple(likes))
        summary = cluster_mix(conn, "2nvzlh2k", exclude_bots=False, max_users=40, n_clusters=4)
        assert summary["users_clustered"] == 40
        assert len(summary["cluster_sizes"]) == 4


def test_playlist_roundtrip(tmp_path: Path):
    db = tmp_path / "t.db"
    init_db(db)
    uid = make_user_id("soundcloud", "dj")
    with connect(db) as conn:
        upsert_listener(
            conn,
            ListenerRow(
                user_id=uid,
                platform="soundcloud",
                handle="dj",
                mix_id="2nvzlh2k",
                sc_user_id=5,
                first_seen_at="2026-01-01T00:00:00+00:00",
            ),
        )
        conn.commit()
        n = insert_playlists(
            conn,
            (
                ScPlaylistRow(
                    user_id=uid,
                    mix_id="2nvzlh2k",
                    sc_user_id=5,
                    playlist_id=42,
                    title="my set",
                    track_count=2,
                    track_ids_json=json.dumps([1, 2]),
                    created_at="2026-01-01",
                    last_modified="2026-01-02",
                    raw_json="{}",
                ),
            ),
        )
        assert n == 1
