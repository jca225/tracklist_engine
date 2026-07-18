"""SoundCloud data-lake SQLite store: connection, schema init, parse + upsert."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from soundcloud.records import ScPlaylist, ScTrack, ScUser

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=60.0)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 60000")
        yield conn
    finally:
        conn.close()


def init_db(db_path: Path) -> None:
    sql = SCHEMA_PATH.read_text()
    with connect(db_path) as conn:
        conn.executescript(sql)
        conn.commit()


def _b(v: object) -> int:
    return 1 if v else 0


def parse_user(raw: dict, *, fetched_at: str, raw_ref: str) -> ScUser:
    return ScUser(
        sc_user_id=int(raw["id"]),
        permalink=raw.get("permalink", ""),
        username=raw.get("username", ""),
        followers_count=raw.get("followers_count"),
        followings_count=raw.get("followings_count"),
        verified=bool(raw.get("verified", False)),
        city=raw.get("city"),
        country=raw.get("country_code") or raw.get("country"),
        description=raw.get("description"),
        fetched_at=fetched_at,
        raw_ref=raw_ref,
    )


def parse_track(raw: dict, *, fetched_at: str, raw_ref: str) -> ScTrack:
    owner = raw.get("user") or {}
    return ScTrack(
        sc_track_id=int(raw["id"]),
        title=raw.get("title", ""),
        owner_sc_user_id=int(owner.get("id", 0)),
        genre=raw.get("genre"),
        tag_list=raw.get("tag_list"),
        duration_ms=raw.get("duration"),
        playback_count=raw.get("playback_count"),
        likes_count=raw.get("likes_count"),
        created_at=raw.get("created_at"),
        permalink=raw.get("permalink"),
        fetched_at=fetched_at,
        raw_ref=raw_ref,
    )


def parse_playlist(raw: dict, *, fetched_at: str, raw_ref: str) -> ScPlaylist:
    owner = raw.get("user") or {}
    return ScPlaylist(
        sc_playlist_id=int(raw["id"]),
        title=raw.get("title", ""),
        owner_sc_user_id=int(owner.get("id", 0)),
        track_count=raw.get("track_count"),
        is_album=bool(raw.get("is_album", False)),
        fetched_at=fetched_at,
        raw_ref=raw_ref,
    )


def upsert_user(conn: sqlite3.Connection, u: ScUser) -> None:
    conn.execute(
        """
        INSERT INTO sc_users (sc_user_id, permalink, username, followers_count,
            followings_count, verified, city, country, description, fetched_at, raw_ref)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(sc_user_id) DO UPDATE SET
            permalink=excluded.permalink, username=excluded.username,
            followers_count=excluded.followers_count, followings_count=excluded.followings_count,
            verified=excluded.verified, city=excluded.city, country=excluded.country,
            description=excluded.description, fetched_at=excluded.fetched_at, raw_ref=excluded.raw_ref
        """,
        (u.sc_user_id, u.permalink, u.username, u.followers_count, u.followings_count,
         _b(u.verified), u.city, u.country, u.description, u.fetched_at, u.raw_ref),
    )


def upsert_track(conn: sqlite3.Connection, t: ScTrack) -> None:
    conn.execute(
        """
        INSERT INTO sc_tracks (sc_track_id, title, owner_sc_user_id, genre, tag_list,
            duration_ms, playback_count, likes_count, created_at, permalink, fetched_at, raw_ref)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(sc_track_id) DO UPDATE SET
            title=excluded.title, owner_sc_user_id=excluded.owner_sc_user_id,
            genre=excluded.genre, tag_list=excluded.tag_list, duration_ms=excluded.duration_ms,
            playback_count=excluded.playback_count, likes_count=excluded.likes_count,
            created_at=excluded.created_at, permalink=excluded.permalink,
            fetched_at=excluded.fetched_at, raw_ref=excluded.raw_ref
        """,
        (t.sc_track_id, t.title, t.owner_sc_user_id, t.genre, t.tag_list, t.duration_ms,
         t.playback_count, t.likes_count, t.created_at, t.permalink, t.fetched_at, t.raw_ref),
    )


def upsert_playlist(conn: sqlite3.Connection, p: ScPlaylist) -> None:
    conn.execute(
        """
        INSERT INTO sc_playlists (sc_playlist_id, title, owner_sc_user_id, track_count,
            is_album, fetched_at, raw_ref)
        VALUES (?,?,?,?,?,?,?)
        ON CONFLICT(sc_playlist_id) DO UPDATE SET
            title=excluded.title, owner_sc_user_id=excluded.owner_sc_user_id,
            track_count=excluded.track_count, is_album=excluded.is_album,
            fetched_at=excluded.fetched_at, raw_ref=excluded.raw_ref
        """,
        (p.sc_playlist_id, p.title, p.owner_sc_user_id, p.track_count, _b(p.is_album),
         p.fetched_at, p.raw_ref),
    )


def add_like(conn: sqlite3.Connection, sc_user_id: int, sc_track_id: int, created_at: str | None) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO sc_likes (sc_user_id, sc_track_id, created_at) VALUES (?,?,?)",
        (sc_user_id, sc_track_id, created_at),
    )


def add_repost(conn: sqlite3.Connection, sc_user_id: int, item_kind: str, item_id: int, created_at: str | None) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO sc_reposts (sc_user_id, item_kind, item_id, created_at) VALUES (?,?,?,?)",
        (sc_user_id, item_kind, item_id, created_at),
    )


def add_follow(conn: sqlite3.Connection, follower_id: int, followee_id: int) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO sc_follows (follower_sc_user_id, followee_sc_user_id) VALUES (?,?)",
        (follower_id, followee_id),
    )


def add_playlist_track(conn: sqlite3.Connection, playlist_id: int, track_id: int, position: int) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO sc_playlist_tracks (sc_playlist_id, sc_track_id, position) VALUES (?,?,?)",
        (playlist_id, track_id, position),
    )


def mark_done(conn: sqlite3.Connection, sc_user_id: int, phase: str, now: str) -> None:
    conn.execute(
        """
        INSERT INTO crawl_checkpoints (sc_user_id, phase, done, updated_at)
        VALUES (?,?,1,?)
        ON CONFLICT(sc_user_id, phase) DO UPDATE SET done=1, updated_at=excluded.updated_at
        """,
        (sc_user_id, phase, now),
    )


def is_done(conn: sqlite3.Connection, sc_user_id: int, phase: str) -> bool:
    row = conn.execute(
        "SELECT done FROM crawl_checkpoints WHERE sc_user_id=? AND phase=?",
        (sc_user_id, phase),
    ).fetchone()
    return bool(row and row["done"])
