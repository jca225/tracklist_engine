"""SQLite warehouse for taste prior data."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .records import ListenerBotScoreRow, ListenerRow, ScLikeRow, ScMixCommentRow, ScPlaylistRow

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

_LISTENER_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("username", "TEXT"),
    ("followers_count", "INTEGER"),
    ("followings_count", "INTEGER"),
    ("verified", "INTEGER DEFAULT 0"),
    ("city", "TEXT"),
    ("country_code", "TEXT"),
)


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})")}


def migrate_db(conn: sqlite3.Connection) -> None:
    """Apply additive schema changes to an existing warehouse."""
    cols = _table_columns(conn, "listeners")
    for name, typedef in _LISTENER_MIGRATIONS:
        if name not in cols:
            conn.execute(f"ALTER TABLE listeners ADD COLUMN {name} {typedef}")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sc_playlists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            mix_id TEXT NOT NULL,
            sc_user_id INTEGER NOT NULL,
            playlist_id INTEGER NOT NULL,
            title TEXT,
            track_count INTEGER,
            track_ids_json TEXT NOT NULL,
            created_at TEXT,
            last_modified TEXT,
            raw_json TEXT NOT NULL,
            UNIQUE(sc_user_id, playlist_id),
            FOREIGN KEY (user_id) REFERENCES listeners(user_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_sc_playlists_user ON sc_playlists(sc_user_id);
        CREATE INDEX IF NOT EXISTS idx_sc_playlists_mix ON sc_playlists(mix_id);
        CREATE TABLE IF NOT EXISTS listener_bot_scores (
            user_id TEXT PRIMARY KEY,
            mix_id TEXT NOT NULL,
            sc_user_id INTEGER,
            bot_score REAL NOT NULL,
            is_bot INTEGER NOT NULL DEFAULT 0,
            reasons_json TEXT NOT NULL DEFAULT '[]',
            computed_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES listeners(user_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_listener_bot_mix ON listener_bot_scores(mix_id);
        CREATE TABLE IF NOT EXISTS sc_track_mert (
            sc_track_id INTEGER NOT NULL,
            mert_version TEXT NOT NULL,
            dim INTEGER NOT NULL DEFAULT 1024,
            embedding BLOB NOT NULL,
            source_url TEXT,
            embedded_at TEXT NOT NULL,
            PRIMARY KEY (sc_track_id, mert_version)
        );
        CREATE TABLE IF NOT EXISTS user_prior_vectors (
            user_id TEXT PRIMARY KEY,
            mix_id TEXT NOT NULL,
            sc_user_id INTEGER,
            mert_version TEXT NOT NULL,
            dim INTEGER NOT NULL DEFAULT 1024,
            n_tracks_used INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            computed_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES listeners(user_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_user_prior_mix ON user_prior_vectors(mix_id);
        CREATE TABLE IF NOT EXISTS taste_clusters (
            user_id TEXT NOT NULL,
            mix_id TEXT NOT NULL,
            cluster_id INTEGER NOT NULL,
            algorithm TEXT NOT NULL,
            computed_at TEXT NOT NULL,
            PRIMARY KEY (user_id, mix_id, algorithm),
            FOREIGN KEY (user_id) REFERENCES listeners(user_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_taste_clusters_mix ON taste_clusters(mix_id, cluster_id);
        """
    )
    conn.commit()


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 60000")
    return conn


def init_db(db_path: Path) -> None:
    sql = SCHEMA_PATH.read_text()
    with connect(db_path) as conn:
        conn.executescript(sql)
        migrate_db(conn)
        conn.commit()


def load_checkpoint(conn: sqlite3.Connection, mix_id: str, phase: str) -> dict[str, Any]:
    row = conn.execute(
        "SELECT checkpoint_json FROM scrape_checkpoints WHERE mix_id = ? AND phase = ?",
        (mix_id, phase),
    ).fetchone()
    if row is None:
        return {}
    return json.loads(row["checkpoint_json"])


def save_checkpoint(conn: sqlite3.Connection, mix_id: str, phase: str, data: dict[str, Any]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO scrape_checkpoints (mix_id, phase, checkpoint_json, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(mix_id, phase) DO UPDATE SET
            checkpoint_json = excluded.checkpoint_json,
            updated_at = excluded.updated_at
        """,
        (mix_id, phase, json.dumps(data, default=str), now),
    )
    conn.commit()


def upsert_listener(conn: sqlite3.Connection, row: ListenerRow) -> None:
    conn.execute(
        """
        INSERT INTO listeners (
            user_id, platform, handle, mix_id, sc_user_id, first_seen_at,
            source_evidence_json, username, followers_count, followings_count,
            verified, city, country_code
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            source_evidence_json = excluded.source_evidence_json,
            username = COALESCE(excluded.username, listeners.username),
            followers_count = COALESCE(excluded.followers_count, listeners.followers_count),
            followings_count = COALESCE(excluded.followings_count, listeners.followings_count),
            verified = CASE WHEN excluded.verified THEN 1 ELSE listeners.verified END,
            city = COALESCE(excluded.city, listeners.city),
            country_code = COALESCE(excluded.country_code, listeners.country_code)
        """,
        (
            row.user_id,
            row.platform,
            row.handle,
            row.mix_id,
            row.sc_user_id,
            row.first_seen_at,
            row.source_evidence_json,
            row.username,
            row.followers_count,
            row.followings_count,
            1 if row.verified else 0,
            row.city,
            row.country_code,
        ),
    )


def insert_likes(conn: sqlite3.Connection, rows: tuple[ScLikeRow, ...]) -> int:
    n = 0
    for r in rows:
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO sc_likes
            (user_id, mix_id, sc_user_id, liked_at, track_id, track_title,
             track_permalink, track_artist_username, track_genre, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r.user_id,
                r.mix_id,
                r.sc_user_id,
                r.liked_at,
                r.track_id,
                r.track_title,
                r.track_permalink,
                r.track_artist_username,
                r.track_genre,
                r.raw_json,
            ),
        )
        n += cur.rowcount
    conn.commit()
    return n


def insert_comments(conn: sqlite3.Connection, rows: tuple[ScMixCommentRow, ...]) -> int:
    n = 0
    for r in rows:
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO sc_mix_comments
            (user_id, mix_id, sc_user_id, sc_track_id, comment_id, commented_at,
             mix_position_ms, body, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r.user_id,
                r.mix_id,
                r.sc_user_id,
                r.sc_track_id,
                r.comment_id,
                r.commented_at,
                r.mix_position_ms,
                r.body,
                r.raw_json,
            ),
        )
        n += cur.rowcount
    conn.commit()
    return n


def insert_playlists(conn: sqlite3.Connection, rows: tuple[ScPlaylistRow, ...]) -> int:
    n = 0
    for r in rows:
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO sc_playlists
            (user_id, mix_id, sc_user_id, playlist_id, title, track_count,
             track_ids_json, created_at, last_modified, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r.user_id,
                r.mix_id,
                r.sc_user_id,
                r.playlist_id,
                r.title,
                r.track_count,
                r.track_ids_json,
                r.created_at,
                r.last_modified,
                r.raw_json,
            ),
        )
        n += cur.rowcount
    conn.commit()
    return n


def upsert_bot_scores(conn: sqlite3.Connection, rows: tuple[ListenerBotScoreRow, ...]) -> int:
    n = 0
    for r in rows:
        conn.execute(
            """
            INSERT INTO listener_bot_scores
            (user_id, mix_id, sc_user_id, bot_score, is_bot, reasons_json, computed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                bot_score = excluded.bot_score,
                is_bot = excluded.is_bot,
                reasons_json = excluded.reasons_json,
                computed_at = excluded.computed_at
            """,
            (
                r.user_id,
                r.mix_id,
                r.sc_user_id,
                r.bot_score,
                1 if r.is_bot else 0,
                r.reasons_json,
                r.computed_at,
            ),
        )
        n += 1
    conn.commit()
    return n


def listener_sc_ids(conn: sqlite3.Connection, mix_id: str) -> list[int]:
    rows = conn.execute(
        "SELECT sc_user_id FROM listeners WHERE mix_id = ? AND sc_user_id IS NOT NULL ORDER BY sc_user_id",
        (mix_id,),
    ).fetchall()
    return [int(r["sc_user_id"]) for r in rows]


def status_counts(conn: sqlite3.Connection) -> dict[str, Any]:
    listeners = conn.execute("SELECT mix_id, COUNT(*) AS n FROM listeners GROUP BY mix_id").fetchall()
    likes = conn.execute("SELECT mix_id, COUNT(*) AS n FROM sc_likes GROUP BY mix_id").fetchall()
    comments = conn.execute("SELECT mix_id, COUNT(*) AS n FROM sc_mix_comments GROUP BY mix_id").fetchall()
    with_position = conn.execute(
        "SELECT mix_id, COUNT(*) AS n FROM sc_mix_comments WHERE mix_position_ms IS NOT NULL GROUP BY mix_id"
    ).fetchall()
    playlists = conn.execute("SELECT mix_id, COUNT(*) AS n FROM sc_playlists GROUP BY mix_id").fetchall()
    bots = conn.execute(
        "SELECT mix_id, COUNT(*) AS n FROM listener_bot_scores WHERE is_bot = 1 GROUP BY mix_id"
    ).fetchall()
    priors = conn.execute("SELECT mix_id, COUNT(*) AS n FROM user_prior_vectors GROUP BY mix_id").fetchall()
    mert_tracks = conn.execute("SELECT COUNT(*) AS n FROM sc_track_mert").fetchone()
    return {
        "listeners_by_mix": {r["mix_id"]: r["n"] for r in listeners},
        "likes_by_mix": {r["mix_id"]: r["n"] for r in likes},
        "comments_by_mix": {r["mix_id"]: r["n"] for r in comments},
        "comments_with_mix_position_ms": {r["mix_id"]: r["n"] for r in with_position},
        "playlists_by_mix": {r["mix_id"]: r["n"] for r in playlists},
        "bots_by_mix": {r["mix_id"]: r["n"] for r in bots},
        "user_priors_by_mix": {r["mix_id"]: r["n"] for r in priors},
        "sc_track_mert_cached": int(mert_tracks["n"]) if mert_tracks else 0,
    }


def clean_user_ids(conn: sqlite3.Connection, mix_id: str, *, exclude_bots: bool = True) -> list[str]:
    """Listeners with bot filter applied."""
    if exclude_bots:
        rows = conn.execute(
            """
            SELECT l.user_id FROM listeners l
            LEFT JOIN listener_bot_scores b ON b.user_id = l.user_id
            WHERE l.mix_id = ? AND COALESCE(b.is_bot, 0) = 0
            """,
            (mix_id,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT user_id FROM listeners WHERE mix_id = ?", (mix_id,)).fetchall()
    return [str(r["user_id"]) for r in rows]


def upsert_track_mert(
    conn: sqlite3.Connection,
    *,
    sc_track_id: int,
    mert_version: str,
    embedding: bytes,
    dim: int,
    source_url: str | None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO sc_track_mert (sc_track_id, mert_version, dim, embedding, source_url, embedded_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(sc_track_id, mert_version) DO UPDATE SET
            embedding = excluded.embedding,
            source_url = excluded.source_url,
            embedded_at = excluded.embedded_at
        """,
        (sc_track_id, mert_version, dim, embedding, source_url, now),
    )
    conn.commit()


def get_track_mert(conn: sqlite3.Connection, sc_track_id: int, mert_version: str) -> bytes | None:
    row = conn.execute(
        "SELECT embedding FROM sc_track_mert WHERE sc_track_id = ? AND mert_version = ?",
        (sc_track_id, mert_version),
    ).fetchone()
    return bytes(row["embedding"]) if row else None


def upsert_user_prior(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    mix_id: str,
    sc_user_id: int | None,
    mert_version: str,
    dim: int,
    n_tracks_used: int,
    embedding: bytes,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO user_prior_vectors
        (user_id, mix_id, sc_user_id, mert_version, dim, n_tracks_used, embedding, computed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            mix_id = excluded.mix_id,
            sc_user_id = excluded.sc_user_id,
            mert_version = excluded.mert_version,
            dim = excluded.dim,
            n_tracks_used = excluded.n_tracks_used,
            embedding = excluded.embedding,
            computed_at = excluded.computed_at
        """,
        (user_id, mix_id, sc_user_id, mert_version, dim, n_tracks_used, embedding, now),
    )
    conn.commit()


def replace_taste_clusters(
    conn: sqlite3.Connection,
    mix_id: str,
    algorithm: str,
    assignments: tuple[tuple[str, int], ...],
) -> int:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("DELETE FROM taste_clusters WHERE mix_id = ? AND algorithm = ?", (mix_id, algorithm))
    for user_id, cluster_id in assignments:
        conn.execute(
            """
            INSERT INTO taste_clusters (user_id, mix_id, cluster_id, algorithm, computed_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, mix_id, cluster_id, algorithm, now),
        )
    conn.commit()
    return len(assignments)


def log_run(
    conn: sqlite3.Connection,
    *,
    phase: str,
    mix_id: str,
    started_at: str,
    output_rows: int,
    params: dict[str, Any],
) -> None:
    finished = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO scrape_runs (phase, mix_id, started_at, finished_at, output_rows, params_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (phase, mix_id, started_at, finished, output_rows, json.dumps(params)),
    )
    conn.commit()
