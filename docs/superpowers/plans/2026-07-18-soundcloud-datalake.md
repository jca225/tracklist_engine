# SoundCloud Data-Lake Substrate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `soundcloud/` module — a reusable anonymous-`client_id` fetch layer + normalized SQLite lake — and use it to sync John's SoundCloud library (`user-327506308`) metadata-first into a pi-storage-hosted data lake.

**Architecture:** Bottom-up, each file one responsibility: `config` (settings/env) → `records` (frozen dataclasses) → `schema.sql` + `store` (SQLite nodes/edges + parse/upsert) → `client` (rate-limited transport, generalized from `personalization/soundcloud_client.py` which becomes a shim) → `fetch` (per-endpoint primitives) → `rawlake` (append-only JSONL audit) → `crawl` (depth-1 frontier driver) → `main` (CLI) → `analysis/first_look` (validation). No live network in tests — httpx is hand-mocked; SQLite is real on `tmp_path`.

**Tech Stack:** Python 3.14, `httpx`, stdlib `sqlite3`, `argparse`, `pytest`. Runs in `venvs/audio`.

## Global Constraints

- Every `.py` file starts with `from __future__ import annotations` and a one-line module docstring. Copied verbatim from house style.
- PEP 604 unions only (`int | None`), never `Optional[int]`.
- Records are `@dataclass(frozen=True)`.
- Auth is **anonymous `client_id` scraping only** — no OAuth, no `/me`, no private items. Public data of any user.
- Canonical persistence lives on **pi-storage** under `SC_LAKE_ROOT` (default `/mnt/storage/data/soundcloud`): `sc_lake.db` + `raw/{entity}/{id}/{fetched_at}.jsonl`. The `soundcloud/` package is code-only; tests use `tmp_path`.
- Store is **off-canonical** — never `music_database.db`, never on the alignment DAG.
- SQLite connections set `PRAGMA foreign_keys = ON`, `journal_mode = WAL`, `busy_timeout = 60000`.
- Core (`client`/`fetch`/`store`/`crawl`) returns/raises values; `main.py` is the fail-fast edge (returns int exit codes, `raise SystemExit(main())`).
- Timestamps (`fetched_at`) are ISO strings passed in by the caller/edge — core stays clock-free (no `datetime.now()` inside pure functions).
- Crawl default depth = 1 (seed's own collections only; neighbors are node rows + follow edges, NOT recursed).
- Run all commands from the worktree root with `venvs/audio/bin/python`.

---

### Task 1: Module scaffold + config

**Files:**
- Create: `soundcloud/__init__.py`
- Create: `soundcloud/config.py`
- Create: `tests/test_soundcloud_config.py`
- Modify: `.gitignore` (add `soundcloud/data/`)

**Interfaces:**
- Produces: `SoundCloudSettings(data_root: Path, db_path: Path, raw_root: Path, rpm: int)` (frozen); `load_settings(*, data_root: Path | None = None, rpm: int | None = None) -> SoundCloudSettings`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_soundcloud_config.py
"""Tests for soundcloud.config env + default resolution."""
from __future__ import annotations

from pathlib import Path

from soundcloud.config import load_settings


def test_defaults_point_at_pi_storage(monkeypatch):
    monkeypatch.delenv("SC_LAKE_ROOT", raising=False)
    monkeypatch.delenv("SC_LAKE_RPM", raising=False)
    s = load_settings()
    assert s.data_root == Path("/mnt/storage/data/soundcloud")
    assert s.db_path == Path("/mnt/storage/data/soundcloud/sc_lake.db")
    assert s.raw_root == Path("/mnt/storage/data/soundcloud/raw")
    assert s.rpm == 60


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("SC_LAKE_ROOT", str(tmp_path))
    monkeypatch.setenv("SC_LAKE_RPM", "20")
    s = load_settings()
    assert s.data_root == tmp_path
    assert s.db_path == tmp_path / "sc_lake.db"
    assert s.rpm == 20


def test_explicit_args_beat_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SC_LAKE_RPM", "20")
    s = load_settings(data_root=tmp_path, rpm=5)
    assert s.data_root == tmp_path
    assert s.rpm == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/test_soundcloud_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'soundcloud'`.

- [ ] **Step 3: Write minimal implementation**

```python
# soundcloud/__init__.py
"""SoundCloud data lake — anon client_id fetch primitives + normalized graph store."""
```

```python
# soundcloud/config.py
"""SoundCloud data-lake settings + env resolution."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path("/mnt/storage/data/soundcloud")


@dataclass(frozen=True)
class SoundCloudSettings:
    data_root: Path
    db_path: Path
    raw_root: Path
    rpm: int


def load_settings(
    *,
    data_root: Path | None = None,
    rpm: int | None = None,
) -> SoundCloudSettings:
    root = data_root or Path(os.environ.get("SC_LAKE_ROOT", str(DEFAULT_ROOT)))
    resolved_rpm = rpm if rpm is not None else int(os.environ.get("SC_LAKE_RPM", "60"))
    return SoundCloudSettings(
        data_root=root,
        db_path=root / "sc_lake.db",
        raw_root=root / "raw",
        rpm=resolved_rpm,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/test_soundcloud_config.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Add data dir to .gitignore**

Append to `.gitignore`:
```
soundcloud/data/
```

- [ ] **Step 6: Commit**

```bash
git add soundcloud/__init__.py soundcloud/config.py tests/test_soundcloud_config.py .gitignore
git commit -m "feat(soundcloud): module scaffold + settings/env resolution"
```

---

### Task 2: Frozen records

**Files:**
- Create: `soundcloud/records.py`
- Create: `tests/test_soundcloud_records.py`

**Interfaces:**
- Produces frozen dataclasses:
  - `ScUser(sc_user_id: int, permalink: str, username: str, followers_count: int | None, followings_count: int | None, verified: bool, city: str | None, country: str | None, description: str | None, fetched_at: str, raw_ref: str)`
  - `ScTrack(sc_track_id: int, title: str, owner_sc_user_id: int, genre: str | None, tag_list: str | None, duration_ms: int | None, playback_count: int | None, likes_count: int | None, created_at: str | None, permalink: str | None, fetched_at: str, raw_ref: str)`
  - `ScPlaylist(sc_playlist_id: int, title: str, owner_sc_user_id: int, track_count: int | None, is_album: bool, fetched_at: str, raw_ref: str)`
  - `CrawlPolicy(seed_user_ids: tuple[int, ...], depth: int = 1, entity_types: frozenset[str] = DEFAULT_ENTITIES, rpm: int = 60)`
  - `DEFAULT_ENTITIES = frozenset({"likes", "reposts", "playlists", "tracks", "followings", "followers"})`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_soundcloud_records.py
"""Tests for soundcloud.records dataclasses."""
from __future__ import annotations

import dataclasses

import pytest

from soundcloud.records import CrawlPolicy, DEFAULT_ENTITIES, ScTrack, ScUser


def test_records_are_frozen():
    u = ScUser(
        sc_user_id=1, permalink="p", username="u", followers_count=None,
        followings_count=None, verified=False, city=None, country=None,
        description=None, fetched_at="2026-07-18T00:00:00Z", raw_ref="raw/users/1/x.jsonl",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        u.username = "x"  # type: ignore[misc]


def test_crawl_policy_defaults():
    p = CrawlPolicy(seed_user_ids=(327506308,))
    assert p.depth == 1
    assert p.entity_types == DEFAULT_ENTITIES
    assert "followers" in p.entity_types


def test_track_holds_owner_id():
    t = ScTrack(
        sc_track_id=9, title="T", owner_sc_user_id=1, genre="house", tag_list=None,
        duration_ms=1000, playback_count=None, likes_count=None, created_at=None,
        permalink=None, fetched_at="2026-07-18T00:00:00Z", raw_ref="r",
    )
    assert t.owner_sc_user_id == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/test_soundcloud_records.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'soundcloud.records'`.

- [ ] **Step 3: Write minimal implementation**

```python
# soundcloud/records.py
"""Frozen records for the SoundCloud data lake (nodes, edges, crawl policy)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScUser:
    sc_user_id: int
    permalink: str
    username: str
    followers_count: int | None
    followings_count: int | None
    verified: bool
    city: str | None
    country: str | None
    description: str | None
    fetched_at: str
    raw_ref: str


@dataclass(frozen=True)
class ScTrack:
    sc_track_id: int
    title: str
    owner_sc_user_id: int
    genre: str | None
    tag_list: str | None
    duration_ms: int | None
    playback_count: int | None
    likes_count: int | None
    created_at: str | None
    permalink: str | None
    fetched_at: str
    raw_ref: str


@dataclass(frozen=True)
class ScPlaylist:
    sc_playlist_id: int
    title: str
    owner_sc_user_id: int
    track_count: int | None
    is_album: bool
    fetched_at: str
    raw_ref: str


DEFAULT_ENTITIES = frozenset(
    {"likes", "reposts", "playlists", "tracks", "followings", "followers"}
)


@dataclass(frozen=True)
class CrawlPolicy:
    seed_user_ids: tuple[int, ...]
    depth: int = 1
    entity_types: frozenset[str] = DEFAULT_ENTITIES
    rpm: int = 60
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/test_soundcloud_records.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add soundcloud/records.py tests/test_soundcloud_records.py
git commit -m "feat(soundcloud): frozen node/edge records + CrawlPolicy"
```

---

### Task 3: Schema + store connection/init

**Files:**
- Create: `soundcloud/schema.sql`
- Create: `soundcloud/store.py`
- Create: `tests/test_soundcloud_store_init.py`

**Interfaces:**
- Produces: `SCHEMA_PATH: Path`; `connect(db_path: Path) -> sqlite3.Connection`; `init_db(db_path: Path) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_soundcloud_store_init.py
"""Tests for soundcloud.store schema init."""
from __future__ import annotations

from pathlib import Path

from soundcloud.store import connect, init_db


def _tables(db_path: Path) -> set[str]:
    with connect(db_path) as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {r["name"] for r in rows}


def test_init_creates_all_tables(tmp_path):
    db = tmp_path / "sc_lake.db"
    init_db(db)
    names = _tables(db)
    assert {
        "sc_users", "sc_tracks", "sc_playlists",
        "sc_likes", "sc_reposts", "sc_follows", "sc_playlist_tracks",
        "sc_recording_map", "crawl_checkpoints",
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
```

Note: `connect` must be usable as a context manager here. Implement it as a `@contextmanager`.

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/test_soundcloud_store_init.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'soundcloud.store'`.

- [ ] **Step 3: Write the schema**

```sql
-- soundcloud/schema.sql
-- SoundCloud data lake (pi-storage: /mnt/storage/data/soundcloud/sc_lake.db).
-- Off-canonical: NOT music_database.db, NOT on the alignment DAG.
-- Rebuildable from raw/{entity}/{id}/{fetched_at}.jsonl.

CREATE TABLE IF NOT EXISTS sc_users (
    sc_user_id        INTEGER PRIMARY KEY,
    permalink         TEXT,
    username          TEXT,
    followers_count   INTEGER,
    followings_count  INTEGER,
    verified          INTEGER NOT NULL DEFAULT 0,
    city              TEXT,
    country           TEXT,
    description       TEXT,
    fetched_at        TEXT NOT NULL,
    raw_ref           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sc_tracks (
    sc_track_id        INTEGER PRIMARY KEY,
    title              TEXT,
    owner_sc_user_id   INTEGER,
    genre              TEXT,
    tag_list           TEXT,
    duration_ms        INTEGER,
    playback_count     INTEGER,
    likes_count        INTEGER,
    created_at         TEXT,
    permalink          TEXT,
    fetched_at         TEXT NOT NULL,
    raw_ref            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sc_playlists (
    sc_playlist_id     INTEGER PRIMARY KEY,
    title              TEXT,
    owner_sc_user_id   INTEGER,
    track_count        INTEGER,
    is_album           INTEGER NOT NULL DEFAULT 0,
    fetched_at         TEXT NOT NULL,
    raw_ref            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sc_likes (
    sc_user_id   INTEGER NOT NULL,
    sc_track_id  INTEGER NOT NULL,
    created_at   TEXT,
    PRIMARY KEY (sc_user_id, sc_track_id)
);

CREATE TABLE IF NOT EXISTS sc_reposts (
    sc_user_id    INTEGER NOT NULL,
    item_kind     TEXT NOT NULL,          -- 'track' | 'playlist'
    item_id       INTEGER NOT NULL,
    created_at    TEXT,
    PRIMARY KEY (sc_user_id, item_kind, item_id)
);

CREATE TABLE IF NOT EXISTS sc_follows (
    follower_sc_user_id  INTEGER NOT NULL,
    followee_sc_user_id  INTEGER NOT NULL,
    PRIMARY KEY (follower_sc_user_id, followee_sc_user_id)
);

CREATE TABLE IF NOT EXISTS sc_playlist_tracks (
    sc_playlist_id  INTEGER NOT NULL,
    sc_track_id     INTEGER NOT NULL,
    position        INTEGER NOT NULL,
    PRIMARY KEY (sc_playlist_id, sc_track_id)
);

-- Identity seam (stub): populated by the future audio-seed / discovery consumers.
CREATE TABLE IF NOT EXISTS sc_recording_map (
    sc_track_id   INTEGER PRIMARY KEY,
    recording_id  TEXT,
    method        TEXT,
    confidence    REAL
);

CREATE TABLE IF NOT EXISTS crawl_checkpoints (
    sc_user_id   INTEGER NOT NULL,
    phase        TEXT NOT NULL,           -- entity type, e.g. 'likes'
    done         INTEGER NOT NULL DEFAULT 0,
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (sc_user_id, phase)
);

CREATE INDEX IF NOT EXISTS idx_sc_tracks_owner ON sc_tracks(owner_sc_user_id);
CREATE INDEX IF NOT EXISTS idx_sc_likes_track ON sc_likes(sc_track_id);
CREATE INDEX IF NOT EXISTS idx_sc_follows_followee ON sc_follows(followee_sc_user_id);
```

- [ ] **Step 4: Write minimal store connection/init**

```python
# soundcloud/store.py
"""SoundCloud data-lake SQLite store: connection, schema init, parse + upsert."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/test_soundcloud_store_init.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add soundcloud/schema.sql soundcloud/store.py tests/test_soundcloud_store_init.py
git commit -m "feat(soundcloud): sc_lake.db schema + connection/init"
```

---

### Task 4: Store parse + upsert (nodes and edges)

**Files:**
- Modify: `soundcloud/store.py`
- Create: `tests/test_soundcloud_store_upsert.py`

**Interfaces:**
- Consumes: `ScUser`, `ScTrack`, `ScPlaylist` from `soundcloud.records`; `connect` from this module.
- Produces:
  - Parsers (raw dict + `fetched_at` + `raw_ref` → record): `parse_user`, `parse_track`, `parse_playlist`.
  - Upserts (conn, record → None): `upsert_user`, `upsert_track`, `upsert_playlist`.
  - Edge writers: `add_like(conn, sc_user_id, sc_track_id, created_at)`, `add_repost(conn, sc_user_id, item_kind, item_id, created_at)`, `add_follow(conn, follower_id, followee_id)`, `add_playlist_track(conn, playlist_id, track_id, position)`.
  - Checkpoint: `mark_done(conn, sc_user_id, phase, now)`, `is_done(conn, sc_user_id, phase) -> bool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_soundcloud_store_upsert.py
"""Tests for soundcloud.store parse + upsert idempotency."""
from __future__ import annotations

from soundcloud import store
from soundcloud.records import ScTrack, ScUser

RAW_USER = {
    "id": 327506308, "permalink": "user-327506308", "username": "John",
    "followers_count": 12, "followings_count": 30, "verified": False,
    "city": "NYC", "country_code": "US", "description": "bio",
}
RAW_TRACK = {
    "id": 555, "title": "Heavy Thunder", "user": {"id": 42},
    "genre": "bass", "tag_list": "trap", "duration": 210000,
    "playback_count": 9, "likes_count": 3, "created_at": "2020/01/01 00:00:00 +0000",
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/test_soundcloud_store_upsert.py -v`
Expected: FAIL — `AttributeError: module 'soundcloud.store' has no attribute 'parse_user'`.

- [ ] **Step 3: Add parsers, upserts, edge writers, checkpoints to `soundcloud/store.py`**

Append to `soundcloud/store.py` (and add `from soundcloud.records import ScPlaylist, ScTrack, ScUser` to the imports):

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/test_soundcloud_store_upsert.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add soundcloud/store.py tests/test_soundcloud_store_upsert.py
git commit -m "feat(soundcloud): parse + idempotent upsert for nodes/edges/checkpoints"
```

---

### Task 5: Client (generalize) + personalization shim

**Files:**
- Create: `soundcloud/client.py`
- Modify: `personalization/soundcloud_client.py` (→ re-export shim)
- Create: `tests/test_soundcloud_client.py`
- Create: `tests/fixtures/sc_resolve_user.json`

**Interfaces:**
- Consumes: `httpx`.
- Produces (moved/generalized from `personalization/soundcloud_client.py`): `RateLimiter`, `sc_client()`, `rl_get()`, `extract_client_id()`, `next_url()`, `resolve(client, rl, client_id, url) -> dict` (generalized — returns any kind), constants `SC_API`, `SKIP_STATUS_CODES`, `USER_AGENT`.
- The shim keeps `resolve_track(...)` (asserts `kind == 'track'`) and re-exports the rest so `personalization` imports are unbroken.

- [ ] **Step 1: Write the fixture**

```json
// tests/fixtures/sc_resolve_user.json
{"kind": "user", "id": 327506308, "permalink": "user-327506308", "username": "John"}
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_soundcloud_client.py
"""Tests for soundcloud.client transport helpers (no live network)."""
from __future__ import annotations

import json
from pathlib import Path

from soundcloud import client as sc

FIX = Path(__file__).resolve().parent / "fixtures"


class FakeResp:
    def __init__(self, payload, status=200, text=None):
        self._payload = payload
        self.status_code = status
        self.text = text if text is not None else json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError("err", request=None, response=None)


def test_next_url_joins_client_id_with_amp():
    assert sc.next_url("https://x/y?a=1", "CID") == "https://x/y?a=1&client_id=CID"
    assert sc.next_url("https://x/y", "CID") == "https://x/y?client_id=CID"


def test_extract_client_id_finds_in_script(monkeypatch):
    home = '<script src="https://a-cdn.sndcdn.com/assets/app.js"></script>'
    js = 'foo,client_id:"abcdef0123456789abcd",bar'
    calls = {"n": 0}

    def fake_rl_get(client, rl, url, **kw):
        calls["n"] += 1
        return FakeResp(None, text=home if url == sc.SC_HOME else js)

    monkeypatch.setattr(sc, "rl_get", fake_rl_get)
    cid = sc.extract_client_id(object(), object())
    assert cid == "abcdef0123456789abcd"


def test_resolve_returns_any_kind(monkeypatch):
    payload = json.loads((FIX / "sc_resolve_user.json").read_text())
    monkeypatch.setattr(sc, "rl_get", lambda *a, **k: FakeResp(payload))
    out = sc.resolve(object(), object(), "CID", "https://soundcloud.com/user-327506308")
    assert out["kind"] == "user"
    assert out["id"] == 327506308


def test_shim_reexports_and_resolve_track_asserts():
    from personalization import soundcloud_client as shim
    assert shim.RateLimiter is sc.RateLimiter
    assert hasattr(shim, "resolve_track")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/test_soundcloud_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'soundcloud.client'`.

- [ ] **Step 4: Create `soundcloud/client.py`** by moving the body of `personalization/soundcloud_client.py` verbatim, then generalizing `resolve_track` → `resolve`:

```python
# soundcloud/client.py
"""SoundCloud api-v2 client helpers — anon client_id, rate-limited, read-only."""
from __future__ import annotations

import logging
import re
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SC_HOME = "https://soundcloud.com/"
SC_API = "https://api-v2.soundcloud.com"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
SCRIPT_RE = re.compile(r'https://[a-z0-9\-\.]+\.sndcdn\.com/assets/[^"\']+\.js')
CLIENT_ID_RE = re.compile(r'client_id\s*[:=]\s*"([A-Za-z0-9]{20,40})"')
SC_CLIENT_LIMITS = httpx.Limits(max_keepalive_connections=0, max_connections=1)
SKIP_STATUS_CODES = frozenset({401, 403, 404, 429, 500, 502, 503})


class RateLimiter:
    def __init__(self, rpm: int) -> None:
        self._interval = 60.0 / max(rpm, 1)
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last = time.monotonic()


def sc_client(**kwargs: Any) -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=30,
        follow_redirects=True,
        limits=SC_CLIENT_LIMITS,
        **kwargs,
    )


def rl_get(client: httpx.Client, rl: RateLimiter, url: str, *, max_retries: int = 3, **kwargs: Any) -> httpx.Response:
    last_err: httpx.TransportError | None = None
    for attempt in range(max_retries):
        try:
            rl.wait()
            resp = client.get(url, **kwargs)
            resp.raise_for_status()
            return resp
        except httpx.TransportError as e:
            last_err = e
            if attempt + 1 >= max_retries:
                raise
            time.sleep(0.5 * (2**attempt))
    assert last_err is not None
    raise last_err


def extract_client_id(client: httpx.Client, rl: RateLimiter) -> str:
    html = rl_get(client, rl, SC_HOME).text
    for script_url in reversed(SCRIPT_RE.findall(html)):
        try:
            js = rl_get(client, rl, script_url).text
        except httpx.HTTPError:
            continue
        m = CLIENT_ID_RE.search(js)
        if m:
            return m.group(1)
    raise RuntimeError("SoundCloud client_id not found in homepage scripts")


def resolve(client: httpx.Client, rl: RateLimiter, client_id: str, url: str) -> dict[str, Any]:
    """Resolve any SoundCloud URL to its api-v2 entity (user/track/playlist)."""
    resp = rl_get(client, rl, f"{SC_API}/resolve", params={"url": url, "client_id": client_id})
    return resp.json()


def next_url(nxt: str, client_id: str) -> str:
    sep = "&" if "?" in nxt else "?"
    return f"{nxt}{sep}client_id={client_id}"
```

- [ ] **Step 5: Replace `personalization/soundcloud_client.py` with a shim**

```python
# personalization/soundcloud_client.py
"""Back-compat shim — canonical client now lives in soundcloud.client."""
from __future__ import annotations

from typing import Any

from soundcloud.client import (  # noqa: F401
    CLIENT_ID_RE,
    SC_API,
    SC_HOME,
    SCRIPT_RE,
    SKIP_STATUS_CODES,
    USER_AGENT,
    RateLimiter,
    extract_client_id,
    next_url,
    rl_get,
    resolve,
    sc_client,
)


def resolve_track(client: Any, rl: RateLimiter, client_id: str, url: str) -> dict[str, Any]:
    """Resolve a URL and assert it is a track (legacy personalization behavior)."""
    data = resolve(client, rl, client_id, url)
    if data.get("kind") != "track":
        raise ValueError(f"expected track, got kind={data.get('kind')}")
    return data
```

- [ ] **Step 6: Run tests (new + personalization regression)**

Run: `venvs/audio/bin/python -m pytest tests/test_soundcloud_client.py -v`
Expected: PASS (4 passed).
Run: `venvs/audio/bin/python -m pytest tests/ -k personalization -q`
Expected: existing personalization tests still PASS (no import breakage).

- [ ] **Step 7: Commit**

```bash
git add soundcloud/client.py personalization/soundcloud_client.py tests/test_soundcloud_client.py tests/fixtures/sc_resolve_user.json
git commit -m "feat(soundcloud): generalize client (resolve any kind); personalization shim"
```

---

### Task 6: Fetch primitives (paged endpoints)

**Files:**
- Create: `soundcloud/fetch.py`
- Create: `tests/test_soundcloud_fetch.py`

**Interfaces:**
- Consumes: `RateLimiter`, `SC_API`, `SKIP_STATUS_CODES`, `next_url`, `rl_get` from `soundcloud.client`.
- Produces generators yielding raw item dicts (paginated), each taking `(client, rl, cid, uid_or_id)`:
  - `user(client, rl, cid, uid) -> dict` (single object, not a generator)
  - `user_likes`, `user_reposts`, `user_playlists`, `user_tracks`, `user_followings`, `user_followers` → `Iterator[dict]`
  - `track(client, rl, cid, tid) -> dict`, `playlist(client, rl, cid, pid) -> dict`
- Internal helper: `_paged(client, rl, cid, path, params=None) -> Iterator[dict]` — follows `next_href`, appends `client_id` via `next_url`, stops on empty collection; a page whose GET raises an `httpx.HTTPStatusError` with a `SKIP_STATUS_CODES` status ends iteration rather than propagating.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_soundcloud_fetch.py
"""Tests for soundcloud.fetch pagination + endpoint routing (fake client)."""
from __future__ import annotations

from soundcloud import fetch


class FakeResp:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


class FakeClient:
    """Serves queued responses; records requested URLs."""
    def __init__(self, pages):
        self._pages = list(pages)
        self.urls: list[str] = []

    def get(self, url, params=None, **kw):
        self.urls.append(url)
        return FakeResp(self._pages.pop(0))


class NullRL:
    def wait(self):
        return None


def test_paged_follows_next_href_until_empty():
    pages = [
        {"collection": [{"id": 1}, {"id": 2}], "next_href": "https://api-v2.soundcloud.com/users/7/likes?cursor=abc"},
        {"collection": [{"id": 3}], "next_href": None},
    ]
    client = FakeClient(pages)
    items = list(fetch.user_likes(client, NullRL(), "CID", 7))
    assert [i["id"] for i in items] == [1, 2, 3]
    # first call hits the likes path; second call follows next_href with client_id appended
    assert "/users/7/likes" in client.urls[0]
    assert "client_id=CID" in client.urls[1]


def test_user_single_object():
    client = FakeClient([{"id": 7, "username": "John"}])
    out = fetch.user(client, NullRL(), "CID", 7)
    assert out["username"] == "John"


def test_paged_stops_on_skip_status():
    import httpx

    class Boom(FakeClient):
        def get(self, url, params=None, **kw):
            self.urls.append(url)
            resp = FakeResp({})
            resp.status_code = 404

            def raise_for_status():
                raise httpx.HTTPStatusError("nf", request=httpx.Request("GET", url),
                                            response=httpx.Response(404, request=httpx.Request("GET", url)))
            resp.raise_for_status = raise_for_status
            return resp

    items = list(fetch.user_tracks(Boom([{}]), NullRL(), "CID", 7))
    assert items == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/test_soundcloud_fetch.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'soundcloud.fetch'`.

- [ ] **Step 3: Write minimal implementation**

```python
# soundcloud/fetch.py
"""SoundCloud api-v2 endpoint primitives — the reusable data-lake fetch surface.

Every function is read-only and does NO parsing/DB work; it yields raw dicts.
Collections paginate via linked_partitioning + next_href.
"""
from __future__ import annotations

from typing import Any, Iterator

import httpx

from soundcloud.client import SC_API, SKIP_STATUS_CODES, next_url, rl_get

_PAGE = 200


def _get_json(client: Any, rl: Any, url: str, params: dict | None = None) -> dict:
    return rl_get(client, rl, url, params=params).json()


def _paged(client: Any, rl: Any, cid: str, path: str, params: dict | None = None) -> Iterator[dict]:
    base = {"client_id": cid, "limit": _PAGE, "linked_partitioning": 1}
    if params:
        base.update(params)
    url = f"{SC_API}{path}"
    first = True
    while url:
        try:
            data = _get_json(client, rl, url, params=base if first else None)
        except httpx.HTTPStatusError as e:
            if e.response is not None and e.response.status_code in SKIP_STATUS_CODES:
                return
            raise
        first = False
        for item in data.get("collection", []):
            yield item
        nxt = data.get("next_href")
        url = next_url(nxt, cid) if nxt else ""


def user(client: Any, rl: Any, cid: str, uid: int) -> dict:
    return _get_json(client, rl, f"{SC_API}/users/{uid}", params={"client_id": cid})


def track(client: Any, rl: Any, cid: str, tid: int) -> dict:
    return _get_json(client, rl, f"{SC_API}/tracks/{tid}", params={"client_id": cid})


def playlist(client: Any, rl: Any, cid: str, pid: int) -> dict:
    return _get_json(client, rl, f"{SC_API}/playlists/{pid}", params={"client_id": cid})


def user_likes(client: Any, rl: Any, cid: str, uid: int) -> Iterator[dict]:
    return _paged(client, rl, cid, f"/users/{uid}/likes")


def user_reposts(client: Any, rl: Any, cid: str, uid: int) -> Iterator[dict]:
    return _paged(client, rl, cid, f"/stream/users/{uid}/reposts")


def user_playlists(client: Any, rl: Any, cid: str, uid: int) -> Iterator[dict]:
    return _paged(client, rl, cid, f"/users/{uid}/playlists")


def user_tracks(client: Any, rl: Any, cid: str, uid: int) -> Iterator[dict]:
    return _paged(client, rl, cid, f"/users/{uid}/tracks")


def user_followings(client: Any, rl: Any, cid: str, uid: int) -> Iterator[dict]:
    return _paged(client, rl, cid, f"/users/{uid}/followings")


def user_followers(client: Any, rl: Any, cid: str, uid: int) -> Iterator[dict]:
    return _paged(client, rl, cid, f"/users/{uid}/followers")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/test_soundcloud_fetch.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add soundcloud/fetch.py tests/test_soundcloud_fetch.py
git commit -m "feat(soundcloud): paged api-v2 fetch primitives (data-lake surface)"
```

**Live-shape note (do during Task 8 dry run, not in tests):** the exact JSON of `likes`/`reposts` items (wrapper vs bare object) is verified with a single live probe before wiring `crawl`. The `crawl` extractors in Task 7 handle both `{"track": {...}}` wrappers and bare `{...}` track objects.

---

### Task 7: Raw lake writer

**Files:**
- Create: `soundcloud/rawlake.py`
- Create: `tests/test_soundcloud_rawlake.py`

**Interfaces:**
- Produces: `write_snapshot(raw_root: Path, entity: str, entity_id: int, fetched_at: str, records: list[dict]) -> str` — writes `{raw_root}/{entity}/{entity_id}/{safe_fetched_at}.jsonl` (one JSON object per line), returns the path **relative to `raw_root`** to store as `raw_ref`. `safe_fetched_at` replaces `:` with `-` for filesystem safety.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_soundcloud_rawlake.py
"""Tests for soundcloud.rawlake append-only snapshots."""
from __future__ import annotations

import json

from soundcloud import rawlake


def test_write_snapshot_returns_relref_and_writes_jsonl(tmp_path):
    ref = rawlake.write_snapshot(tmp_path, "likes", 7, "2026-07-18T00:00:00Z",
                                 [{"id": 1}, {"id": 2}])
    assert ref == "likes/7/2026-07-18T00-00-00Z.jsonl"
    lines = (tmp_path / ref).read_text().splitlines()
    assert [json.loads(x)["id"] for x in lines] == [1, 2]


def test_empty_records_still_writes_file(tmp_path):
    ref = rawlake.write_snapshot(tmp_path, "user", 7, "2026-07-18T00:00:00Z", [])
    assert (tmp_path / ref).exists()
    assert (tmp_path / ref).read_text() == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/test_soundcloud_rawlake.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'soundcloud.rawlake'`.

- [ ] **Step 3: Write minimal implementation**

```python
# soundcloud/rawlake.py
"""Append-only raw JSONL snapshots — the audit/reprocess layer under the lake."""
from __future__ import annotations

import json
from pathlib import Path


def write_snapshot(raw_root: Path, entity: str, entity_id: int, fetched_at: str,
                   records: list[dict]) -> str:
    safe = fetched_at.replace(":", "-")
    rel = f"{entity}/{entity_id}/{safe}.jsonl"
    path = raw_root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, default=str) + "\n")
    return rel
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/test_soundcloud_rawlake.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add soundcloud/rawlake.py tests/test_soundcloud_rawlake.py
git commit -m "feat(soundcloud): append-only raw JSONL snapshot writer"
```

---

### Task 8: Crawl driver (depth-1 frontier)

**Files:**
- Create: `soundcloud/crawl.py`
- Create: `tests/test_soundcloud_crawl.py`

**Interfaces:**
- Consumes: `CrawlPolicy` from `records`; `fetch` module; `store` module; `rawlake.write_snapshot`; `SoundCloudSettings`.
- Produces: `crawl(conn, settings, policy, client, rl, cid, now, fetch_mod=fetch) -> dict` returning counts `{"users": n, "tracks": n, "likes": n, "reposts": n, "playlists": n, "follows": n}`. `fetch_mod` is injectable for tests. For each seed user it: fetches+upserts the user; for each enabled entity phase not already `is_done`, pulls the collection, writes a raw snapshot, upserts nodes + edges, then `mark_done`. Depth-1: followings/followers become `sc_users` node rows (parsed from the returned user objects) + `sc_follows` edges, NOT recursed. `depth > 1` enqueues neighbor ids (still each processed by the same per-user routine); the frontier dedups via a visited set seeded from users already fully checkpointed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_soundcloud_crawl.py
"""Tests for soundcloud.crawl depth-1 frontier via an injected fake fetch."""
from __future__ import annotations

from soundcloud import crawl, store
from soundcloud.config import load_settings
from soundcloud.records import CrawlPolicy


class FakeFetch:
    """Stand-in for soundcloud.fetch with canned data for user 7."""
    def user(self, c, rl, cid, uid):
        return {"id": uid, "permalink": f"u{uid}", "username": f"U{uid}"}

    def user_likes(self, c, rl, cid, uid):
        yield {"id": 100, "title": "L", "user": {"id": 42}, "created_at": "2020/01/01 00:00:00 +0000"}

    def user_reposts(self, c, rl, cid, uid):
        yield {"type": "track-repost", "track": {"id": 200, "title": "R", "user": {"id": 43}},
               "created_at": "2020/02/02 00:00:00 +0000"}

    def user_playlists(self, c, rl, cid, uid):
        yield {"id": 300, "title": "P", "user": {"id": uid}, "track_count": 1,
               "tracks": [{"id": 301, "title": "PT", "user": {"id": 44}}]}

    def user_tracks(self, c, rl, cid, uid):
        yield {"id": 400, "title": "Own", "user": {"id": uid}}

    def user_followings(self, c, rl, cid, uid):
        yield {"id": 500, "permalink": "u500", "username": "U500"}

    def user_followers(self, c, rl, cid, uid):
        yield {"id": 600, "permalink": "u600", "username": "U600"}


def test_depth1_pulls_seed_collections_no_recurse(tmp_path, monkeypatch):
    monkeypatch.setenv("SC_LAKE_ROOT", str(tmp_path))
    settings = load_settings()
    store.init_db(settings.db_path)
    policy = CrawlPolicy(seed_user_ids=(7,), depth=1)
    with store.connect(settings.db_path) as conn:
        counts = crawl.crawl(conn, settings, policy, object(), object(), "CID",
                             "2026-07-18T00:00:00Z", fetch_mod=FakeFetch())
        conn.commit()
        n_users = conn.execute("SELECT COUNT(*) FROM sc_users").fetchone()[0]
        n_likes = conn.execute("SELECT COUNT(*) FROM sc_likes").fetchone()[0]
        n_follows = conn.execute("SELECT COUNT(*) FROM sc_follows").fetchone()[0]
        # neighbor 500 is a node but was NOT itself crawled (no likes for 500)
        n_likes_of_500 = conn.execute("SELECT COUNT(*) FROM sc_likes WHERE sc_user_id=500").fetchone()[0]
    assert n_likes == 1
    assert n_follows == 2                     # following 500 + follower 600
    assert n_users >= 3                       # seed 7 + 500 + 600 (+ owners)
    assert n_likes_of_500 == 0
    assert counts["likes"] == 1


def test_resume_skips_completed_phase(tmp_path, monkeypatch):
    monkeypatch.setenv("SC_LAKE_ROOT", str(tmp_path))
    settings = load_settings()
    store.init_db(settings.db_path)
    policy = CrawlPolicy(seed_user_ids=(7,), depth=1)
    with store.connect(settings.db_path) as conn:
        crawl.crawl(conn, settings, policy, object(), object(), "CID",
                    "2026-07-18T00:00:00Z", fetch_mod=FakeFetch())
        conn.commit()

        class Boom(FakeFetch):
            def user_likes(self, c, rl, cid, uid):
                raise AssertionError("likes phase should have been skipped on resume")

        # Should not raise — likes already checkpointed done.
        crawl.crawl(conn, settings, policy, object(), object(), "CID",
                    "2026-07-18T00:00:01Z", fetch_mod=Boom())
        conn.commit()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/test_soundcloud_crawl.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'soundcloud.crawl'`.

- [ ] **Step 3: Write minimal implementation**

```python
# soundcloud/crawl.py
"""Depth-1 (parameterizable) frontier crawler over the fetch primitives."""
from __future__ import annotations

from collections import deque
from typing import Any

from soundcloud import fetch as _fetch
from soundcloud import rawlake, store
from soundcloud.config import SoundCloudSettings
from soundcloud.records import CrawlPolicy


def _repost_item(raw: dict) -> tuple[str, dict] | None:
    if "track" in raw and raw["track"]:
        return "track", raw["track"]
    if "playlist" in raw and raw["playlist"]:
        return "playlist", raw["playlist"]
    return None


def _crawl_user(conn, settings: SoundCloudSettings, uid: int, entities: frozenset[str],
                client: Any, rl: Any, cid: str, now: str, fx: Any, counts: dict) -> list[int]:
    """Process one user's own collections. Returns neighbor user ids discovered."""
    raw_root = settings.raw_root
    u_raw = fx.user(client, rl, cid, uid)
    ref = rawlake.write_snapshot(raw_root, "user", uid, now, [u_raw])
    store.upsert_user(conn, store.parse_user(u_raw, fetched_at=now, raw_ref=ref))
    counts["users"] += 1
    neighbors: list[int] = []

    if "likes" in entities and not store.is_done(conn, uid, "likes"):
        items = list(fx.user_likes(client, rl, cid, uid))
        ref = rawlake.write_snapshot(raw_root, "likes", uid, now, items)
        for it in items:
            tr = it.get("track", it)
            if not tr or "id" not in tr:
                continue
            store.upsert_track(conn, store.parse_track(tr, fetched_at=now, raw_ref=ref))
            store.add_like(conn, uid, int(tr["id"]), it.get("created_at"))
            counts["tracks"] += 1
            counts["likes"] += 1
        store.mark_done(conn, uid, "likes", now)

    if "reposts" in entities and not store.is_done(conn, uid, "reposts"):
        items = list(fx.user_reposts(client, rl, cid, uid))
        ref = rawlake.write_snapshot(raw_root, "reposts", uid, now, items)
        for it in items:
            parsed = _repost_item(it)
            if not parsed:
                continue
            kind, obj = parsed
            if kind == "track" and "id" in obj:
                store.upsert_track(conn, store.parse_track(obj, fetched_at=now, raw_ref=ref))
            store.add_repost(conn, uid, kind, int(obj["id"]), it.get("created_at"))
            counts["reposts"] += 1
        store.mark_done(conn, uid, "reposts", now)

    if "playlists" in entities and not store.is_done(conn, uid, "playlists"):
        items = list(fx.user_playlists(client, rl, cid, uid))
        ref = rawlake.write_snapshot(raw_root, "playlists", uid, now, items)
        for pl in items:
            store.upsert_playlist(conn, store.parse_playlist(pl, fetched_at=now, raw_ref=ref))
            counts["playlists"] += 1
            for pos, tr in enumerate(pl.get("tracks", []) or []):
                if "id" not in tr:
                    continue
                store.upsert_track(conn, store.parse_track(tr, fetched_at=now, raw_ref=ref))
                store.add_playlist_track(conn, int(pl["id"]), int(tr["id"]), pos)
        store.mark_done(conn, uid, "playlists", now)

    if "tracks" in entities and not store.is_done(conn, uid, "tracks"):
        items = list(fx.user_tracks(client, rl, cid, uid))
        ref = rawlake.write_snapshot(raw_root, "tracks", uid, now, items)
        for tr in items:
            if "id" not in tr:
                continue
            store.upsert_track(conn, store.parse_track(tr, fetched_at=now, raw_ref=ref))
            counts["tracks"] += 1
        store.mark_done(conn, uid, "tracks", now)

    for phase, edge_from_seed in (("followings", True), ("followers", False)):
        if phase in entities and not store.is_done(conn, uid, phase):
            items = list(getattr(fx, f"user_{phase}")(client, rl, cid, uid))
            ref = rawlake.write_snapshot(raw_root, phase, uid, now, items)
            for nu in items:
                if "id" not in nu:
                    continue
                nid = int(nu["id"])
                store.upsert_user(conn, store.parse_user(nu, fetched_at=now, raw_ref=ref))
                counts["users"] += 1
                if edge_from_seed:
                    store.add_follow(conn, uid, nid)
                else:
                    store.add_follow(conn, nid, uid)
                counts["follows"] += 1
                neighbors.append(nid)
            store.mark_done(conn, uid, phase, now)

    return neighbors


def crawl(conn, settings: SoundCloudSettings, policy: CrawlPolicy, client: Any, rl: Any,
          cid: str, now: str, fetch_mod: Any = _fetch) -> dict:
    counts = {k: 0 for k in ("users", "tracks", "likes", "reposts", "playlists", "follows")}
    visited: set[int] = set()
    queue: deque[tuple[int, int]] = deque((uid, 0) for uid in policy.seed_user_ids)
    while queue:
        uid, depth = queue.popleft()
        if uid in visited:
            continue
        visited.add(uid)
        neighbors = _crawl_user(conn, settings, uid, policy.entity_types,
                                client, rl, cid, now, fetch_mod, counts)
        if depth + 1 < policy.depth:
            for nid in neighbors:
                if nid not in visited:
                    queue.append((nid, depth + 1))
    return counts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/test_soundcloud_crawl.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add soundcloud/crawl.py tests/test_soundcloud_crawl.py
git commit -m "feat(soundcloud): depth-1 frontier crawl driver + resume via checkpoints"
```

---

### Task 9: CLI (`main.py`)

**Files:**
- Create: `soundcloud/main.py`
- Create: `tests/test_soundcloud_main.py`

**Interfaces:**
- Consumes: `load_settings`; `store`; `crawl.crawl`; `client` (`sc_client`, `RateLimiter`, `extract_client_id`, `resolve`); `records.CrawlPolicy`.
- Produces CLI: `sync-user <profile_url_or_id>` (depth-1), `crawl --seed <id> [--depth N] [--rpm N]`, `stats`. `main(argv=None) -> int`. A `--now` hidden arg (ISO string) is injected in tests so core stays clock-free; when absent, `main` stamps `datetime.now(timezone.utc).isoformat()` at the edge.
- `sync-user`: if arg is all-digits → treat as id; else `resolve` the profile URL to an id. Build `CrawlPolicy(seed_user_ids=(id,), depth=1, rpm=settings.rpm)`, run `crawl`, print counts.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_soundcloud_main.py
"""Tests for soundcloud.main CLI dispatch (crawl injected via monkeypatch)."""
from __future__ import annotations

from soundcloud import main as sc_main
from soundcloud import store


def test_stats_runs_on_empty_db(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SC_LAKE_ROOT", str(tmp_path))
    rc = sc_main.main(["stats"])
    assert rc == 0
    assert "sc_users" in capsys.readouterr().out


def test_crawl_subcommand_invokes_driver(tmp_path, monkeypatch):
    monkeypatch.setenv("SC_LAKE_ROOT", str(tmp_path))
    seen = {}

    def fake_crawl(conn, settings, policy, client, rl, cid, now, fetch_mod=None):
        seen["seed"] = policy.seed_user_ids
        seen["depth"] = policy.depth
        return {"users": 1, "tracks": 0, "likes": 0, "reposts": 0, "playlists": 0, "follows": 0}

    monkeypatch.setattr(sc_main.crawl, "crawl", fake_crawl)
    monkeypatch.setattr(sc_main, "_bootstrap_client", lambda settings: (object(), object(), "CID"))
    rc = sc_main.main(["crawl", "--seed", "7", "--depth", "1", "--now", "2026-07-18T00:00:00Z"])
    assert rc == 0
    assert seen["seed"] == (7,)
    assert seen["depth"] == 1


def test_sync_user_resolves_profile_url(tmp_path, monkeypatch):
    monkeypatch.setenv("SC_LAKE_ROOT", str(tmp_path))
    monkeypatch.setattr(sc_main, "_bootstrap_client", lambda settings: (object(), object(), "CID"))
    monkeypatch.setattr(sc_main.client, "resolve", lambda c, rl, cid, url: {"kind": "user", "id": 327506308})
    captured = {}

    def fake_crawl(conn, settings, policy, *a, **k):
        captured["seed"] = policy.seed_user_ids
        return {"users": 1, "tracks": 0, "likes": 0, "reposts": 0, "playlists": 0, "follows": 0}

    monkeypatch.setattr(sc_main.crawl, "crawl", fake_crawl)
    rc = sc_main.main(["sync-user", "https://soundcloud.com/user-327506308",
                       "--now", "2026-07-18T00:00:00Z"])
    assert rc == 0
    assert captured["seed"] == (327506308,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/test_soundcloud_main.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'soundcloud.main'`.

- [ ] **Step 3: Write minimal implementation**

```python
# soundcloud/main.py
"""CLI — SoundCloud data lake: sync-user / crawl / stats."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from typing import Any

from soundcloud import client, crawl, store
from soundcloud.config import load_settings
from soundcloud.records import CrawlPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("soundcloud")

_TABLES = ("sc_users", "sc_tracks", "sc_playlists", "sc_likes", "sc_reposts",
           "sc_follows", "sc_playlist_tracks")


def _now(args: argparse.Namespace) -> str:
    return args.now or datetime.now(timezone.utc).isoformat()


def _bootstrap_client(settings) -> tuple[Any, Any, str]:
    c = client.sc_client()
    rl = client.RateLimiter(settings.rpm)
    cid = client.extract_client_id(c, rl)
    return c, rl, cid


def _run_crawl(policy: CrawlPolicy, now: str) -> int:
    settings = load_settings(rpm=policy.rpm)
    store.init_db(settings.db_path)
    c, rl, cid = _bootstrap_client(settings)
    with store.connect(settings.db_path) as conn:
        counts = crawl.crawl(conn, settings, policy, c, rl, cid, now)
        conn.commit()
    logger.info("crawl counts: %s", counts)
    return 0


def cmd_sync_user(args: argparse.Namespace) -> int:
    settings = load_settings()
    target = args.target
    if target.isdigit():
        uid = int(target)
    else:
        c, rl, cid = _bootstrap_client(settings)
        resolved = client.resolve(c, rl, cid, target)
        if resolved.get("kind") != "user":
            logger.error("resolved kind=%s, expected user", resolved.get("kind"))
            return 2
        uid = int(resolved["id"])
    return _run_crawl(CrawlPolicy(seed_user_ids=(uid,), depth=1, rpm=settings.rpm), _now(args))


def cmd_crawl(args: argparse.Namespace) -> int:
    settings = load_settings(rpm=args.rpm)
    policy = CrawlPolicy(seed_user_ids=(args.seed,), depth=args.depth, rpm=settings.rpm)
    return _run_crawl(policy, _now(args))


def cmd_stats(args: argparse.Namespace) -> int:
    settings = load_settings()
    store.init_db(settings.db_path)
    with store.connect(settings.db_path) as conn:
        for t in _TABLES:
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"{t:20s} {n}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SoundCloud data lake")
    p.add_argument("--now", default=None, help=argparse.SUPPRESS)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_sync = sub.add_parser("sync-user", help="Sync one user's public library (depth-1)")
    p_sync.add_argument("target", help="profile URL or numeric sc_user_id")
    p_sync.set_defaults(func=cmd_sync_user)

    p_crawl = sub.add_parser("crawl", help="Frontier crawl from a seed user id")
    p_crawl.add_argument("--seed", type=int, required=True)
    p_crawl.add_argument("--depth", type=int, default=1)
    p_crawl.add_argument("--rpm", type=int, default=None)
    p_crawl.set_defaults(func=cmd_crawl)

    p_stats = sub.add_parser("stats", help="Print node/edge coverage")
    p_stats.set_defaults(func=cmd_stats)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

Note: `--now` is a top-level arg so it precedes the subcommand (`main(["sync-user", url, "--now", ...])` works because argparse threads parent args; if the parser rejects post-subcommand placement, the tests pass `--now` after the subcommand and argparse attaches it to the subparser — add `--now` to each subparser instead if needed).

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/test_soundcloud_main.py -v`
Expected: PASS (3 passed). If `--now` placement fails, add `--now` to each subparser and re-run.

- [ ] **Step 5: Commit**

```bash
git add soundcloud/main.py tests/test_soundcloud_main.py
git commit -m "feat(soundcloud): CLI sync-user/crawl/stats"
```

---

### Task 10: First-look validation analysis

**Files:**
- Create: `soundcloud/analysis/__init__.py`
- Create: `soundcloud/analysis/first_look.py`
- Create: `tests/test_soundcloud_first_look.py`

**Interfaces:**
- Consumes: `store.connect`.
- Produces: `report(db_path: Path) -> dict` with keys `top_liked_owners` (list of `(owner_sc_user_id, like_count)`), `genre_distribution` (dict genre→count over liked tracks), `following_like_overlap` (count of owners of liked tracks who are also followed), `playlist_sizes` (list of track_counts). Plus `def main(argv=None) -> int` that prints the report.

- [ ] **Step 1: Write the failing test**

```python
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
        store.upsert_user(conn, ScUser(7, "me", "Me", None, None, False, None, None, None, FA, "r"))
        store.upsert_track(conn, ScTrack(100, "A", 42, "house", None, None, None, None, None, None, FA, "r"))
        store.upsert_track(conn, ScTrack(101, "B", 42, "house", None, None, None, None, None, None, FA, "r"))
        store.upsert_track(conn, ScTrack(102, "C", 99, "techno", None, None, None, None, None, None, FA, "r"))
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/test_soundcloud_first_look.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'soundcloud.analysis'`.

- [ ] **Step 3: Write minimal implementation**

```python
# soundcloud/analysis/__init__.py
"""SoundCloud data-lake analyses (validation + research first-looks)."""
```

```python
# soundcloud/analysis/first_look.py
"""First-look descriptive report over the SoundCloud lake — validation, not research."""
from __future__ import annotations

import argparse
from pathlib import Path

from soundcloud import store
from soundcloud.config import load_settings


def report(db_path: Path) -> dict:
    with store.connect(db_path) as conn:
        top = conn.execute(
            """
            SELECT t.owner_sc_user_id AS owner, COUNT(*) AS n
            FROM sc_likes l JOIN sc_tracks t ON t.sc_track_id = l.sc_track_id
            GROUP BY t.owner_sc_user_id ORDER BY n DESC, owner ASC
            """
        ).fetchall()
        genres = conn.execute(
            """
            SELECT t.genre AS g, COUNT(*) AS n
            FROM sc_likes l JOIN sc_tracks t ON t.sc_track_id = l.sc_track_id
            WHERE t.genre IS NOT NULL GROUP BY t.genre
            """
        ).fetchall()
        overlap = conn.execute(
            """
            SELECT COUNT(DISTINCT t.owner_sc_user_id)
            FROM sc_likes l JOIN sc_tracks t ON t.sc_track_id = l.sc_track_id
            JOIN sc_follows f ON f.followee_sc_user_id = t.owner_sc_user_id
            """
        ).fetchone()[0]
        sizes = conn.execute("SELECT track_count FROM sc_playlists").fetchall()
    return {
        "top_liked_owners": [(r["owner"], r["n"]) for r in top],
        "genre_distribution": {r["g"]: r["n"] for r in genres},
        "following_like_overlap": overlap,
        "playlist_sizes": [r["track_count"] for r in sizes],
    }


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="SoundCloud lake first-look").parse_args(argv)
    r = report(load_settings().db_path)
    print(f"top liked owners: {r['top_liked_owners'][:10]}")
    print(f"genre distribution: {r['genre_distribution']}")
    print(f"following↔like overlap (owners): {r['following_like_overlap']}")
    print(f"playlists: {len(r['playlist_sizes'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/test_soundcloud_first_look.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add soundcloud/analysis/ tests/test_soundcloud_first_look.py
git commit -m "feat(soundcloud): first-look validation report over the lake"
```

---

### Task 11: Module guide + repo wiring + full check

**Files:**
- Create: `soundcloud/CLAUDE.md`
- Modify: `CLAUDE.md` (root — add `soundcloud/` to the per-module index bullet list)

- [ ] **Step 1: Write `soundcloud/CLAUDE.md`**

```markdown
# soundcloud — SoundCloud data-lake substrate

Reusable anonymous-`client_id` fetch layer + normalized graph store. **Off the
alignment DAG, off `music_database.db`.** The general SoundCloud ingestion
primitive that `personalization/` and `lab/` consume; NOT a specific consumer.

- **Auth:** anon `client_id` only (`client.py`, generalized from
  `personalization/soundcloud_client.py`, which is now a re-export shim). Public
  data of any user; no OAuth, no `/me`, no private items/reposts stream.
- **Storage (pi-storage, canonical):** `SC_LAKE_ROOT` (default
  `/mnt/storage/data/soundcloud`) → `sc_lake.db` + `raw/{entity}/{id}/*.jsonl`.
  The package is code-only; Mac queries over SSH.
- **Layers:** `config` → `records` → `schema.sql`+`store` → `client` → `fetch`
  → `rawlake` → `crawl` (depth-1 frontier) → `main` (CLI) → `analysis/first_look`.

## Commands

```bash
venvs/audio/bin/python -m soundcloud.main sync-user https://soundcloud.com/user-327506308
venvs/audio/bin/python -m soundcloud.main crawl --seed 327506308 --depth 1
venvs/audio/bin/python -m soundcloud.main stats
venvs/audio/bin/python -m soundcloud.analysis.first_look
```

## Deferred consumers (own specs later)

Audio-seed corpus (populates `sc_recording_map`), discovery backbone, MERT
embeddings, systemd crawler service, migrating personalization off the shim,
crawl depth > 1, OAuth.
```

- [ ] **Step 2: Add `soundcloud/` to the root CLAUDE.md module index**

In `CLAUDE.md`, under "## Per-module guides", add this bullet after the `core` entry:

```markdown
- **[soundcloud/CLAUDE.md](soundcloud/CLAUDE.md)** — SoundCloud data-lake
  substrate (anon `client_id` fetch + `sc_lake.db` on pi-storage). General
  ingestion primitive consumed by `personalization`/`lab`; off the alignment DAG.
```

- [ ] **Step 3: Run the full module test suite**

Run: `venvs/audio/bin/python -m pytest tests/ -k soundcloud -v`
Expected: PASS (all soundcloud tests green).

- [ ] **Step 4: Run guardrails + entropy audit**

Run: `venvs/audio/bin/python scripts/guardrails.py`
Expected: `guardrails: OK`.
Run: `venvs/audio/bin/python scripts/entropy_audit.py`
Expected: `bare_except 0 (baseline 0)`, no new fenced instances (the new module uses no bare `except:` and passes `timeout=` to its httpx client). If any count rose, fix the offending line (do NOT bump the baseline).

- [ ] **Step 5: Commit**

```bash
git add soundcloud/CLAUDE.md CLAUDE.md
git commit -m "docs(soundcloud): module guide + root index wiring"
```

---

### Task 12: Live smoke run (manual, pi-side) — NON-BLOCKING

**Goal:** Validate against the real API on the actual seed profile. This is an *operational* step, run once the code is deployed; it is not a unit test and does not gate the merge.

- [ ] **Step 1:** Deploy the module to pi-storage (`make deploy`) OR run locally with `SC_LAKE_ROOT=$(mktemp -d)` for a Mac dry run.
- [ ] **Step 2:** Verify the JSON shapes match the parsers with one live probe:
  ```bash
  SC_LAKE_ROOT=/tmp/sc_probe venvs/audio/bin/python -m soundcloud.main sync-user https://soundcloud.com/user-327506308
  ```
- [ ] **Step 3:** Inspect coverage:
  ```bash
  SC_LAKE_ROOT=/tmp/sc_probe venvs/audio/bin/python -m soundcloud.main stats
  SC_LAKE_ROOT=/tmp/sc_probe venvs/audio/bin/python -m soundcloud.analysis.first_look
  ```
- [ ] **Step 4:** If any field is missing/misnamed vs the live JSON (e.g. likes wrapper shape), fix the parser/extractor, add a regression fixture under `tests/fixtures/` capturing the real shape, and re-run the affected unit test. Commit the fix + fixture.
- [ ] **Step 5:** Once validated on Mac, run pi-side writing to canonical `/mnt/storage/data/soundcloud/` and confirm `stats` over SSH.

---

## Self-Review

**Spec coverage:**
- §2.1 client + shim → Task 5 ✓
- §2.2 fetch primitives → Task 6 ✓
- §2.3 raw lake → Task 7 ✓
- §2.4 store (nodes/edges/identity seam stub/checkpoints) → Tasks 3–4 (`sc_recording_map` created empty in schema) ✓
- §2.5 crawl frontier depth-1 + resume → Task 8 ✓
- §2.6 CLI → Task 9 ✓
- §2.7 first-look analysis → Task 10 ✓
- Pi-storage hosting / `SC_LAKE_ROOT` → Task 1 config + Task 11 guide ✓
- Testing strategy (no live network, fakes, tmp_path) → every task ✓
- Deferred items untouched (audio, MERT, identity population, systemd, depth>1) → not implemented ✓

**Placeholder scan:** No TBD/TODO; every code step has runnable code and exact commands.

**Type consistency:** `parse_user/track/playlist` return `Sc*` records consumed by `upsert_*`; `crawl.crawl(conn, settings, policy, client, rl, cid, now, fetch_mod=...)` signature matches its callers in `main._run_crawl` and all tests; `CrawlPolicy` fields (`seed_user_ids`, `depth`, `entity_types`, `rpm`) consistent across records/crawl/main; `write_snapshot` returns the `raw_ref` string stored on every node/edge row.

**Known risk (flagged, not a gap):** exact SoundCloud api-v2 JSON shapes for `likes`/`reposts` pagination are pinned by the Task 12 live probe; Task 6/8 extractors already handle both wrapped and bare object forms, and Task 12 adds a real-shape regression fixture if anything differs.
