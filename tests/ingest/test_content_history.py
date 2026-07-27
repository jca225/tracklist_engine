"""B1 — content_history ledger: append + read the generation chain.

The ledger is the byte-level shadow of track_audio_correction (spec v3.2):
every acquisition/correction event appends a generation stamped with its
hashes + axes. Chain key = (recording_id, stem, variant, kind) (spec v4.6) —
axis-granular so an acappella lineage never merges with the regular one, and a
separated-stem lineage never merges with a downloaded-master lineage.

Soundness property under test: the ledger keeps the OLD sha256 bindable across
a replace (the never-drop-hash invariant B2 relies on), and keeps distinct
axis/kind lineages strictly separate (an axis error would be a wrong label).
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

from core.result import Err, Ok
from ingest.content_history import (
    SCHEMA,
    Generation,
    append_generation,
    chain,
)

_REPO = Path(__file__).resolve().parents[2]
_MIGRATION = _REPO / "scripts" / "migrations" / "migrate_content_history.sql"
_SCHEMA_SQL = _REPO / "web_crawler" / "database" / "schema.sql"


@pytest.fixture
def db(tmp_path: Path) -> Path:
    p = tmp_path / "t.db"
    with sqlite3.connect(p) as c:
        c.executescript(SCHEMA)
    return p


def _gen(**kw) -> Generation:
    base = dict(
        recording_id="rec_congrats",
        stem="regular",
        variant="regular",
        kind="master",
    )
    base.update(kw)
    return Generation(**base)


def test_append_and_read_chain(db: Path):
    r0 = append_generation(
        db, _gen(content_sha256="aaa", op="fetch", source="ingest.main")
    )
    r1 = append_generation(
        db, _gen(content_sha256="bbb", op="replace", source="replace_track_audio")
    )
    assert isinstance(r0, Ok) and isinstance(r1, Ok)

    ch = chain(db, "rec_congrats", "regular", "regular", "master")
    assert [g["content_sha256"] for g in ch] == ["aaa", "bbb"]
    # generation ordinal auto-assigned per chain key, starting at 0
    assert [g["generation"] for g in ch] == [0, 1]
    # the retired 'aaa' bytes are STILL in the ledger after the replace-gen —
    # this is the never-drop-hash invariant B2 depends on.
    assert "aaa" in {g["content_sha256"] for g in ch}


def test_chain_key_isolates_stem_axis(db: Path):
    """Acappella and regular of the same recording are SEPARATE chains — a
    naive recording_id-only key would let an acappella clip bind a regular
    generation (a stem-axis wrong label, spec v3.1/v4.6)."""
    append_generation(db, _gen(stem="regular", content_sha256="reg0"))
    append_generation(db, _gen(stem="acappella", content_sha256="aca0"))

    reg = chain(db, "rec_congrats", "regular", "regular", "master")
    aca = chain(db, "rec_congrats", "acappella", "regular", "master")
    assert [g["content_sha256"] for g in reg] == ["reg0"]
    assert [g["content_sha256"] for g in aca] == ["aca0"]
    # each chain restarts generation numbering at 0 (per-key ordinal)
    assert reg[0]["generation"] == 0 and aca[0]["generation"] == 0


def test_chain_key_isolates_kind(db: Path):
    """A demucs-vocals (separated) lineage and a downloaded-acappella-master
    lineage must not merge even at the same (recording, stem, variant) — v4.6."""
    append_generation(
        db, _gen(stem="acappella", kind="master", content_sha256="dl_master")
    )
    append_generation(
        db, _gen(stem="acappella", kind="separated", content_sha256="demucs_vox")
    )

    m = chain(db, "rec_congrats", "acappella", "regular", "master")
    s = chain(db, "rec_congrats", "acappella", "regular", "separated")
    assert [g["content_sha256"] for g in m] == ["dl_master"]
    assert [g["content_sha256"] for g in s] == ["demucs_vox"]


def test_explicit_generation_is_respected(db: Path):
    """Backfill path stamps generation 0 explicitly (migration seeds gen 0)."""
    r = append_generation(db, _gen(content_sha256="seed", generation=0))
    assert isinstance(r, Ok)
    ch = chain(db, "rec_congrats", "regular", "regular", "master")
    assert ch[0]["generation"] == 0


def test_bad_kind_rejected(db: Path):
    r = append_generation(db, _gen(kind="bogus", content_sha256="x"))
    assert isinstance(r, Err)
    assert r.error.kind == "bad_kind"


def test_valid_flag_defaults_true_and_filters(db: Path):
    """valid=1 by default; chain(valid_only=True) hides tombstoned generations
    (the B4 invalidation path writes valid=0)."""
    append_generation(db, _gen(content_sha256="live"))
    append_generation(db, _gen(content_sha256="dead", valid=0))
    all_gens = chain(db, "rec_congrats", "regular", "regular", "master")
    live_only = chain(
        db, "rec_congrats", "regular", "regular", "master", valid_only=True
    )
    assert {g["content_sha256"] for g in all_gens} == {"live", "dead"}
    assert {g["content_sha256"] for g in live_only} == {"live"}


def test_append_self_heals_missing_table(tmp_path: Path):
    """The pi auto-pull timer (issue #73) ships code without running
    migrations, so a hook can fire before content_history exists. append must
    create the table rather than crash; chain must read it back."""
    p = tmp_path / "no_table.db"
    p.touch()  # empty DB, no content_history
    r = append_generation(p, _gen(content_sha256="firstever"))
    assert isinstance(r, Ok)
    ch = chain(p, "rec_congrats", "regular", "regular", "master")
    assert [g["content_sha256"] for g in ch] == ["firstever"]


def test_chain_on_missing_table_returns_empty(tmp_path: Path):
    p = tmp_path / "empty.db"
    p.touch()
    assert chain(p, "whatever", "regular", "regular", "master") == []


def test_parent_hashes_roundtrip(db: Path):
    """C0 — separated generations carry parent master hashes for derivation cert."""
    r = append_generation(
        db,
        _gen(
            stem="acappella",
            kind="separated",
            content_sha256="stem_old",
            parent_content_sha256="parent_master_sha",
            parent_payload_sha256="parent_payload_sha",
            op="re-separate",
        ),
    )
    assert isinstance(r, Ok)
    ch = chain(db, "rec_congrats", "acappella", "regular", "separated")
    assert ch[0]["parent_content_sha256"] == "parent_master_sha"
    assert ch[0]["parent_payload_sha256"] == "parent_payload_sha"


def test_self_heal_adds_parent_columns_to_legacy_table(tmp_path: Path):
    """Pre-C0 DBs have content_history without parent_*; append must ALTER."""
    p = tmp_path / "legacy.db"
    legacy = """
    CREATE TABLE content_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recording_id TEXT NOT NULL,
        version TEXT,
        stem TEXT NOT NULL,
        variant TEXT NOT NULL,
        kind TEXT NOT NULL,
        track_audio_id INTEGER,
        content_sha256 TEXT,
        payload_sha256 TEXT,
        op TEXT,
        source TEXT,
        generation INTEGER NOT NULL DEFAULT 0,
        valid INTEGER NOT NULL DEFAULT 1,
        ts DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """
    with sqlite3.connect(p) as c:
        c.executescript(legacy)
    r = append_generation(
        p,
        _gen(
            kind="separated",
            stem="acappella",
            content_sha256="s1",
            parent_content_sha256="p1",
        ),
    )
    assert isinstance(r, Ok)
    ch = chain(p, "rec_congrats", "acappella", "regular", "separated")
    assert ch[0]["parent_content_sha256"] == "p1"


def test_parent_migration_alters_legacy_table(tmp_path: Path):
    """migrate_content_history_parent.sql adds parent_* onto a pre-C0 table."""
    p = tmp_path / "legacy.db"
    parent_mig = _REPO / "scripts" / "migrations" / "migrate_content_history_parent.sql"
    with sqlite3.connect(p) as c:
        c.executescript(
            """
            CREATE TABLE content_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recording_id TEXT NOT NULL,
                version TEXT,
                stem TEXT NOT NULL,
                variant TEXT NOT NULL,
                kind TEXT NOT NULL,
                track_audio_id INTEGER,
                content_sha256 TEXT,
                payload_sha256 TEXT,
                op TEXT,
                source TEXT,
                generation INTEGER NOT NULL DEFAULT 0,
                valid INTEGER NOT NULL DEFAULT 1,
                ts DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        # Apply only the ALTER lines (skip PRAGMA / comments).
        for line in parent_mig.read_text().splitlines():
            s = line.strip()
            if s.upper().startswith("ALTER TABLE"):
                c.execute(s.rstrip(";"))
        cols = {r[1] for r in c.execute("PRAGMA table_info(content_history)")}
    assert "parent_content_sha256" in cols
    assert "parent_payload_sha256" in cols


# --- migration: backfill + idempotency + DDL drift guard ---

_TRACK_AUDIO = """
CREATE TABLE track_audio (
    track_audio_id INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id TEXT NOT NULL, track_id TEXT NOT NULL,
    platform TEXT, source_url TEXT, player_id TEXT, path TEXT,
    sha256 TEXT, stem TEXT NOT NULL DEFAULT 'regular',
    variant TEXT NOT NULL DEFAULT 'regular'
);
"""


def _norm(sql: str) -> str:
    # strip -- line comments first (before newlines are collapsed), then
    # whitespace-normalize: drift is a structural difference, not a comment one.
    no_comments = re.sub(r"--[^\n]*", "", sql)
    return re.sub(r"\s+", " ", no_comments).strip().lower()


def _create_stmt(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='content_history'"
    ).fetchone()
    return _norm(row[0]) if row else ""


def test_migration_backfills_generation_zero(tmp_path: Path):
    p = tmp_path / "canon.db"
    with sqlite3.connect(p) as c:
        c.executescript(_TRACK_AUDIO)
        c.execute(
            "INSERT INTO track_audio (recording_id, track_id, sha256, stem, variant) "
            "VALUES ('recA','recA','sha_a','regular','regular')"
        )
        c.execute(
            "INSERT INTO track_audio (recording_id, track_id, sha256, stem, variant) "
            "VALUES ('recB','recB','sha_b','acappella','regular')"
        )
        # a row with NULL sha256 must NOT be backfilled
        c.execute(
            "INSERT INTO track_audio (recording_id, track_id, sha256) "
            "VALUES ('recC','recC',NULL)"
        )
        c.executescript(_MIGRATION.read_text())
        rows = c.execute(
            "SELECT recording_id, stem, kind, generation, op, source, content_sha256 "
            "FROM content_history ORDER BY recording_id"
        ).fetchall()
    assert [r[0] for r in rows] == ["recA", "recB"]  # recC (null sha) skipped
    assert all(r[2] == "master" and r[3] == 0 and r[4] == "fetch" for r in rows)
    assert rows[1][1] == "acappella"  # stem axis carried from track_audio


def test_migration_is_idempotent(tmp_path: Path):
    p = tmp_path / "canon.db"
    with sqlite3.connect(p) as c:
        c.executescript(_TRACK_AUDIO)
        c.execute(
            "INSERT INTO track_audio (recording_id, track_id, sha256) "
            "VALUES ('recA','recA','sha_a')"
        )
        c.executescript(_MIGRATION.read_text())
        c.executescript(_MIGRATION.read_text())  # second run = no-op
        (n,) = c.execute("SELECT COUNT(*) FROM content_history").fetchone()
    assert n == 1  # not doubled


def test_ddl_matches_across_all_three_sources():
    """SCHEMA (module), schema.sql, and the migration must define an identical
    content_history table — the single-source promise. Guard against drift.

    Applies each source in full (the migration needs a track_audio stub for its
    backfill), then compares the CREATE statement sqlite recorded."""

    def create_stmt_after(scripts: list[str]) -> str:
        with sqlite3.connect(":memory:") as c:
            for s in scripts:
                c.executescript(s)
            return _create_stmt(c)

    from_module = create_stmt_after([SCHEMA])
    from_schema = create_stmt_after([_SCHEMA_SQL.read_text()])
    from_migration = create_stmt_after([_TRACK_AUDIO, _MIGRATION.read_text()])

    assert from_module
    assert from_module == from_schema, (
        "schema.sql content_history drifted from module SCHEMA"
    )
    assert from_module == from_migration, (
        "migration content_history drifted from module SCHEMA"
    )
