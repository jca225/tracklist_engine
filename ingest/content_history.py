"""Content-history hash ledger (ingest stage) — the byte-level shadow of
track_audio_correction.

Every acquisition/correction event (fetch, retry, re-separate, retag, replace,
relink, detach, …) appends a *generation* stamped with the artifact's hashes and
its four identity axes. The ledger is **append-only**: a `replace`/re-separate
must record the retiring row's sha256 *before* the row is deleted (B2), so a GT
clip pinned to the old bytes still content-binds exactly next export (spec v2.2)
instead of silently abstaining.

Chain key = ``(recording_id, stem, variant, kind)`` (spec v4.6), ``kind ∈
{master, separated, derived}``:

* axis-granular so an acappella lineage never merges with the regular one (a
  stem-axis merge would be a wrong label — spec v3.1);
* ``kind`` keeps a demucs-vocals (``separated``) lineage distinct from a
  downloaded-acappella-master (``master``) lineage even at the same axes;
* ``track_audio_id`` is NOT the key — a ``replace`` mints a new id, and chains
  must span rows (v4.6). It is stored denormalized for provenance only.

Binding across generations is NOT done here — that is a certificate-gated
decision in the catalog builder (B3, spec v4.1). This module only records and
reads the chain. ``valid=0`` is a tombstone (B4, spec v4.2): an invalidated
generation stays in the ledger for audit but is no longer a legal bind target.

stdlib + core.db only (must run on pi's bare python3).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from core.db import connect
from core.errors import DbError
from core.result import Err, Ok, Result

KINDS = ("master", "separated", "derived")

# Single source of DDL — schema.sql and the migration embed this verbatim, and
# tests apply it directly, so the three cannot drift. NO foreign key on
# track_audio_id: a `replace` deletes the track_audio row and an ON DELETE
# CASCADE would take the history row with it, defeating the never-drop-hash
# invariant (B2). The ledger must outlive the rows it records.
SCHEMA = """
CREATE TABLE IF NOT EXISTS content_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id    TEXT NOT NULL,
    version         TEXT,                      -- denormalized axis audit (v4.6); recording_id pins it
    stem            TEXT NOT NULL,             -- track_audio row's realization value (regular|acappella|instrumental)
    variant         TEXT NOT NULL,             -- regular|extended
    kind            TEXT NOT NULL,             -- master|separated|derived
    track_audio_id  INTEGER,                   -- row this generation realized (nullable; replace mints a new id)
    content_sha256  TEXT,                      -- full-file sha256 (== track_audio.sha256)
    payload_sha256  TEXT,                      -- mdat / FLAC decoded-PCM MD5 (tag-invariant)
    parent_content_sha256 TEXT,                -- sound parent master content hash (derivation cert, C0)
    parent_payload_sha256 TEXT,                -- optional parent payload hash (derivation cert, C0)
    op              TEXT,                      -- fetch|retry|re-separate|retag|replace|relink|detach|...
    source          TEXT,                      -- replace_track_audio|acquire_variant|manual|...
    generation      INTEGER NOT NULL DEFAULT 0,
    valid           INTEGER NOT NULL DEFAULT 1,-- tombstone: 0 = invalidated, not a legal bind target (v4.2)
    ts              DATETIME DEFAULT CURRENT_TIMESTAMP,
    CHECK (kind IN ('master','separated','derived')),
    CHECK (valid IN (0,1))
);
CREATE INDEX IF NOT EXISTS idx_content_history_chain
    ON content_history(recording_id, stem, variant, kind, generation);
CREATE INDEX IF NOT EXISTS idx_content_history_content_sha ON content_history(content_sha256);
CREATE INDEX IF NOT EXISTS idx_content_history_payload_sha ON content_history(payload_sha256);
"""

# Additive columns for DBs whose CREATE ran before C0 (pi auto-pull may ship
# code before migrate_content_history_parent.sql). Applied after SCHEMA.
_PARENT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("parent_content_sha256", "TEXT"),
    ("parent_payload_sha256", "TEXT"),
)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """CREATE IF NOT EXISTS + ALTER missing parent_* columns (C0 self-heal)."""
    conn.executescript(SCHEMA)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(content_history)")}
    for name, typ in _PARENT_COLUMNS:
        if name not in cols:
            conn.execute(f"ALTER TABLE content_history ADD COLUMN {name} {typ}")


@dataclass(frozen=True)
class Generation:
    """One generation on a ``(recording_id, stem, variant, kind)`` chain."""

    recording_id: str
    stem: str
    variant: str
    kind: str
    content_sha256: str | None = None
    payload_sha256: str | None = None
    parent_content_sha256: str | None = None
    parent_payload_sha256: str | None = None
    track_audio_id: int | None = None
    version: str | None = None
    op: str | None = None
    source: str | None = None
    generation: int | None = None  # None → next ordinal within the chain key
    valid: int = 1


def _next_generation(conn: sqlite3.Connection, g: Generation) -> int:
    row = conn.execute(
        """
        SELECT COALESCE(MAX(generation) + 1, 0) AS n
        FROM content_history
        WHERE recording_id = ? AND stem = ? AND variant = ? AND kind = ?
        """,
        (g.recording_id, g.stem, g.variant, g.kind),
    ).fetchone()
    return int(row["n"])


def _validate(g: Generation) -> DbError | None:
    if g.kind not in KINDS:
        return DbError(kind="bad_kind", detail=f"{g.kind!r} not in {KINDS}")
    if g.valid not in (0, 1):
        return DbError(kind="bad_valid", detail=f"valid={g.valid!r} not in (0,1)")
    return None


def append_generation_on(conn: sqlite3.Connection, g: Generation) -> int:
    """Insert a generation on an existing connection; return the new row id.

    For callers that must record a generation **atomically with their own
    mutation** — e.g. `replace_track_audio` recording the retiring row's sha256
    in the *same transaction* as the DELETE, so the churn can never delete the
    evidence (spec P9). The caller owns the commit. Raises ``ValueError`` on
    invalid input and propagates ``sqlite3.Error`` (so it rolls back with the
    caller's transaction).
    """
    err = _validate(g)
    if err is not None:
        raise ValueError(f"{err.kind}: {err.detail}")
    # Self-heal: ensure the table (+ C0 parent_* columns) exists before writing.
    # The pi's auto-pull timer (issue #73) ships code without running migrations,
    # so a never-drop-hash hook can land before the backfill / parent migrations
    # run — this makes the first append create/alter rather than crash.
    _ensure_schema(conn)
    gen = g.generation if g.generation is not None else _next_generation(conn, g)
    cur = conn.execute(
        """
        INSERT INTO content_history
          (recording_id, version, stem, variant, kind, track_audio_id,
           content_sha256, payload_sha256, parent_content_sha256,
           parent_payload_sha256, op, source, generation, valid)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            g.recording_id,
            g.version,
            g.stem,
            g.variant,
            g.kind,
            g.track_audio_id,
            g.content_sha256,
            g.payload_sha256,
            g.parent_content_sha256,
            g.parent_payload_sha256,
            g.op,
            g.source,
            gen,
            g.valid,
        ),
    )
    return int(cur.lastrowid)


def append_generation(db_path: Path, g: Generation) -> Result[int, DbError]:
    """Append a generation to its chain. Returns the new row id.

    ``generation`` is auto-assigned as the next ordinal within the chain key
    when left ``None`` (the backfill path passes ``generation=0`` explicitly).
    The ordinal is an *audit* index, not the bind key — binding is by hash
    equality regardless of ordinal — so a concurrent-writer race on the ordinal
    cannot produce a wrong label, only a duplicated ordinal.
    """
    err = _validate(g)
    if err is not None:
        return Err(err)
    try:
        with connect(db_path) as conn:
            rid = append_generation_on(conn, g)
            conn.commit()
            return Ok(rid)
    except sqlite3.Error as e:
        return Err(DbError(kind="integrity", detail=str(e)))


def chain(
    db_path: Path,
    recording_id: str,
    stem: str,
    variant: str,
    kind: str,
    *,
    valid_only: bool = False,
) -> list[dict]:
    """Read the generation chain for a key, oldest first.

    ``valid_only=True`` hides tombstoned (``valid=0``) generations — the shape
    the exporter consumes, so it never binds a clip to an invalidated
    generation (B4 / spec v4.2).
    """
    q = (
        "SELECT * FROM content_history "
        "WHERE recording_id = ? AND stem = ? AND variant = ? AND kind = ?"
    )
    if valid_only:
        q += " AND valid = 1"
    q += " ORDER BY generation, id"
    with connect(db_path) as conn:
        try:
            rows = conn.execute(q, (recording_id, stem, variant, kind)).fetchall()
        except sqlite3.OperationalError:
            # Table not created yet (code auto-landed before the migration ran,
            # issue #73). "No history recorded" is the correct semantic — an
            # empty chain, never a crash.
            return []
    return [dict(r) for r in rows]


def tombstone_on(
    conn: sqlite3.Connection,
    recording_id: str,
    *,
    stem: str | None = None,
    variant: str | None = None,
    kind: str | None = None,
) -> int:
    """Invalidate (valid→0) the generations on a chain, on an existing
    connection; return the count invalidated. The caller owns the commit.

    A ``relink`` (the audio was attached to the wrong recording) or ``detach``
    (abstain) means the generations recorded under this recording/stem are no
    longer a legal bind target (spec v4.2). We flip ``valid`` rather than delete
    — the hash is kept for audit, and ``chain(valid_only=True)`` already hides
    it from binding. The narrowest specified scope is invalidated: pass only
    ``recording_id`` to tombstone the whole recording, or add ``stem``/
    ``variant``/``kind`` to narrow it. Tolerates a not-yet-created table
    (returns 0) for auto-pull safety (issue #73)."""
    clauses = ["recording_id = ?"]
    params: list[object] = [recording_id]
    for col, val in (("stem", stem), ("variant", variant), ("kind", kind)):
        if val is not None:
            clauses.append(f"{col} = ?")
            params.append(val)
    where = " AND ".join(clauses)
    try:
        cur = conn.execute(
            f"UPDATE content_history SET valid = 0 WHERE {where} AND valid = 1",
            tuple(params),
        )
    except sqlite3.OperationalError:
        return 0  # no table yet → nothing to invalidate
    return cur.rowcount


def tombstone(
    db_path: Path,
    recording_id: str,
    *,
    stem: str | None = None,
    variant: str | None = None,
    kind: str | None = None,
) -> Result[int, DbError]:
    """Standalone tombstone wrapper (owns its transaction)."""
    try:
        with connect(db_path) as conn:
            n = tombstone_on(conn, recording_id, stem=stem, variant=variant, kind=kind)
            conn.commit()
            return Ok(n)
    except sqlite3.Error as e:
        return Err(DbError(kind="integrity", detail=str(e)))


# --- C1: stem-regen never-drop (hash existing stems before overwrite) ---

_STEM_FILE_TO_AXIS = {
    "vocals": "acappella",
    "instrumental": "instrumental",
}
_STEM_EXTS = (".flac", ".wav", ".mp3", ".m4a")


def _existing_stem_path(stem_dest: Path, name: str) -> Path | None:
    for ext in _STEM_EXTS:
        p = stem_dest / f"{name}{ext}"
        if p.is_file():
            return p
    return None


def record_stems_before_overwrite(
    db_path: Path,
    stem_dest: Path,
    *,
    recording_id: str,
    variant: str,
    parent_content_sha256: str | None,
    parent_payload_sha256: str | None = None,
    track_audio_id: int | None = None,
    source: str = "run_separation",
) -> int:
    """C1 — hash existing vocals/instrumental stems and append separated gens.

    Call **before** a re-separation overwrites ``stem_dest``. Only the DJ-axis
    stems (vocals→acappella, instrumental→instrumental) are recorded; drums/
    bass/other are ignored (P15). Returns the number of generations appended.

    Requires ``parent_content_sha256`` (the regular master's content hash at
    separation time) — without it there is no derivation certificate later.
    Fail-soft: missing files / missing table / IO errors return 0 rather than
    raising into the separation hot path.
    """
    if not parent_content_sha256 or not recording_id:
        return 0
    if not stem_dest.is_dir():
        return 0
    n = 0
    try:
        # Lazy import: content_hash pulls nothing heavy; keep content_history
        # importable on bare pi python3 for ledger-only callers.
        from labeling.identity.content_hash import file_sha256, flac_pcm_md5
    except ImportError:
        return 0
    try:
        with connect(db_path) as conn:
            for fname, axis in _STEM_FILE_TO_AXIS.items():
                path = _existing_stem_path(stem_dest, fname)
                if path is None:
                    continue
                try:
                    csha = file_sha256(path)
                except OSError:
                    continue
                psha = None
                if path.suffix.lower() == ".flac":
                    try:
                        psha = flac_pcm_md5(path)
                    except OSError:
                        psha = None
                append_generation_on(
                    conn,
                    Generation(
                        recording_id=recording_id,
                        stem=axis,
                        variant=variant or "regular",
                        kind="separated",
                        content_sha256=csha,
                        payload_sha256=psha,
                        parent_content_sha256=parent_content_sha256,
                        parent_payload_sha256=parent_payload_sha256,
                        track_audio_id=track_audio_id,
                        op="re-separate",
                        source=source,
                    ),
                )
                n += 1
            if n:
                conn.commit()
    except (sqlite3.Error, ValueError, OSError):
        return 0
    return n
