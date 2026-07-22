"""B2 — never-drop-hash: a replace records the retiring row's sha256 into
content_history (same transaction) BEFORE the DELETE.

The redownload/replace churn is the incompleteness factory (spec P9): deleting
a track_audio row drops the only sound evidence of its bytes, silently
un-binding every GT clip pinned to them. This test pins the invariant that the
delete can never happen without the hash first landing in the ledger.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from scripts.replace_track_audio import _delete_old_row_if_exists

# Minimal standalone track_audio (no FK → nothing for PRAGMA foreign_keys to
# enforce). content_history self-creates on first append.
_TRACK_AUDIO = """
CREATE TABLE track_audio (
    track_audio_id INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id TEXT NOT NULL, track_id TEXT NOT NULL,
    platform TEXT, source_url TEXT, player_id TEXT, path TEXT,
    sha256 TEXT, stem TEXT NOT NULL DEFAULT 'regular',
    variant TEXT NOT NULL DEFAULT 'regular'
);
"""


def _mk_db(
    tmp_path: Path, sha: str | None, stem: str = "regular", variant: str = "regular"
) -> tuple[Path, int]:
    p = tmp_path / "canon.db"
    with sqlite3.connect(p) as c:
        c.executescript(_TRACK_AUDIO)
        cur = c.execute(
            "INSERT INTO track_audio (recording_id, track_id, path, sha256, stem, variant) "
            "VALUES ('recX','recX',?,?,?,?)",
            (str(tmp_path / "nonexistent.m4a"), sha, stem, variant),
        )
        taid = int(cur.lastrowid)
    return p, taid


def _history(db: Path) -> list[dict]:
    with sqlite3.connect(db) as c:
        c.row_factory = sqlite3.Row
        try:
            rows = c.execute("SELECT * FROM content_history").fetchall()
        except sqlite3.OperationalError:
            return []  # never appended → table legitimately absent
        return [dict(r) for r in rows]


def _track_audio_count(db: Path, taid: int) -> int:
    with sqlite3.connect(db) as c:
        (n,) = c.execute(
            "SELECT COUNT(*) FROM track_audio WHERE track_audio_id=?", (taid,)
        ).fetchone()
    return n


def test_replace_records_old_hash_then_deletes(tmp_path: Path):
    db, taid = _mk_db(
        tmp_path, sha="old_bytes_sha", stem="acappella", variant="extended"
    )
    _delete_old_row_if_exists(db, tmp_path, taid)

    hist = _history(db)
    assert len(hist) == 1
    h = hist[0]
    assert h["content_sha256"] == "old_bytes_sha"
    assert h["recording_id"] == "recX"
    assert h["kind"] == "master"
    assert h["op"] == "replace"
    assert h["source"] == "replace_track_audio"
    # axes carried from the retiring row (a stem/variant error here would be a
    # wrong label on a future bind)
    assert h["stem"] == "acappella"
    assert h["variant"] == "extended"
    assert h["track_audio_id"] == taid
    # and the row is actually gone
    assert _track_audio_count(db, taid) == 0


def test_null_sha_row_deleted_without_history(tmp_path: Path):
    db, taid = _mk_db(tmp_path, sha=None)
    _delete_old_row_if_exists(db, tmp_path, taid)
    assert _history(db) == []  # nothing to preserve
    assert _track_audio_count(db, taid) == 0


def test_purge_op_is_stamped(tmp_path: Path):
    db, taid = _mk_db(tmp_path, sha="sibling_sha")
    _delete_old_row_if_exists(db, tmp_path, taid, op="purge")
    hist = _history(db)
    assert len(hist) == 1 and hist[0]["op"] == "purge"


def test_missing_row_is_noop(tmp_path: Path):
    db, taid = _mk_db(tmp_path, sha="x")
    _delete_old_row_if_exists(db, tmp_path, 9999)  # no such taid
    assert _history(db) == []
    assert _track_audio_count(db, taid) == 1  # the real row untouched
