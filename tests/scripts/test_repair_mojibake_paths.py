"""Tests for scripts/repair_mojibake_paths.py (issue #74 step-2 repair tool).

Covers the load-bearing safety properties: the locale hard-gate, the
never-point-at-a-missing-file rule, and correct dry-run vs apply behaviour.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from scripts import repair_mojibake_paths as rmp

_SCHEMA = (
    Path(__file__).resolve().parents[2] / "web_crawler" / "database" / "schema.sql"
)

CORRECT_NAME = "Sign Of The Times – Acapella.m4a"  # real en-dash U+2013
MOJIBAKE_NAME = CORRECT_NAME.encode("utf-8").decode("latin-1")  # â€"


def _db(tmp_path: Path, correct_path: str) -> Path:
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.executescript(_SCHEMA.read_text())
    conn.execute("PRAGMA foreign_keys=OFF")
    mojibake_path = correct_path.encode("utf-8").decode("latin-1")
    conn.execute(
        "INSERT INTO track_audio (track_audio_id, recording_id, track_id, platform, "
        "source_url, player_id, path, stem, variant) "
        "VALUES (1,'R1','R1','youtube','u','p1',?,'acappella','regular')",
        (mojibake_path,),
    )
    # a clean control row that must never be flagged
    conn.execute(
        "INSERT INTO track_audio (track_audio_id, recording_id, track_id, platform, "
        "source_url, player_id, path, stem, variant) "
        "VALUES (2,'R2','R2','youtube','u','p2','/o/R2/plain.m4a','regular','regular')"
    )
    conn.commit()
    conn.close()
    return db


def test_find_mojibake_flags_only_the_bad_row(tmp_path: Path) -> None:
    correct = str(tmp_path / CORRECT_NAME)
    db = _db(tmp_path, correct)
    conn = sqlite3.connect(db)
    rows = rmp.find_mojibake(conn)
    conn.close()
    assert len(rows) == 1
    taid, cur, repaired, _exists = rows[0]
    assert taid == 1
    assert repaired == correct
    assert cur != correct


def test_apply_refused_when_not_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    correct = str(tmp_path / CORRECT_NAME)
    Path(correct).write_bytes(b"x")  # file present on disk
    db = _db(tmp_path, correct)
    monkeypatch.setattr(rmp, "_fs_is_utf8", lambda: False)
    rc = rmp.run(str(db), apply=True)
    assert rc == 2  # refused
    # DB unchanged
    conn = sqlite3.connect(db)
    stored = conn.execute(
        "SELECT path FROM track_audio WHERE track_audio_id=1"
    ).fetchone()[0]
    conn.close()
    assert stored != correct  # still mojibake


def test_apply_repairs_when_utf8_and_file_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    correct = str(tmp_path / CORRECT_NAME)
    Path(correct).write_bytes(b"x")  # repaired path resolves on disk
    db = _db(tmp_path, correct)
    monkeypatch.setattr(rmp, "_fs_is_utf8", lambda: True)
    rc = rmp.run(str(db), apply=True)
    assert rc == 0
    conn = sqlite3.connect(db)
    stored = conn.execute(
        "SELECT path FROM track_audio WHERE track_audio_id=1"
    ).fetchone()[0]
    conn.close()
    assert stored == correct


def test_apply_skips_when_repaired_file_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    correct = str(tmp_path / CORRECT_NAME)  # deliberately NOT created on disk
    db = _db(tmp_path, correct)
    monkeypatch.setattr(rmp, "_fs_is_utf8", lambda: True)
    rc = rmp.run(str(db), apply=True)
    assert rc == 1  # skipped -> nonzero
    conn = sqlite3.connect(db)
    stored = conn.execute(
        "SELECT path FROM track_audio WHERE track_audio_id=1"
    ).fetchone()[0]
    conn.close()
    assert stored != correct  # untouched: never point DB at a missing file


def test_dry_run_never_mutates(tmp_path: Path) -> None:
    correct = str(tmp_path / CORRECT_NAME)
    Path(correct).write_bytes(b"x")
    db = _db(tmp_path, correct)
    rmp.run(str(db), apply=False)
    conn = sqlite3.connect(db)
    stored = conn.execute(
        "SELECT path FROM track_audio WHERE track_audio_id=1"
    ).fetchone()[0]
    conn.close()
    assert stored != correct  # dry-run left the mojibake in place
