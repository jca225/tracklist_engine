from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pytest

from scripts import replace_stem_audio as rsa
from ingest.same_song_guard import GuardVerdict

_SCHEMA = """
CREATE TABLE work (work_id TEXT PRIMARY KEY, title TEXT);
CREATE TABLE recording (recording_id TEXT PRIMARY KEY, work_id TEXT, full_name TEXT);
CREATE TABLE track_audio (
  track_audio_id INTEGER PRIMARY KEY, recording_id TEXT, track_id TEXT, stem TEXT,
  variant TEXT DEFAULT 'regular', sha256 TEXT,
  path TEXT, platform TEXT, player_id TEXT, source_url TEXT,
  is_reference INTEGER DEFAULT 0, downloaded_at DATETIME);
CREATE TABLE track_audio_correction (
  correction_id INTEGER PRIMARY KEY AUTOINCREMENT, set_id TEXT, position TEXT,
  track_id TEXT NOT NULL, axis TEXT NOT NULL, action TEXT NOT NULL,
  old_track_audio_id INTEGER, old_platform TEXT, old_player_id TEXT, old_url TEXT,
  new_track_audio_id INTEGER, new_platform TEXT, new_player_id TEXT, new_url TEXT,
  old_recording_id TEXT, new_recording_id TEXT, stem_value TEXT, reason TEXT,
  source TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  CHECK (axis IN ('version','variant','stem','recording')),
  CHECK (action IN ('replace','add','relink','detach')));
"""

_RECORDING_ID = "42wv4vp"
_OLD_TAID = 1


@pytest.fixture
def db(tmp_path: Path) -> Path:
    p = tmp_path / "t.db"
    regular_path = tmp_path / "regular.wav"
    regular_path.write_bytes(b"regular-audio")
    old_stem_path = tmp_path / "old_stem.wav"
    old_stem_path.write_bytes(b"old-stem-audio")
    with sqlite3.connect(p) as c:
        c.executescript(_SCHEMA)
        c.execute("INSERT INTO work VALUES ('w1','Good Time')")
        c.execute(
            f"INSERT INTO recording VALUES ('{_RECORDING_ID}','w1','Owl City - Good Time')"
        )
        # regular reference row (content_gate's fingerprint anchor)
        c.execute(
            "INSERT INTO track_audio "
            "(track_audio_id, recording_id, track_id, stem, path, platform, "
            " player_id, source_url, is_reference) "
            "VALUES (2, ?, ?, 'regular', ?, 'youtube', 'reg1', 'http://reg', 1)",
            (_RECORDING_ID, _RECORDING_ID, str(regular_path)),
        )
        # the pre-existing stem row this test replaces
        c.execute(
            "INSERT INTO track_audio "
            "(track_audio_id, recording_id, track_id, stem, path, platform, "
            " player_id, source_url, is_reference) "
            "VALUES (?, ?, ?, 'acappella', ?, 'youtube', 'old123', 'http://old', 0)",
            (_OLD_TAID, _RECORDING_ID, _RECORDING_ID, str(old_stem_path)),
        )
    return p


def _argv(db: Path, tmp_path: Path, *, force: bool = False) -> list[str]:
    argv = [
        "--track-audio-id",
        str(_OLD_TAID),
        "--url",
        "https://example.com/candidate",
        "--set-id",
        "1fsnxchk",
        "--position",
        "148w1",
        "--reason",
        "test replace",
        "--no-identity-check",
        "--db",
        str(db),
        "--audio-root",
        str(tmp_path),
    ]
    if force:
        argv.append("--force")
    return argv


def _fake_rta_main_factory(db: Path, new_path: Path, captured: dict):
    """Mirrors what the REAL replace_track_audio.main() does on a successful
    replace: insert the new row, delete the old row, and commit a
    'stem/replace' correction — all BEFORE replace_stem_audio.py's own
    content gate runs. This is exactly the ordering finding #2 is about."""

    def _fake_rta_main(argv):
        with sqlite3.connect(db) as c:
            cur = c.execute(
                "INSERT INTO track_audio "
                "(recording_id, track_id, stem, path, platform, player_id, "
                " source_url, is_reference) "
                "VALUES (?, ?, 'acappella', ?, 'youtube', 'new123', 'http://new', 0)",
                (_RECORDING_ID, _RECORDING_ID, str(new_path)),
            )
            new_taid = cur.lastrowid
            captured["new_taid"] = new_taid
            c.execute("DELETE FROM track_audio WHERE track_audio_id = ?", (_OLD_TAID,))
            c.execute(
                "INSERT INTO track_audio_correction "
                "(track_id, axis, action, old_track_audio_id, new_track_audio_id, "
                " set_id, position, reason, source) "
                "VALUES (?, 'stem', 'replace', ?, ?, '1fsnxchk', '148w1', "
                " 'test replace', 'replace_track_audio')",
                (_RECORDING_ID, _OLD_TAID, new_taid),
            )
            c.commit()
        return 0

    return _fake_rta_main


def test_replace_stem_audio_reaps_on_content_refuse(
    db: Path, tmp_path: Path, monkeypatch
):
    """IMPORTANT #3: replace_stem_audio's content-gate path had no coverage.
    On refuse (no --force): the just-written row must be reaped and exactly
    one axis='recording' action='detach' correction logged.

    Also verifies IMPORTANT #2's fix: rta.main() already committed a
    'stem/replace' correction for the row before this gate ran (mirrored by
    the fake above, matching the real internal behavior); the detach row
    must make the rollback explicit instead of leaving that as a silent,
    misleading "replace succeeded" ledger entry for audio that no longer
    exists.
    """
    new_path = tmp_path / "new_replacement.wav"
    new_path.write_bytes(b"bad-new-audio")
    captured: dict = {}

    monkeypatch.setattr(rsa.rta, "main", _fake_rta_main_factory(db, new_path, captured))
    monkeypatch.setattr(
        "ingest.stem_guard_runner.content_gate",
        lambda stem_axis, regular_path, candidate_path: GuardVerdict(
            False, "content", "WRONG_SONG: fake mismatch for test"
        ),
    )

    rc = rsa.main(_argv(db, tmp_path))

    assert rc == 3
    assert not new_path.exists()  # reaped
    with sqlite3.connect(db) as c:
        c.row_factory = sqlite3.Row
        detach_rows = c.execute(
            "SELECT * FROM track_audio_correction WHERE axis='recording' AND action='detach'"
        ).fetchall()
        n_taudio = c.execute(
            "SELECT count(*) FROM track_audio WHERE recording_id=? AND stem='acappella'",
            (_RECORDING_ID,),
        ).fetchone()[0]
    assert len(detach_rows) == 1
    assert n_taudio == 0  # nothing survives for this stem — old deleted, new reaped

    detach = detach_rows[0]
    assert detach["old_track_audio_id"] == captured["new_taid"]
    assert (
        "rolled back" in (detach["reason"] or "").lower()
        or "rolls back" in (detach["reason"] or "").lower()
    )


def test_replace_stem_audio_force_overrides_content_refuse(
    db: Path, tmp_path: Path, monkeypatch
):
    """--force must proceed without reaping despite a content refuse."""
    new_path = tmp_path / "new_replacement.wav"
    new_path.write_bytes(b"bad-new-audio")
    captured: dict = {}

    monkeypatch.setattr(rsa.rta, "main", _fake_rta_main_factory(db, new_path, captured))
    monkeypatch.setattr(
        "ingest.stem_guard_runner.content_gate",
        lambda stem_axis, regular_path, candidate_path: GuardVerdict(
            False, "content", "WRONG_SONG: fake mismatch for test"
        ),
    )

    rc = rsa.main(_argv(db, tmp_path, force=True))

    assert rc == 0
    assert new_path.exists()  # NOT reaped
    with sqlite3.connect(db) as c:
        n_taudio = c.execute(
            "SELECT count(*) FROM track_audio WHERE recording_id=? AND stem='acappella'",
            (_RECORDING_ID,),
        ).fetchone()[0]
        n_detach = c.execute(
            "SELECT count(*) FROM track_audio_correction WHERE axis='recording' AND action='detach'"
        ).fetchone()[0]
    assert n_taudio == 1
    assert n_detach == 0
