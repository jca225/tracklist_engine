from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ingest.stem_guard_runner import (
    RecordingContext,
    log_detach,
    recording_context,
    title_gate,
)
from ingest.same_song_guard import GuardVerdict

_SCHEMA = """
CREATE TABLE work (work_id TEXT PRIMARY KEY, title TEXT);
CREATE TABLE recording (recording_id TEXT PRIMARY KEY, work_id TEXT, full_name TEXT);
CREATE TABLE track_audio (
  track_audio_id INTEGER PRIMARY KEY, recording_id TEXT, stem TEXT, path TEXT,
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


@pytest.fixture
def db(tmp_path: Path) -> Path:
    p = tmp_path / "t.db"
    with sqlite3.connect(p) as c:
        c.executescript(_SCHEMA)
        c.execute("INSERT INTO work VALUES ('w1','Good Time')")
        c.execute(
            "INSERT INTO recording VALUES ('42wv4vp','w1','Owl City - Good Time')"
        )
        c.execute(
            "INSERT INTO track_audio VALUES (1,'42wv4vp','regular','/x/good.wav',1,NULL)"
        )
    return p


def test_recording_context_reads_title_and_regular_path(db: Path):
    ctx = recording_context(db, "42wv4vp")
    assert ctx == RecordingContext(
        title="Owl City - Good Time", regular_path="/x/good.wav"
    )


def test_recording_context_missing_recording(db: Path):
    ctx = recording_context(db, "nope")
    assert ctx.title == ""
    assert ctx.regular_path is None


def test_title_gate_refuses_disjoint():
    v = title_gate("Come On Over Baby", "Owl City - Good Time")
    assert v.accept is False and v.channel == "title"


def test_log_detach_writes_recording_correction(db: Path):
    v = GuardVerdict(
        False, "title", "title-token disjoint: 'Come On Over Baby' vs 'Good Time'"
    )
    log_detach(
        db,
        recording_id="42wv4vp",
        set_id="1fsnxchk",
        position="148w1",
        acquired_title="Come On Over Baby",
        verdict=v,
    )
    with sqlite3.connect(db) as c:
        c.row_factory = sqlite3.Row
        row = c.execute("SELECT * FROM track_audio_correction").fetchone()
    assert row["axis"] == "recording" and row["action"] == "detach"
    assert row["old_recording_id"] == "42wv4vp" and row["new_recording_id"] is None
    assert row["source"] == "same_song_guard"
