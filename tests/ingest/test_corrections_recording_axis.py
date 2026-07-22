from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from core.result import Ok
from ingest.corrections import Correction, log_correction

# minimal standalone schema for the ledger table under test (post-migration shape)
_SCHEMA = """
CREATE TABLE track_audio_correction (
    correction_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    set_id TEXT, position TEXT, track_id TEXT NOT NULL,
    axis TEXT NOT NULL, action TEXT NOT NULL,
    old_track_audio_id INTEGER, old_platform TEXT, old_player_id TEXT, old_url TEXT,
    new_track_audio_id INTEGER, new_platform TEXT, new_player_id TEXT, new_url TEXT,
    old_recording_id TEXT, new_recording_id TEXT,
    stem_value TEXT, reason TEXT, source TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    CHECK (axis IN ('version','variant','stem','recording')),
    CHECK (action IN ('replace','add','relink','detach'))
);
"""


@pytest.fixture
def db(tmp_path: Path) -> Path:
    p = tmp_path / "t.db"
    with sqlite3.connect(p) as c:
        c.executescript(_SCHEMA)
    return p


def test_detach_correction_roundtrips(db: Path):
    c = Correction(
        track_id="42wv4vp",
        axis="recording",
        action="detach",
        set_id="1fsnxchk",
        position="148w1",
        old_recording_id="42wv4vp",
        new_recording_id=None,
        reason="title-token disjoint: 'Come On Over Baby' vs 'Good Time'",
        source="same_song_guard",
    )
    r = log_correction(db, c)
    assert isinstance(r, Ok)
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM track_audio_correction").fetchone()
    assert row["axis"] == "recording"
    assert row["action"] == "detach"
    assert row["old_recording_id"] == "42wv4vp"
    assert row["new_recording_id"] is None


def test_relink_correction_roundtrips(db: Path):
    c = Correction(
        track_id="abc123",
        axis="recording",
        action="relink",
        old_recording_id="42wv4vp",
        new_recording_id="9zzz000",
        source="remediation",
    )
    assert isinstance(log_correction(db, c), Ok)
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT new_recording_id FROM track_audio_correction"
        ).fetchone()
    assert row["new_recording_id"] == "9zzz000"
