from __future__ import annotations

import sqlite3
from pathlib import Path
from ingest.matchability import has_matchable_features


def _fixture_db(tmp_path: Path) -> Path:
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE track_audio (track_audio_id INTEGER PRIMARY KEY, recording_id TEXT, stem TEXT);
        CREATE TABLE track_fingerprints (recording_id TEXT, stem TEXT);
        CREATE TABLE track_mert_measures (track_audio_id INTEGER);
        INSERT INTO track_audio VALUES (1, 'recA', 'regular');   -- fully matchable
        INSERT INTO track_audio VALUES (2, 'recB', 'regular');   -- fp only, no mert
        INSERT INTO track_audio VALUES (3, 'recC', 'regular');   -- nothing
        INSERT INTO track_fingerprints VALUES ('recA', 'regular');
        INSERT INTO track_fingerprints VALUES ('recB', 'regular');
        INSERT INTO track_mert_measures VALUES (1);
    """)
    conn.commit()
    conn.close()
    return db


def test_matchable_true_when_fp_and_mert_present(tmp_path):
    assert has_matchable_features(_fixture_db(tmp_path), 1) is True


def test_not_matchable_when_mert_missing(tmp_path):
    assert has_matchable_features(_fixture_db(tmp_path), 2) is False


def test_not_matchable_when_nothing(tmp_path):
    assert has_matchable_features(_fixture_db(tmp_path), 3) is False


def test_not_matchable_when_row_absent(tmp_path):
    assert has_matchable_features(_fixture_db(tmp_path), 999) is False
