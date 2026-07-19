from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from labeling.write_back_ground_truth import write_back


_REPO = Path(__file__).resolve().parents[2]
_BB11 = _REPO / "labeling" / "fixtures" / "bb11_ground_truth.yaml"


def _make_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE set_ground_truth (
                set_id TEXT NOT NULL,
                label TEXT NOT NULL,
                recording_id TEXT,
                claimed_stem TEXT,
                set_start_s REAL,
                set_end_s REAL,
                ref_start_s REAL,
                ref_end_s REAL,
                tempo_ratio REAL,
                pitch_shift_semi INTEGER,
                ref_source TEXT,
                is_loop INTEGER,
                ref_segments_json TEXT,
                media_links_json TEXT,
                source TEXT,
                PRIMARY KEY (set_id, label)
            )
            """
        )


def test_write_back_replaces_stale_rows_for_only_the_yaml_set(tmp_path: Path):
    db = tmp_path / "db.sqlite"
    _make_db(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO set_ground_truth (set_id, label) VALUES (?, ?)",
            ("2nvzlh2k", "stale-label"),
        )
        conn.execute(
            "INSERT INTO set_ground_truth (set_id, label) VALUES (?, ?)",
            ("other-set", "keep-me"),
        )

    assert write_back(db, _BB11, force_ungated=True) == 0

    with sqlite3.connect(db) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM set_ground_truth "
                "WHERE set_id='2nvzlh2k' AND label='stale-label'"
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM set_ground_truth WHERE set_id='2nvzlh2k'"
            ).fetchone()[0]
            > 100
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM set_ground_truth "
                "WHERE set_id='other-set' AND label='keep-me'"
            ).fetchone()[0]
            == 1
        )


def test_write_back_rolls_back_delete_when_insert_fails(tmp_path: Path):
    db = tmp_path / "db.sqlite"
    _make_db(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO set_ground_truth (set_id, label) VALUES (?, ?)",
            ("2nvzlh2k", "old-row"),
        )
        conn.execute(
            """
            CREATE TRIGGER reject_gt_insert
            BEFORE INSERT ON set_ground_truth
            BEGIN
              SELECT RAISE(ABORT, 'injected failure');
            END
            """
        )

    with pytest.raises(sqlite3.IntegrityError, match="injected failure"):
        write_back(db, _BB11, force_ungated=True)

    with sqlite3.connect(db) as conn:
        assert conn.execute(
            "SELECT label FROM set_ground_truth WHERE set_id='2nvzlh2k'"
        ).fetchall() == [("old-row",)]
