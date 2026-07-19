from __future__ import annotations

import json
import time
from pathlib import Path

from labeling.gt_release_gate import (
    sha256_file,
    stamp_path,
    verify_stamp,
    write_stamp,
)
from labeling.write_back_ground_truth import write_back


_REPO = Path(__file__).resolve().parents[2]
_BB11 = _REPO / "labeling" / "fixtures" / "bb11_ground_truth.yaml"


def test_sha256_stable(tmp_path: Path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello")
    assert sha256_file(p) == sha256_file(p)


def test_verify_stamp_accepts_matching_yaml(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "labeling.gt_release_gate.STAMP_DIR",
        tmp_path / "stamps",
    )
    als = tmp_path / "hand.als"
    als.write_bytes(b"als-bytes")
    path = write_stamp(
        set_id="2nvzlh2k",
        yaml_path=_BB11,
        als_path=als,
        audit_summary={"n_ok": 1, "skipped": False},
        ack_mismatches=False,
    )
    assert path.is_file()
    ok, reason = verify_stamp(_BB11, set_id="2nvzlh2k")
    assert ok, reason


def test_verify_stamp_rejects_yaml_byte_change(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "labeling.gt_release_gate.STAMP_DIR",
        tmp_path / "stamps",
    )
    yaml_copy = tmp_path / "gt.yaml"
    yaml_copy.write_bytes(_BB11.read_bytes())
    als = tmp_path / "hand.als"
    als.write_bytes(b"als")
    write_stamp(
        set_id="2nvzlh2k",
        yaml_path=yaml_copy,
        als_path=als,
        audit_summary={"n_ok": 1},
        ack_mismatches=False,
    )
    yaml_copy.write_bytes(yaml_copy.read_bytes() + b"\n# touched\n")
    ok, reason = verify_stamp(yaml_copy, set_id="2nvzlh2k")
    assert not ok
    assert "changed" in reason


def test_verify_stamp_rejects_expired(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "labeling.gt_release_gate.STAMP_DIR",
        tmp_path / "stamps",
    )
    als = tmp_path / "hand.als"
    als.write_bytes(b"als")
    write_stamp(
        set_id="2nvzlh2k",
        yaml_path=_BB11,
        als_path=als,
        audit_summary={"n_ok": 1},
        ack_mismatches=False,
    )
    stamp = stamp_path("2nvzlh2k")
    payload = json.loads(stamp.read_text())
    payload["created_unix"] = time.time() - 100_000
    stamp.write_text(json.dumps(payload))
    ok, reason = verify_stamp(_BB11, set_id="2nvzlh2k", max_age_s=3600)
    assert not ok
    assert "expired" in reason


def test_write_back_refuses_without_stamp(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "labeling.gt_release_gate.STAMP_DIR",
        tmp_path / "empty_stamps",
    )
    db = tmp_path / "db.sqlite"
    assert write_back(db, _BB11) == 3


def test_write_back_force_ungated_still_writes(tmp_path: Path):
    import sqlite3

    db = tmp_path / "db.sqlite"
    with sqlite3.connect(db) as conn:
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
    assert write_back(db, _BB11, force_ungated=True) == 0
    with sqlite3.connect(db) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM set_ground_truth WHERE set_id='2nvzlh2k'"
            ).fetchone()[0]
            > 100
        )
