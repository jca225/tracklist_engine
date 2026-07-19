from __future__ import annotations

import json
import time
from pathlib import Path

from labeling.gt_release_gate import (
    partition_failures,
    sha256_file,
    stamp_path,
    verify_fixture_stamp,
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


def test_partition_failures_acks_known_debt():
    failures = [
        {"status": "POSITION_MISMATCH", "slot": "012", "mix_start_s": 343.2},
        {"status": "WRONG_AUDIO", "slot": "099", "mix_start_s": 1.0},
    ]
    acks = [
        {"status": "POSITION_MISMATCH", "slot": "012", "mix_start_s": 343.25},
    ]
    unacked, acked = partition_failures(failures, acks)
    assert len(acked) == 1
    assert len(unacked) == 1
    assert unacked[0]["slot"] == "099"


def test_verify_stamp_accepts_matching_yaml(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "labeling.gt_release_gate.STAMP_DIR",
        tmp_path / "stamps",
    )
    monkeypatch.setattr(
        "labeling.gt_release_gate.FIXTURE_STAMP_DIR",
        tmp_path / "fixture_stamps",
    )
    als = tmp_path / "hand.als"
    als.write_bytes(b"als-bytes")
    path, committed = write_stamp(
        set_id="2nvzlh2k",
        yaml_path=_BB11,
        als_path=als,
        audit_summary={"n_ok": 1, "skipped": False},
        ack_mismatches=False,
    )
    assert path.is_file() and committed.is_file()
    ok, reason = verify_stamp(_BB11, set_id="2nvzlh2k")
    assert ok, reason
    ok2, reason2 = verify_fixture_stamp(_BB11)
    assert ok2, reason2


def test_verify_stamp_rejects_yaml_byte_change(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "labeling.gt_release_gate.STAMP_DIR",
        tmp_path / "stamps",
    )
    monkeypatch.setattr(
        "labeling.gt_release_gate.FIXTURE_STAMP_DIR",
        tmp_path / "fixture_stamps",
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
    assert "mismatch" in reason or "changed" in reason


def test_verify_stamp_rejects_expired(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "labeling.gt_release_gate.STAMP_DIR",
        tmp_path / "stamps",
    )
    monkeypatch.setattr(
        "labeling.gt_release_gate.FIXTURE_STAMP_DIR",
        tmp_path / "fixture_stamps",
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
    # Also expire the committed copy so fallback cannot succeed.
    from labeling.gt_release_gate import fixture_stamp_path

    c = fixture_stamp_path("2nvzlh2k")
    c.write_text(json.dumps(payload))
    ok, reason = verify_stamp(_BB11, set_id="2nvzlh2k", max_age_s=3600)
    assert not ok
    assert "expired" in reason


def test_write_back_refuses_without_stamp(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "labeling.gt_release_gate.STAMP_DIR",
        tmp_path / "empty_stamps",
    )
    monkeypatch.setattr(
        "labeling.gt_release_gate.FIXTURE_STAMP_DIR",
        tmp_path / "empty_fixture_stamps",
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


def test_committed_bb_fixture_stamps_match_repo_yaml():
    """Seeded stamps in the PR must stay in lockstep with fixture bytes."""
    for name in ("bb11_ground_truth.yaml", "bb12_ground_truth.yaml"):
        ok, reason = verify_fixture_stamp(_REPO / "labeling" / "fixtures" / name)
        assert ok, reason
