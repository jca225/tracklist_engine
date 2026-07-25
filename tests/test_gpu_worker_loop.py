"""Poison-pill quarantine for `gpu_worker.py --loop`.

Regression guard for the infinite-spin class: a track with a track_audio row
but no track_analysis row is re-selected by `_next_unanalyzed` on every
iteration. When it always errors, the loop must be able to exclude it so the
queue drains instead of spinning on the same poison track forever (the bug
that burned 496 consecutive failures in the loop driver before it grew a skip set).

`analysis.gpu_worker` imports `.pipeline` (torch/librosa) at module load, so
this is skipped in the lightweight CI env — same pattern as the other
heavy-dep tests (test_fibers, test_trajectory_decode).
"""

from __future__ import annotations

import sqlite3

import pytest

pytest.importorskip("torch")  # gpu_worker -> pipeline -> torch

from analysis.gpu_worker import _next_unanalyzed  # noqa: E402


def _make_db(path, *, analyzed=(), tracks=(101, 102, 103)):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE track_audio (
            track_audio_id INTEGER PRIMARY KEY,
            track_id TEXT,
            path TEXT
        );
        CREATE TABLE track_analysis (track_audio_id INTEGER PRIMARY KEY);
        """
    )
    for tid in tracks:
        conn.execute(
            "INSERT INTO track_audio (track_audio_id, track_id, path) VALUES (?, ?, ?)",
            (tid, f"trk{tid}", f"/mnt/storage/objects/trk{tid}/a.flac"),
        )
    for tid in analyzed:
        conn.execute("INSERT INTO track_analysis (track_audio_id) VALUES (?)", (tid,))
    conn.commit()
    conn.close()


def test_next_unanalyzed_returns_lowest_unanalyzed(tmp_path):
    db = tmp_path / "m.db"
    _make_db(db)
    r = _next_unanalyzed(db)
    assert r.is_ok()
    assert r.value is not None
    assert r.value[0] == 101


def test_exclude_skips_poison_track(tmp_path):
    # 101 is a poison track (always errors, never gets a track_analysis row).
    # Without exclude it would be re-selected forever; with exclude the loop
    # advances to 102.
    db = tmp_path / "m.db"
    _make_db(db)
    r = _next_unanalyzed(db, None, frozenset({101}))
    assert r.is_ok()
    assert r.value is not None
    assert r.value[0] == 102


def test_all_excluded_drains_queue(tmp_path):
    # Every remaining track quarantined -> queue reports drained (None), so the
    # loop exits instead of spinning.
    db = tmp_path / "m.db"
    _make_db(db)
    r = _next_unanalyzed(db, None, frozenset({101, 102, 103}))
    assert r.is_ok()
    assert r.value is None


def test_exclude_composes_with_analyzed(tmp_path):
    # 101 already analyzed (excluded by the IS NULL join), 102 quarantined ->
    # next is 103.
    db = tmp_path / "m.db"
    _make_db(db, analyzed=(101,))
    r = _next_unanalyzed(db, None, frozenset({102}))
    assert r.is_ok()
    assert r.value is not None
    assert r.value[0] == 103
