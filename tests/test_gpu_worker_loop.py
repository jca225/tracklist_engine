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


def _make_db(path, *, analyzed=(), tracks=(101, 102, 103), slots=(), media_links=()):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE track_audio (
            track_audio_id INTEGER PRIMARY KEY,
            track_id TEXT,
            path TEXT
        );
        CREATE TABLE track_analysis (track_audio_id INTEGER PRIMARY KEY);
        CREATE TABLE set_track_slots (set_id TEXT, track_id TEXT);
        CREATE TABLE dj_set_track_media_links (set_id TEXT, track_id TEXT);
        """
    )
    for tid in tracks:
        conn.execute(
            "INSERT INTO track_audio (track_audio_id, track_id, path) VALUES (?, ?, ?)",
            (tid, f"trk{tid}", f"/mnt/storage/objects/trk{tid}/a.flac"),
        )
    for tid in analyzed:
        conn.execute("INSERT INTO track_analysis (track_audio_id) VALUES (?)", (tid,))
    for set_id, tid in slots:
        conn.execute(
            "INSERT INTO set_track_slots (set_id, track_id) VALUES (?, ?)",
            (set_id, f"trk{tid}"),
        )
    for set_id, tid in media_links:
        conn.execute(
            "INSERT INTO dj_set_track_media_links (set_id, track_id) VALUES (?, ?)",
            (set_id, f"trk{tid}"),
        )
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


def test_set_filter_uses_canonical_spine_not_media_links(tmp_path):
    # Bug-audit C2: set-scope filtering must come from set_track_slots (the
    # canonical per-set spine), not dj_set_track_media_links (scrape-only). A
    # "gap" track — manually ingested audio present in the spine but with NO
    # scraped media link — was silently skipped under set filtering. Here 102 is
    # such a gap row: in the spine for set "S", absent from media links. It must
    # be selected; the old code (querying dj_set_track_media_links) returned None.
    db = tmp_path / "m.db"
    _make_db(
        db,
        slots=[("S", 102)],
        media_links=[("S", 999)],  # a different, non-ingested track
    )
    r = _next_unanalyzed(db, ("S",))
    assert r.is_ok()
    assert r.value is not None
    assert r.value[0] == 102


def test_set_filter_restricts_to_set(tmp_path):
    # A track in the spine of a DIFFERENT set is not selected.
    db = tmp_path / "m.db"
    _make_db(db, slots=[("OTHER", 101), ("S", 103)])
    r = _next_unanalyzed(db, ("S",))
    assert r.is_ok()
    assert r.value is not None
    assert r.value[0] == 103
