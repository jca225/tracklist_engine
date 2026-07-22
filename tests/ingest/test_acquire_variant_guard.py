from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pytest

from scripts import acquire_variant as av
from scripts import replace_track_audio as rta
from ingest.stem_guard_runner import recording_context

_SCHEMA = """
CREATE TABLE work (work_id TEXT PRIMARY KEY, title TEXT);
CREATE TABLE recording (recording_id TEXT PRIMARY KEY, work_id TEXT, full_name TEXT);
CREATE TABLE track_audio (
  track_audio_id INTEGER PRIMARY KEY, recording_id TEXT, stem TEXT, path TEXT,
  platform TEXT, player_id TEXT, source_url TEXT,
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
    return p


def test_pretitle_guard_refuses_and_logs_detach(db: Path):
    """Runner-level contract (Task 4): title_gate + log_detach behave honestly
    when driven by hand. Locks the contract the wiring below must honor."""
    ctx = recording_context(db, "42wv4vp")
    # simulate the pre-download title-channel decision the ingest performs
    from ingest.stem_guard_runner import log_detach, title_gate

    v = title_gate("Come On Over Baby (All I Want Is You)", ctx.title)
    assert v.accept is False
    log_detach(
        db,
        recording_id="42wv4vp",
        set_id="1fsnxchk",
        position="148w1",
        acquired_title="Come On Over Baby (All I Want Is You)",
        verdict=v,
    )

    with sqlite3.connect(db) as c:
        n_audio = c.execute("SELECT count(*) FROM track_audio").fetchone()[0]
        n_corr = c.execute(
            "SELECT count(*) FROM track_audio_correction WHERE axis='recording' AND action='detach'"
        ).fetchone()[0]
    assert n_audio == 0  # nothing attached
    assert n_corr == 1  # one honest detach logged


def _canonical_args(
    db: Path, *, url: str, audio_root: Path, force: bool = False
) -> argparse.Namespace:
    """Build the Namespace canonical_ingest expects, bypassing argparse
    (mirrors what main() hands it in canonical mode)."""
    return argparse.Namespace(
        url=url,
        role="acappella",
        name=None,
        slot=None,
        dest=None,
        track_id="42wv4vp",
        track_audio_id=None,
        file=None,
        player_id=None,
        force=force,
        acquired_title=None,
        db=db,
        audio_root=audio_root,
        set_id="1fsnxchk",
        reason=None,
        no_log=True,
        promote_reference=False,
        no_promote_reference=False,
    )


def test_canonical_ingest_wiring_refuses_on_title_mismatch(
    db: Path, tmp_path: Path, monkeypatch
):
    """Wiring-level test: canonical_ingest itself (not just the runner
    functions) must refuse pre-download when the probed source title is a
    different song, write NO track_audio row, and log exactly one
    axis='recording' action='detach' correction. No network/download may run."""
    monkeypatch.setattr(
        "ingest.stem_guard_runner.probe_url_title",
        lambda url, yt_dlp, **kw: "Come On Over Baby",
    )

    def _boom(*a, **kw):
        raise AssertionError("insert path must not run when the title gate refuses")

    monkeypatch.setattr(rta, "_replace_via_url", _boom)
    monkeypatch.setattr(rta, "_replace_via_file", _boom)

    args = _canonical_args(
        db, url="https://example.com/come-on-over-baby", audio_root=tmp_path
    )
    rc = av.canonical_ingest(args)

    assert rc == 3
    with sqlite3.connect(db) as c:
        n_audio = c.execute("SELECT count(*) FROM track_audio").fetchone()[0]
        n_corr = c.execute(
            "SELECT count(*) FROM track_audio_correction "
            "WHERE axis='recording' AND action='detach'"
        ).fetchone()[0]
    assert n_audio == 0
    assert n_corr == 1


def test_canonical_ingest_wiring_force_overrides_title_gate(
    db: Path, tmp_path: Path, monkeypatch
):
    """--force must bypass the pre-download refuse and let the insert
    proceed (loud warning, but not blocked)."""
    monkeypatch.setattr(
        "ingest.stem_guard_runner.probe_url_title",
        lambda url, yt_dlp, **kw: "Come On Over Baby",
    )
    called = {}

    def _fake_replace_via_url(db_path, audio_root, track_id, url, **kw):
        called["hit"] = True
        with sqlite3.connect(db_path) as c:
            c.execute(
                "INSERT INTO track_audio (recording_id, stem, path, is_reference) "
                "VALUES (?, ?, ?, 0)",
                (track_id, kw.get("stem", "acappella"), "/x/candidate.wav"),
            )
        return 0

    monkeypatch.setattr(rta, "_replace_via_url", _fake_replace_via_url)

    args = _canonical_args(
        db,
        url="https://example.com/come-on-over-baby",
        audio_root=tmp_path,
        force=True,
    )
    rc = av.canonical_ingest(args)

    assert called.get("hit") is True
    assert rc == 0
    with sqlite3.connect(db) as c:
        n_audio = c.execute("SELECT count(*) FROM track_audio").fetchone()[0]
    assert n_audio == 1


def test_canonical_ingest_wiring_reaps_row_on_content_refuse(
    db: Path, tmp_path: Path, monkeypatch
):
    """Post-insert content gate: the title channel passes (overlapping
    titles), but the content channel refuses -> the just-inserted row must be
    reaped (deleted + file unlinked) and a detach correction logged."""
    monkeypatch.setattr(
        "ingest.stem_guard_runner.probe_url_title",
        lambda url, yt_dlp, **kw: "Good Time",
    )

    inserted_path = tmp_path / "candidate.wav"
    inserted_path.write_bytes(b"fake")

    def _fake_replace_via_url(db_path, audio_root, track_id, url, **kw):
        with sqlite3.connect(db_path) as c:
            c.execute(
                "INSERT INTO track_audio (recording_id, stem, path, is_reference) "
                "VALUES (?, ?, ?, 0)",
                (track_id, kw.get("stem", "acappella"), str(inserted_path)),
            )
        return 0

    monkeypatch.setattr(rta, "_replace_via_url", _fake_replace_via_url)

    from ingest.same_song_guard import GuardVerdict

    monkeypatch.setattr(
        "ingest.stem_guard_runner.content_gate",
        lambda stem_axis, regular_path, candidate_path: GuardVerdict(
            False, "content", "WRONG_SONG: fake mismatch for test"
        ),
    )

    args = _canonical_args(db, url="https://example.com/good-time", audio_root=tmp_path)
    rc = av.canonical_ingest(args)

    assert rc == 3
    assert not inserted_path.exists()  # reaped
    with sqlite3.connect(db) as c:
        n_audio = c.execute("SELECT count(*) FROM track_audio").fetchone()[0]
        n_corr = c.execute(
            "SELECT count(*) FROM track_audio_correction "
            "WHERE axis='recording' AND action='detach'"
        ).fetchone()[0]
    assert n_audio == 0
    assert n_corr == 1


def test_canonical_ingest_reaps_newly_inserted_row_not_stale_reference_sibling(
    db: Path, tmp_path: Path, monkeypatch
):
    """CRITICAL regression (finding #1): when a promoted is_reference=1
    sibling already exists for (recording_id, stem), the post-insert content
    gate + reap must act on the row THIS call just inserted — not whichever
    row `is_reference DESC, downloaded_at DESC` ordering prefers.

    Constructed so it FAILS against the old `_lookup_audio_path` ordering
    (which would reap the pre-existing GOOD row — it has is_reference=1 — and
    leave the newly-inserted BAD row behind) and PASSES once the gate/reap
    target the actually-inserted row (highest track_audio_id)."""
    good_path = tmp_path / "good_preexisting.wav"
    good_path.write_bytes(b"good")
    with sqlite3.connect(db) as c:
        c.execute(
            "INSERT INTO track_audio "
            "(recording_id, stem, path, is_reference, downloaded_at) "
            "VALUES (?, 'acappella', ?, 1, '2020-01-01 00:00:00')",
            ("42wv4vp", str(good_path)),
        )

    monkeypatch.setattr(
        "ingest.stem_guard_runner.probe_url_title",
        lambda url, yt_dlp, **kw: "Good Time",
    )

    bad_path = tmp_path / "bad_new_insert.wav"
    bad_path.write_bytes(b"bad")

    def _fake_replace_via_url(db_path, audio_root, track_id, url, **kw):
        # is_reference=0, downloaded_at NULL — loses to the pre-existing
        # is_reference=1 row under the OLD ordering, but its AUTOINCREMENT id
        # is the newest.
        with sqlite3.connect(db_path) as c:
            c.execute(
                "INSERT INTO track_audio (recording_id, stem, path, is_reference) "
                "VALUES (?, ?, ?, 0)",
                (track_id, kw.get("stem", "acappella"), str(bad_path)),
            )
        return 0

    monkeypatch.setattr(rta, "_replace_via_url", _fake_replace_via_url)

    from ingest.same_song_guard import GuardVerdict

    monkeypatch.setattr(
        "ingest.stem_guard_runner.content_gate",
        lambda stem_axis, regular_path, candidate_path: GuardVerdict(
            False, "content", "WRONG_SONG: fake mismatch for test"
        ),
    )

    args = _canonical_args(db, url="https://example.com/good-time", audio_root=tmp_path)
    rc = av.canonical_ingest(args)

    assert rc == 3
    assert not bad_path.exists()  # the NEWLY-INSERTED bad row was reaped
    assert good_path.exists()  # the pre-existing GOOD row survives
    with sqlite3.connect(db) as c:
        rows = c.execute(
            "SELECT path FROM track_audio WHERE recording_id='42wv4vp' AND stem='acappella'"
        ).fetchall()
    assert {r[0] for r in rows} == {str(good_path)}


def test_canonical_ingest_accept_path_logs_stem_add_correction(
    db: Path, tmp_path: Path, monkeypatch
):
    """IMPORTANT #4: the existing wiring tests hard-code no_log=True, so the
    accept-path `_log_to_ledger` call is untested. Drive the accept path
    (title overlaps, content accepts) with no_log=False and assert exactly
    one axis='stem' action='add' correction is logged."""
    monkeypatch.setattr(
        "ingest.stem_guard_runner.probe_url_title",
        lambda url, yt_dlp, **kw: "Good Time",
    )

    inserted_path = tmp_path / "candidate_good.wav"
    inserted_path.write_bytes(b"fake")

    def _fake_replace_via_url(db_path, audio_root, track_id, url, **kw):
        with sqlite3.connect(db_path) as c:
            c.execute(
                "INSERT INTO track_audio "
                "(recording_id, stem, path, platform, player_id, source_url, is_reference) "
                "VALUES (?, ?, ?, 'youtube', 'good1', 'http://good', 0)",
                (track_id, kw.get("stem", "acappella"), str(inserted_path)),
            )
        return 0

    monkeypatch.setattr(rta, "_replace_via_url", _fake_replace_via_url)

    from ingest.same_song_guard import GuardVerdict

    monkeypatch.setattr(
        "ingest.stem_guard_runner.content_gate",
        lambda stem_axis, regular_path, candidate_path: GuardVerdict(
            True, None, "accept"
        ),
    )

    args = _canonical_args(db, url="https://example.com/good-time", audio_root=tmp_path)
    args.no_log = False
    args.reason = "test accept-path logging"

    rc = av.canonical_ingest(args)

    assert rc == 0
    with sqlite3.connect(db) as c:
        n_audio = c.execute("SELECT count(*) FROM track_audio").fetchone()[0]
        c.row_factory = sqlite3.Row
        corr = c.execute("SELECT * FROM track_audio_correction").fetchall()
    assert n_audio == 1  # accepted row survives, not reaped
    assert len(corr) == 1
    assert corr[0]["axis"] == "stem" and corr[0]["action"] == "add"
    assert corr[0]["source"] == "acquire_variant"
