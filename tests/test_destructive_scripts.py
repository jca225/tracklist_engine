"""Dry-run and apply tests for canonical-state mutators.

Uses real schema.sql fixtures on tmp paths — no pi-storage, no network.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from core import db as db_adapter
from core.models import AudioAsset
from core.result import Ok
from scripts import replace_track_audio as rta


_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent / "web_crawler" / "database" / "schema.sql"
)


@pytest.fixture
def canonical_env(tmp_path: Path) -> tuple[Path, Path]:
    """Fresh DB + audio_root under tmp_path."""
    db = tmp_path / "test.db"
    audio_root = tmp_path / "storage"
    conn = sqlite3.connect(db)
    conn.executescript(_SCHEMA_PATH.read_text())
    conn.commit()
    conn.close()
    return db, audio_root


def _write_object(
    audio_root: Path,
    track_id: str,
    name: str,
    content: bytes = b"audio-payload",
) -> Path:
    d = audio_root / "objects" / track_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_bytes(content)
    return p


def _insert_track(
    db: Path,
    audio_root: Path,
    track_id: str,
    *,
    platform: str = "youtube",
    player_id: str = "vid1",
    content: bytes = b"registered-audio",
) -> tuple[int, Path]:
    path = _write_object(
        audio_root,
        track_id,
        f"{track_id}__{platform}__{player_id}.m4a",
        content,
    )
    asset = AudioAsset(
        track_audio_id=None,
        track_id=track_id,
        platform=platform,
        source_url=f"https://example.com/{player_id}",
        player_id=player_id,
        path=str(path),
        sha256="abc",
        duration_s=200.0,
        sample_rate=44100,
        codec="m4a",
        bitrate_kbps=128,
    )
    r = db_adapter.insert_audio(db, asset)
    assert isinstance(r, Ok)
    return r.value, path


def _count_track_audio(db: Path) -> int:
    with sqlite3.connect(db) as conn:
        return conn.execute("SELECT COUNT(*) FROM track_audio").fetchone()[0]


def _count_corrections(db: Path) -> int:
    with sqlite3.connect(db) as conn:
        return conn.execute("SELECT COUNT(*) FROM track_audio_correction").fetchone()[0]


def _paths_on_disk(audio_root: Path) -> set[str]:
    root = audio_root / "objects"
    if not root.is_dir():
        return set()
    return {str(p) for p in root.rglob("*") if p.is_file()}


# ── replace_track_audio: file mode (no network) ─────────────────────────────


def test_replace_via_file_replaces_row_and_cascades_analysis(
    canonical_env: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    db, audio_root = canonical_env
    old_taid, old_path = _insert_track(db, audio_root, "REP01", player_id="old")

    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO track_analysis (track_audio_id, beat_times_json) VALUES (?, '[]')",
            (old_taid,),
        )
        conn.commit()

    new_src = tmp_path / "replacement.m4a"
    new_src.write_bytes(b"new-studio-master")

    rc = rta._replace_via_file(
        db,
        audio_root,
        "REP01",
        new_src,
        "manual_v2",
        old_taid,
        promote_reference=True,
        purge_siblings=False,
    )
    assert rc == 0
    assert not old_path.exists()

    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT track_audio_id, path, platform, player_id, is_reference "
            "FROM track_audio WHERE track_id='REP01'",
        ).fetchall()
        analysis = conn.execute("SELECT COUNT(*) FROM track_analysis").fetchone()[0]

    assert len(rows) == 1
    new_taid, new_path, platform, player_id, is_ref = rows[0]
    assert new_taid != old_taid
    assert platform == "manual"
    assert player_id == "manual_v2"
    assert is_ref == 1
    assert Path(new_path).is_file()
    assert analysis == 0


def test_replace_via_file_respects_no_promote(
    canonical_env: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    db, audio_root = canonical_env
    old_taid, _ = _insert_track(db, audio_root, "REP02", player_id="old")

    new_src = tmp_path / "replacement.m4a"
    new_src.write_bytes(b"alternate")

    rc = rta._replace_via_file(
        db,
        audio_root,
        "REP02",
        new_src,
        "manual_v3",
        old_taid,
        promote_reference=False,
        purge_siblings=False,
    )
    assert rc == 0

    with sqlite3.connect(db) as conn:
        is_ref = conn.execute(
            "SELECT is_reference FROM track_audio WHERE track_id='REP02'",
        ).fetchone()[0]
    assert is_ref == 0


def test_delete_old_row_if_exists_is_noop_when_missing(
    canonical_env: tuple[Path, Path],
) -> None:
    db, audio_root = canonical_env
    rta._delete_old_row_if_exists(db, audio_root, 99999)
    assert _count_track_audio(db) == 0


def _set_listed_duration(db: Path, track_id: str, seconds: int) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO set_track_slots (set_id, row_index, track_id, "
            "duration_seconds) VALUES (?, 0, ?, ?)",
            (f"set_{track_id}", track_id, seconds),
        )
        conn.commit()


def _regular_asset(track_id: str, path: str, dur: float | None) -> AudioAsset:
    return AudioAsset(
        track_audio_id=None,
        track_id=track_id,
        platform="manual",
        source_url="file:///x",
        player_id="p",
        path=path,
        sha256="z",
        duration_s=dur,
        sample_rate=None,
        codec="m4a",
        bitrate_kbps=None,
        stem="regular",
        variant="regular",
    )


def test_duration_gate_rejects_short_regular(
    canonical_env: tuple[Path, Path],
) -> None:
    # a 40s file for a song the tracklist lists at 240s = preview-clip class
    db, _ = canonical_env
    _set_listed_duration(db, "DGATE1", 240)
    assert rta._duration_gate(db, _regular_asset("DGATE1", "/x.m4a", 40.0)) is not None
    # a full-length file passes
    assert rta._duration_gate(db, _regular_asset("DGATE1", "/x.m4a", 235.0)) is None


def test_duration_gate_never_gates_stems(canonical_env: tuple[Path, Path]) -> None:
    # a legitimately-short acappella stab (DJ Kool 8.5s) must NOT be gated,
    # even against a long listed length
    db, _ = canonical_env
    _set_listed_duration(db, "DGATE2", 240)
    stab = replace(_regular_asset("DGATE2", "/x.m4a", 8.5), stem="acappella")
    assert rta._duration_gate(db, stab) is None


def test_duration_gate_allows_when_nothing_listed(
    canonical_env: tuple[Path, Path],
) -> None:
    # no scraped length -> nothing to contradict -> allow
    db, _ = canonical_env
    assert rta._duration_gate(db, _regular_asset("NOPE", "/x.m4a", 5.0)) is None


def test_insert_failure_preserves_old_row(
    canonical_env: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Insert-before-delete invariant: a failed insert must NOT delete the
    existing row (the delete-before-insert bug that ate ~23 references,
    e.g. DJ Kool - Let Me Clear My Throat's full original)."""
    from core.result import Err
    from core.errors import DbError

    db, audio_root = canonical_env
    old_taid, old_path = _insert_track(db, audio_root, "REP99", player_id="old")

    monkeypatch.setattr(
        rta.db_adapter,
        "insert_audio_or_reap",
        lambda *a, **k: Err(DbError(kind="insert_failed", detail="simulated")),
    )
    new_asset = AudioAsset(
        track_audio_id=None,
        track_id="REP99",
        platform="manual",
        source_url="file:///x",
        player_id="new",
        path=str(audio_root / "new.m4a"),
        sha256="z",
        duration_s=None,
        sample_rate=None,
        codec="m4a",
        bitrate_kbps=None,
        stem="regular",
        variant="regular",
    )
    rc = rta._insert_and_report(
        db,
        audio_root,
        new_asset,
        promote_reference=True,
        purge_siblings=False,
        old_track_audio_id=old_taid,
    )
    assert rc == 1
    # old row and its file survive
    assert _count_track_audio(db) == 1
    assert old_path.exists()
    with sqlite3.connect(db) as conn:
        surviving = conn.execute(
            "SELECT track_audio_id FROM track_audio WHERE track_id='REP99'"
        ).fetchone()[0]
    assert surviving == old_taid
