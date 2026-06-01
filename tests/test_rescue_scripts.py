"""Tests for scripts/rescue_common.py phase-2 FK cascade deletes."""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest

from core import db as db_adapter
from core.db import connect
from core.models import AudioAsset
from core.result import Ok
from scripts.rescue_common import phase2_replace

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "web_crawler" / "database" / "schema.sql"


@pytest.fixture
def canonical_env(tmp_path: Path) -> tuple[Path, Path]:
    db = tmp_path / "test.db"
    audio_root = tmp_path / "storage"
    conn = sqlite3.connect(db)
    conn.executescript(_SCHEMA_PATH.read_text())
    conn.commit()
    conn.close()
    return db, audio_root


@dataclass(frozen=True)
class _Candidate:
    track_id: str
    yt_track_audio_id: int
    yt_audio_path: str

    @property
    def needs_replace(self) -> bool:
        return True


@dataclass
class _Stats:
    phase2_replaced: int = 0
    phase2_skipped: int = 0
    phase2_failed: int = 0


def _insert_track(
    db: Path,
    audio_root: Path,
    track_id: str,
    *,
    platform: str = "youtube",
    player_id: str = "vid1",
    content: bytes = b"registered-audio",
) -> tuple[int, Path]:
    path = audio_root / "objects" / track_id / f"{track_id}__{platform}__{player_id}.m4a"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
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


def test_phase2_replace_cascades_analysis(canonical_env: tuple[Path, Path]) -> None:
    db_path, audio_root = canonical_env
    old_taid, old_path = _insert_track(
        db_path, audio_root, "tid1", content=b"x" * 200_000,
    )
    new_taid, new_path = _insert_track(
        db_path, audio_root, "tid1",
        platform="youtube_music", player_id="vid2",
        content=b"y" * 200_000,
    )

    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO track_analysis (track_audio_id, beat_times_json) VALUES (?, '[]')",
            (old_taid,),
        )
        conn.commit()

    stats = _Stats()
    log = logging.getLogger("test_rescue_scripts")
    candidate = _Candidate("tid1", old_taid, str(old_path))
    out = phase2_replace(
        (candidate,),
        {"tid1": new_taid},
        audio_root,
        db_path,
        stats,
        log,
        replacement_label="ytmusic",
    )
    assert out.phase2_replaced == 1
    assert not old_path.is_file()
    assert new_path.is_file()

    with connect(db_path) as conn:
        assert conn.execute(
            "SELECT 1 FROM track_audio WHERE track_audio_id = ?", (old_taid,),
        ).fetchone() is None
        assert conn.execute(
            "SELECT 1 FROM track_analysis WHERE track_audio_id = ?", (old_taid,),
        ).fetchone() is None
        assert conn.execute(
            "SELECT 1 FROM track_audio WHERE track_audio_id = ?", (new_taid,),
        ).fetchone() is not None
