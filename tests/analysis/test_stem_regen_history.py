"""C1 — stem-regen never-drop: hash existing stems before overwrite."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from core.models import AudioAsset
from ingest.content_history import SCHEMA, chain, record_stems_before_overwrite


def test_record_stems_before_overwrite(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    with sqlite3.connect(db) as c:
        c.executescript(SCHEMA)
    stem_dir = tmp_path / "stems" / "1"
    stem_dir.mkdir(parents=True)
    vocals = stem_dir / "vocals.flac"
    # Minimal bytes; file_sha256 only needs readable bytes. flac_pcm_md5 may
    # return None for non-FLAC-shaped files — that's fine for this unit test.
    vocals.write_bytes(b"fake-vocals-bytes" * 50)
    (stem_dir / "drums.flac").write_bytes(b"drums")  # must NOT be recorded

    n = record_stems_before_overwrite(
        db,
        stem_dir,
        recording_id="recA",
        variant="extended",
        parent_content_sha256="PARENT_SHA",
        track_audio_id=1,
        source="test",
    )
    assert n == 1
    ch = chain(db, "recA", "acappella", "extended", "separated", valid_only=True)
    assert len(ch) == 1
    assert ch[0]["parent_content_sha256"] == "PARENT_SHA"
    assert ch[0]["op"] == "re-separate"
    assert ch[0]["content_sha256"]  # hashed


def test_record_skips_without_parent_hash(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    with sqlite3.connect(db) as c:
        c.executescript(SCHEMA)
    stem_dir = tmp_path / "stems"
    stem_dir.mkdir()
    (stem_dir / "vocals.flac").write_bytes(b"x")
    assert (
        record_stems_before_overwrite(
            db,
            stem_dir,
            recording_id="recA",
            variant="regular",
            parent_content_sha256=None,
        )
        == 0
    )


def test_maybe_record_via_pipeline_env(tmp_path: Path, monkeypatch) -> None:
    """analyze_track path: TRACKLIST_CONTENT_HISTORY_DB wires the hook."""
    from analysis.pipeline import _maybe_record_stem_regen

    db = tmp_path / "t.db"
    with sqlite3.connect(db) as c:
        c.executescript(SCHEMA)
    monkeypatch.setenv("TRACKLIST_CONTENT_HISTORY_DB", str(db))

    stem_root = tmp_path / "stems"
    dest = stem_root / "42"
    dest.mkdir(parents=True)
    (dest / "instrumental.flac").write_bytes(b"instr-bytes" * 40)

    asset = AudioAsset(
        track_audio_id=42,
        track_id="recZ",
        platform="youtube",
        source_url="",
        player_id="",
        path="/tmp/x.m4a",
        sha256="PARENT_Z",
        duration_s=None,
        sample_rate=None,
        codec=None,
        bitrate_kbps=None,
        stem="regular",
        variant="regular",
    )
    _maybe_record_stem_regen(asset, stem_root)
    ch = chain(db, "recZ", "instrumental", "regular", "separated")
    assert len(ch) == 1
    assert ch[0]["parent_content_sha256"] == "PARENT_Z"
