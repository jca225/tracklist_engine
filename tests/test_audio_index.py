"""Tests for track_audio_id local audio index."""

from __future__ import annotations

import json
from pathlib import Path

from labeling.audio_index import (
    build_audio_index,
    load_audio_index,
    lookup_ref,
    lookup_stem,
    refresh_audio_index,
    write_audio_index,
)
from workspaces.alignment_prototype.stem_resolve import resolve_stem


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_build_prefers_manifest_paths(tmp_path: Path) -> None:
    local = _touch(tmp_path / "tracks" / "001__Song.m4a")
    vocals = _touch(tmp_path / "stems" / "001__Song" / "vocals.flac")
    tracks = [
        {
            "track_audio_id": 42,
            "slot_label": "001",
            "local_path": str(local),
            "stems": {"vocals": str(vocals)},
        }
    ]
    index = build_audio_index(tmp_path, tracks)
    assert index["by_track_audio_id"]["42"]["local_path"] == str(local)
    assert index["by_track_audio_id"]["42"]["stems"]["vocals"] == str(vocals)


def test_build_fills_unique_disk_fallback(tmp_path: Path) -> None:
    fallback = _touch(tmp_path / "tracks" / "001__Song.m4a")
    stem = _touch(tmp_path / "stems" / "001__Song" / "vocals.flac")
    tracks = [
        {
            "track_audio_id": 42,
            "slot_label": "001",
            "local_path": str(tmp_path / "missing.m4a"),
            "stems": {},
        }
    ]
    index = build_audio_index(tmp_path, tracks)
    assert index["by_track_audio_id"]["42"]["local_path"] == str(fallback)
    assert index["by_track_audio_id"]["42"]["stems"]["vocals"] == str(stem)


def test_build_omits_ambiguous_slot_candidates(tmp_path: Path) -> None:
    _touch(tmp_path / "tracks" / "001__Song.m4a")
    _touch(tmp_path / "tracks" / "001__Song [128bpm 1B].m4a")
    tracks = [
        {
            "track_audio_id": 42,
            "slot_label": "001",
            "local_path": str(tmp_path / "missing.m4a"),
            "stems": {},
        }
    ]
    index = build_audio_index(tmp_path, tracks)
    assert "42" not in index["by_track_audio_id"]


def test_resolve_stem_uses_audio_index(tmp_path: Path) -> None:
    indexed = _touch(tmp_path / "stems" / "indexed" / "vocals.flac")
    _touch(tmp_path / "stems" / "001w1__A" / "vocals.flac")
    _touch(tmp_path / "stems" / "001w1__B" / "vocals.flac")
    write_audio_index(
        tmp_path,
        {
            "version": 1,
            "by_track_audio_id": {
                "42": {"local_path": None, "stems": {"vocals": str(indexed)}}
            },
        },
    )
    track = {"track_audio_id": 42, "stems": {}}
    assert resolve_stem(tmp_path, "001w1", track, "vocals") == indexed


def test_lookup_helpers_and_refresh_roundtrip(tmp_path: Path) -> None:
    local = _touch(tmp_path / "tracks" / "002__Song.m4a")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "set_id": "1fsnxchk",
                "tracks": [
                    {
                        "track_id": "rec7",
                        "track_audio_id": 7,
                        "slot_label": "002",
                        "local_path": str(local),
                        "stems": {},
                    }
                ]
            }
        )
    )
    path = refresh_audio_index(tmp_path)
    assert path.name == "audio_index.json"
    index = load_audio_index(tmp_path)
    assert lookup_ref(index, 7) == local
    assert lookup_stem(index, 7, "vocals") is None
