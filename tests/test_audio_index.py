"""Tests for track_audio_id local audio index."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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
    assert index["by_track_audio_id"]["42"]["local_sha256"]
    assert index["by_track_audio_id"]["42"]["recording_id"] is None
    assert index["by_track_audio_id"]["42"]["stems"]["vocals"] == str(vocals)
    assert index["by_track_audio_id"]["42"]["stem_sha256"]["vocals"]


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


def test_existing_index_fails_closed_instead_of_using_stale_manifest_stem(
    tmp_path: Path,
) -> None:
    stale = _touch(tmp_path / "stems" / "stale" / "vocals.flac")
    write_audio_index(
        tmp_path,
        {"version": 2, "by_track_audio_id": {"42": {"stems": {}}}},
    )
    track = {"track_audio_id": 42, "stems": {"vocals": str(stale)}}
    assert resolve_stem(tmp_path, "001w1", track, "vocals") is None


def test_trajectory_ref_resolution_uses_audio_index(tmp_path: Path) -> None:
    pytest.importorskip("librosa")
    from workspaces.alignment_prototype.trajectory.features import resolve_span_audio

    indexed = _touch(tmp_path / "tracks" / "indexed.m4a")
    stale = _touch(tmp_path / "tracks" / "stale.m4a")
    mix = _touch(tmp_path / "mix.m4a")
    write_audio_index(
        tmp_path,
        {
            "version": 2,
            "by_track_audio_id": {
                "42": {"local_path": str(indexed), "stems": {}}
            },
        },
    )
    audio = resolve_span_audio(
        tmp_path,
        {"rec42": {"track_audio_id": 42, "local_path": str(stale), "stems": {}}},
        {},
        mix,
        {"track_id": "rec42", "claimed_stem": "regular", "slot_label": "001"},
    )
    assert not isinstance(audio, str)
    assert audio.ref_path == indexed


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
