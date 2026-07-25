"""Brick 7 — the workspace feature producers over the provenance substrate.

GPU/network-free: exercises the declared registrations, the deterministic MERT
serialization, and the cache-migration path against a synthetic npz (the real
fingerprint compute + pi pull are exercised by the producers CLI, not here).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from core.provenance import (
    REGISTRY,
    ArtifactKind,
    ArtifactStore,
    ProvenanceRepository,
    connect,
)
from workspaces.alignment_prototype import producers
from workspaces.alignment_prototype.producers import (
    _serialize_mert_series,
    _sniff_suffix,
    migrate_cached_mert,
)


@pytest.fixture
def world(tmp_path):
    conn = connect(tmp_path)
    store = ArtifactStore(tmp_path, conn)
    repo = ProvenanceRepository(conn)
    return conn, store, repo


def test_producers_are_declared_in_the_registry():
    assert REGISTRY.producers_of("landmark_fingerprint") == ["analyze.landmark_fp"]
    assert REGISTRY.artifact_type("mert_features").media_type == (
        "application/x-mert-measures"
    )
    graph = REGISTRY.graph()
    assert graph["analyze.landmark_fp"]["inputs"] == {"audio": "audio"}
    assert graph["analyze.landmark_fp"]["outputs"] == {
        "features": "landmark_fingerprint"
    }


def test_sniff_suffix_recognizes_common_containers():
    assert _sniff_suffix(b"\x00\x00\x00\x20ftypM4A \x00" + b"\x00" * 8) == ".m4a"
    assert _sniff_suffix(b"ID3\x04" + b"\x00" * 12) == ".mp3"
    assert _sniff_suffix(b"RIFF\x00\x00\x00\x00WAVE") == ".wav"
    assert _sniff_suffix(b"fLaC" + b"\x00" * 12) == ".flac"
    assert _sniff_suffix(b"OggS" + b"\x00" * 12) == ".ogg"
    assert _sniff_suffix(b"garbage-bytes!!!") == ".audio"


def test_serialize_mert_series_is_deterministic_and_parseable():
    start = np.arange(4, dtype=np.float64)
    end = start + 1.0
    vec = np.ones((4, 8), dtype=np.float32)
    a = _serialize_mert_series(start, end, vec, layer=6)
    b = _serialize_mert_series(start, end, vec, layer=6)
    assert a == b  # byte-stable -> content-addressing is stable
    header = json.loads(a.split(b"\n", 1)[0])
    assert header["kind"] == "mert_features"
    assert header["layer"] == 6
    assert header["n_measures"] == 4 and header["dim"] == 8
    # payload round-trips
    body = a.split(b"\n", 1)[1]
    n = start.nbytes
    assert np.frombuffer(body[:n], dtype=np.float64).tolist() == start.tolist()


def _fake_mert_cache(tmp_path, set_id: str, recording_id: str):
    cache = tmp_path / f"{set_id}_mert.npz"
    np.savez(
        cache,
        set_audio_id=np.int64(1),
        mix_start=np.zeros(2),
        mix_end=np.ones(2),
        mix_vec=np.zeros((2, 8), dtype=np.float32),
        ref_ids=json.dumps([recording_id]),
        **{
            f"ref_{recording_id}_start": np.arange(3, dtype=np.float64),
            f"ref_{recording_id}_end": np.arange(3, dtype=np.float64) + 1.0,
            f"ref_{recording_id}_vec": np.full((3, 8), 0.5, dtype=np.float32),
        },
    )
    return cache


def test_migrate_cached_mert_registers_derived_versioned_feature(
    world, tmp_path, monkeypatch
):
    conn, store, repo = world
    cache = _fake_mert_cache(tmp_path, "fakeset", "rec1")
    monkeypatch.setattr(producers, "_cache_path", lambda set_id: cache)

    audio = store.put_bytes(
        b"audio-bytes",
        kind=ArtifactKind.AUDIO,
        media_type="audio/mp4",
        source_system="test",
    )
    feat = migrate_cached_mert(
        "rec1", "fakeset", audio, store=store, repo=repo, code_commit="abc1234"
    )
    assert feat is not None
    assert feat.kind == "mert_features"
    assert store.verify(feat.content_sha256)

    # Derived from the audio artifact by a ProcessSpec-versioned run whose
    # model_refs name the MERT model (the migration's whole point).
    der = conn.execute(
        "SELECT d.parent_sha256, d.relation, r.process_spec_id FROM derivation d "
        "JOIN run r ON r.run_id=d.run_id WHERE d.child_sha256=?",
        (feat.content_sha256,),
    ).fetchone()
    assert der["parent_sha256"] == audio.content_sha256
    assert der["relation"] == "migrated_from_cache"
    spec = conn.execute(
        "SELECT model_refs_json, code_commit FROM process_spec WHERE process_spec_id=?",
        (der["process_spec_id"],),
    ).fetchone()
    assert "m-a-p/MERT-v1-330M" in json.loads(spec["model_refs_json"])
    assert spec["code_commit"] == "abc1234"


def test_migrate_cached_mert_skips_missing_cache_and_recording(
    world, tmp_path, monkeypatch, capsys
):
    _, store, repo = world
    audio = store.put_bytes(
        b"a", kind=ArtifactKind.AUDIO, media_type="audio/mp4", source_system="test"
    )
    # missing cache file
    monkeypatch.setattr(
        producers, "_cache_path", lambda set_id: tmp_path / "absent.npz"
    )
    assert (
        migrate_cached_mert(
            "rec1", "nope", audio, store=store, repo=repo, code_commit="c"
        )
        is None
    )
    # cache present, recording absent
    cache = _fake_mert_cache(tmp_path, "fakeset", "other_rec")
    monkeypatch.setattr(producers, "_cache_path", lambda set_id: cache)
    assert (
        migrate_cached_mert(
            "rec1", "fakeset", audio, store=store, repo=repo, code_commit="c"
        )
        is None
    )
    out = capsys.readouterr().out
    assert "skipped" in out


def test_unsafe_recording_id_is_rejected_before_any_ssh(tmp_path):
    with pytest.raises(ValueError, match="not a safe identifier"):
        producers._pull_audio_from_pi("x'; DROP TABLE track_audio;--", tmp_path)
