"""Brick 8 — Whisper + HuBERT as first-class registered producer types.

Three layers, honest about what each grounds:

- Declarations: the registry graph lists ``analyze.whisper`` and
  ``analyze.hubert`` with their declared kinds/model_refs (always runs).
- Plumbing: each producer runs through ``Registry.run`` with the MODEL CALL
  monkeypatched — proving the full provenance path (derivation edge,
  ProcessSpec, model_refs) without loading a model (always runs).
- Real model: ``analyze.hubert`` runs the REAL facebook/hubert-base-ls960 on a
  SHORT real audio slice pulled by the backfill runner — skipped with the
  documented reason when no pulled audio / model is available (e.g. CI).
  Whisper's real-model run is exercised by the backfill CLI (large-v3-turbo is
  too slow for the default test suite); opt in via RUN_WHISPER_PRODUCER_TEST=1.
"""

from __future__ import annotations

import io
import json
import os

import numpy as np
import pytest

from core.provenance import (
    REGISTRY,
    ArtifactKind,
    ArtifactStore,
    ProvenanceRepository,
    connect,
)
from alignment import producers
from alignment.producers import _serialize_hubert_features

_CENTRAL_PULLED = (
    producers._OUT_ROOT.parent / "central" / ".pulled"
)  # backfill runner's read-only pi pulls


@pytest.fixture
def world(tmp_path):
    conn = connect(tmp_path)
    store = ArtifactStore(tmp_path, conn)
    repo = ProvenanceRepository(conn)
    return conn, store, repo


# --- declarations ------------------------------------------------------------
def test_whisper_and_hubert_are_declared_in_the_registry():
    graph = REGISTRY.graph()
    assert graph["analyze.whisper"] == {
        "inputs": {"audio": "audio"},
        "outputs": {"transcript": "whisper_transcript"},
    }
    assert graph["analyze.hubert"] == {
        "inputs": {"audio": "audio"},
        "outputs": {"features": "hubert_features"},
    }
    assert REGISTRY.producers_of("whisper_transcript") == ["analyze.whisper"]
    assert REGISTRY.producers_of("hubert_features") == ["analyze.hubert"]
    assert REGISTRY.artifact_type("whisper_transcript").media_type == (
        "application/json"
    )
    assert REGISTRY.artifact_type("hubert_features").media_type == (
        "application/x-hubert-frames"
    )
    assert REGISTRY.transformation("analyze.whisper").model_refs == (
        "openai/whisper-large-v3-turbo",
    )
    assert REGISTRY.transformation("analyze.hubert").model_refs == (
        "facebook/hubert-base-ls960",
        "layer=9",
    )


def test_hubert_serialization_is_deterministic_and_parseable():
    feat = np.linspace(0, 1, 768 * 5, dtype=np.float32).reshape(768, 5)
    a = _serialize_hubert_features(feat, layer=9)
    b = _serialize_hubert_features(feat, layer=9)
    assert a == b  # byte-stable -> content-addressing is stable
    header = json.loads(a.split(b"\n", 1)[0])
    assert header["kind"] == "hubert_features"
    assert header["model"] == "facebook/hubert-base-ls960"
    assert header["dim"] == 768 and header["n_frames"] == 5
    body = a.split(b"\n", 1)[1]
    assert np.frombuffer(body, dtype=np.float32).reshape(768, 5)[0, 0] == feat[0, 0]


# --- plumbing (model call faked; provenance path real) -----------------------
def _run_with_provenance(world, name, out_role):
    conn, store, repo = world
    audio = store.put_bytes(
        b"RIFF0000WAVEfake-audio",
        kind=ArtifactKind.AUDIO,
        media_type="audio/wav",
        source_system="test",
    )
    out = REGISTRY.run(
        name, {"audio": audio}, store=store, repo=repo, code_commit="test0000"
    )[out_role]
    der = conn.execute(
        "SELECT d.parent_sha256, d.relation, r.process_spec_id FROM derivation d "
        "JOIN run r ON r.run_id=d.run_id WHERE d.child_sha256=?",
        (out.content_sha256,),
    ).fetchone()
    assert der["parent_sha256"] == audio.content_sha256
    assert der["relation"] == name
    spec = conn.execute(
        "SELECT model_refs_json FROM process_spec WHERE process_spec_id=?",
        (der["process_spec_id"],),
    ).fetchone()
    return out, json.loads(spec["model_refs_json"])


def test_whisper_producer_full_provenance_with_faked_model(world, monkeypatch):
    from alignment import lyrics_align

    words = [{"w": "hello", "s": 0.5, "e": 0.9}, {"w": "world", "s": 1.0, "e": 1.4}]
    monkeypatch.setattr(
        lyrics_align, "transcribe_words", lambda path, enhance=None: words
    )
    out, model_refs = _run_with_provenance(world, "analyze.whisper", "transcript")
    assert out.kind == "whisper_transcript"
    assert "openai/whisper-large-v3-turbo" in model_refs
    _, store, _ = world
    with store.open(out.content_sha256) as f:
        payload = json.loads(f.read())
    assert payload["words"] == words and payload["n_words"] == 2


def test_hubert_producer_full_provenance_with_faked_model(world, monkeypatch):
    from alignment import stem_placement

    feat = np.full((768, 7), 0.5, dtype=np.float32)
    monkeypatch.setattr(stem_placement, "hubert_of", lambda path, layer=9: feat)
    out, model_refs = _run_with_provenance(world, "analyze.hubert", "features")
    assert out.kind == "hubert_features"
    assert "facebook/hubert-base-ls960" in model_refs
    _, store, _ = world
    with store.open(out.content_sha256) as f:
        header = json.loads(f.read().split(b"\n", 1)[0])
    assert header["dim"] == 768 and header["n_frames"] == 7


def test_hubert_none_result_fails_the_run_honestly(world, monkeypatch):
    from alignment import stem_placement

    conn, store, repo = world
    monkeypatch.setattr(stem_placement, "hubert_of", lambda path, layer=9: None)
    audio = store.put_bytes(
        b"x", kind=ArtifactKind.AUDIO, media_type="audio/wav", source_system="test"
    )
    with pytest.raises(RuntimeError, match="hubert_of returned None"):
        REGISTRY.run(
            "analyze.hubert",
            {"audio": audio},
            store=store,
            repo=repo,
            code_commit="test0000",
        )
    assert conn.execute("SELECT status FROM run").fetchone()["status"] == "failed"


# --- real model on a short real slice ---------------------------------------
def _real_slice_wav(dur_s: float) -> bytes | None:
    """A few seconds of REAL pulled audio as wav bytes, or None if absent."""
    if not _CENTRAL_PULLED.is_dir():
        return None
    candidates = sorted(
        p
        for p in _CENTRAL_PULLED.iterdir()
        if p.suffix.lower() in (".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus")
    )
    if not candidates:
        return None
    import librosa
    import soundfile as sf

    y, sr = librosa.load(str(candidates[0]), sr=None, mono=True, duration=dur_s)
    buf = io.BytesIO()
    sf.write(buf, y, int(sr), format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_hubert_producer_runs_real_model_on_short_real_slice(world):
    wav = _real_slice_wav(3.0)
    if wav is None:
        pytest.skip(
            "no real pulled audio under out/provenance/central/.pulled "
            "(run provenance_backfill against pi-storage first); the real-model "
            "hubert grounding also runs via the backfill CLI"
        )
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    conn, store, repo = world
    audio = store.put_bytes(
        wav, kind=ArtifactKind.AUDIO, media_type="audio/wav", source_system="test"
    )
    try:
        out = REGISTRY.run(
            "analyze.hubert",
            {"audio": audio},
            store=store,
            repo=repo,
            code_commit="test0000",
        )["features"]
    except OSError as exc:  # model weights unavailable (offline CI)
        pytest.skip(f"hubert model not loadable here: {exc}")
    assert store.verify(out.content_sha256)
    with store.open(out.content_sha256) as f:
        header = json.loads(f.read().split(b"\n", 1)[0])
    assert header["dim"] == 768 and header["n_frames"] > 0
    assert conn.execute("SELECT status FROM run").fetchone()["status"] == "succeeded"


@pytest.mark.skipif(
    os.environ.get("RUN_WHISPER_PRODUCER_TEST") != "1",
    reason="whisper-large-v3-turbo load+decode adds minutes to every test run; "
    "the real-model whisper grounding runs via the backfill CLI "
    "(set RUN_WHISPER_PRODUCER_TEST=1 to run it here too)",
)
def test_whisper_producer_runs_real_model_on_short_real_slice(world):
    wav = _real_slice_wav(10.0)
    if wav is None:
        pytest.skip("no real pulled audio under out/provenance/central/.pulled")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    _, store, repo = world
    audio = store.put_bytes(
        wav, kind=ArtifactKind.AUDIO, media_type="audio/wav", source_system="test"
    )
    out = REGISTRY.run(
        "analyze.whisper",
        {"audio": audio},
        store=store,
        repo=repo,
        code_commit="test0000",
    )["transcript"]
    with store.open(out.content_sha256) as f:
        payload = json.loads(f.read())
    assert payload["model"] == "openai/whisper-large-v3-turbo"
    assert isinstance(payload["words"], list)
