"""Tests for the Essentia subprocess adapter.

The fast tests stub out subprocess.run and exercise the JSON parsing /
error mapping. The integration test (skipped unless ESSENTIA_INTEGRATION=1)
spawns the real venvs/essentia/ worker against a fixture audio file.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from audio_pipeline.analysis.adapters import essentia_adapter
from audio_pipeline.analysis.errors import EssentiaError
from audio_pipeline.analysis.models import EssentiaFeatures
from audio_pipeline.result import Err, Ok


def _completed(stdout: str, returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr,
    )


_SP_ONLY_PAYLOAD = {
    "version": "essentia_v2",
    "models_present": [],
    "key": {"tonic": "F#", "mode": "minor", "strength": 0.92, "profile": "edma"},
    "rhythm": {"bpm": 172.27, "n_beats": 583},
    "danceability_sp": 1.15,
}


_FULL_PAYLOAD = {
    **_SP_ONLY_PAYLOAD,
    "models_present": [
        "danceability_tf", "discogs_effnet", "emomusic", "mood_acoustic",
        "mood_aggressive", "mood_happy", "msd_musicnn", "voice_instrumental", "yamnet",
    ],
    "mood_happy": 0.13,
    "mood_acoustic": 0.02,
    "mood_aggressive": 0.42,
    "voice_prob": 0.46,
    "danceability_tf": 0.97,
    "valence_raw": 5.29, "arousal_raw": 5.59,
    "valence": 0.54, "arousal": 0.57,
    "yamnet": {
        "speech_mean": 0.001, "conversation_mean": 0.0001, "singing_mean": 0.0004,
        "cheering_max": 0.004, "applause_max": 0.004, "crowd_max": 0.009,
    },
    "speechiness": 0.001,
    "liveness": 0.009,
}


def test_parses_sp_only_when_models_absent(tmp_path: Path) -> None:
    audio = tmp_path / "fake.m4a"
    audio.write_bytes(b"x")
    with patch.object(essentia_adapter, "_ESSENTIA_PYTHON", Path(__file__)), \
         patch.object(subprocess, "run", return_value=_completed(json.dumps(_SP_ONLY_PAYLOAD))):
        result = essentia_adapter.analyze(audio, track_audio_id=7)
    assert isinstance(result, Ok)
    feat: EssentiaFeatures = result.value
    assert feat.version == "essentia_v2"
    assert feat.key_tonic == "F#"
    assert feat.danceability_sp == pytest.approx(1.15)
    # All TF-derived fields are None when no models are present.
    for missing in ("mood_happy", "valence", "arousal", "speechiness", "liveness", "voice_prob"):
        assert getattr(feat, missing) is None, missing


def test_parses_full_payload(tmp_path: Path) -> None:
    audio = tmp_path / "fake.m4a"
    audio.write_bytes(b"x")
    with patch.object(essentia_adapter, "_ESSENTIA_PYTHON", Path(__file__)), \
         patch.object(subprocess, "run", return_value=_completed(json.dumps(_FULL_PAYLOAD))):
        result = essentia_adapter.analyze(audio, track_audio_id=7)
    assert isinstance(result, Ok)
    feat: EssentiaFeatures = result.value
    assert feat.valence == pytest.approx(0.54)
    assert feat.arousal == pytest.approx(0.57)
    assert feat.speechiness == pytest.approx(0.001)
    assert feat.liveness == pytest.approx(0.009)
    assert feat.voice_prob == pytest.approx(0.46)
    assert "yamnet" in feat.models_present


def test_venv_missing_returns_err(tmp_path: Path) -> None:
    audio = tmp_path / "fake.m4a"
    audio.write_bytes(b"x")
    with patch.object(essentia_adapter, "_ESSENTIA_PYTHON", tmp_path / "nope"):
        result = essentia_adapter.analyze(audio, track_audio_id=1)
    assert isinstance(result, Err)
    assert result.error.kind == "venv_missing"


def test_audio_missing_returns_err(tmp_path: Path) -> None:
    with patch.object(essentia_adapter, "_ESSENTIA_PYTHON", Path(__file__)):
        result = essentia_adapter.analyze(tmp_path / "missing.m4a", track_audio_id=1)
    assert isinstance(result, Err)
    assert result.error.kind == "audio_missing"


def test_worker_error_payload_propagates(tmp_path: Path) -> None:
    audio = tmp_path / "fake.m4a"
    audio.write_bytes(b"x")
    err_payload = {"error": "RuntimeError: decode failed", "trace": "..."}
    with patch.object(essentia_adapter, "_ESSENTIA_PYTHON", Path(__file__)), \
         patch.object(subprocess, "run", return_value=_completed(json.dumps(err_payload), returncode=1)):
        result = essentia_adapter.analyze(audio, track_audio_id=1)
    assert isinstance(result, Err)
    assert result.error.kind == "worker_failed"
    assert "decode failed" in result.error.detail


def test_timeout_returns_err(tmp_path: Path) -> None:
    audio = tmp_path / "fake.m4a"
    audio.write_bytes(b"x")
    with patch.object(essentia_adapter, "_ESSENTIA_PYTHON", Path(__file__)), \
         patch.object(subprocess, "run", side_effect=subprocess.TimeoutExpired("x", 1)):
        result = essentia_adapter.analyze(audio, track_audio_id=1, timeout_s=1)
    assert isinstance(result, Err)
    assert result.error.kind == "timeout"


def test_bad_json_returns_err(tmp_path: Path) -> None:
    audio = tmp_path / "fake.m4a"
    audio.write_bytes(b"x")
    with patch.object(essentia_adapter, "_ESSENTIA_PYTHON", Path(__file__)), \
         patch.object(subprocess, "run", return_value=_completed("not json")):
        result = essentia_adapter.analyze(audio, track_audio_id=1)
    assert isinstance(result, Err)
    assert result.error.kind == "bad_json"


# Override via ESSENTIA_FIXTURE_AUDIO=/path/to/track.m4a once a canonical
# fixture exists (likely under /mnt/storage/objects/... once tracks are
# re-downloaded to pi-storage).
_INTEGRATION_AUDIO = Path(os.environ.get("ESSENTIA_FIXTURE_AUDIO", "/nonexistent"))


@pytest.mark.skipif(
    os.environ.get("ESSENTIA_INTEGRATION") != "1" or not _INTEGRATION_AUDIO.exists(),
    reason="set ESSENTIA_INTEGRATION=1 and ESSENTIA_FIXTURE_AUDIO=<path>",
)
def test_real_worker_against_real_audio() -> None:
    result = essentia_adapter.analyze(_INTEGRATION_AUDIO, track_audio_id=999)
    assert isinstance(result, Ok), getattr(result, "error", None)
    feat = result.value
    assert feat.version == "essentia_v2"
    assert feat.key_mode in ("major", "minor")
    assert 40 < feat.bpm < 250
    assert feat.n_beats > 50
    # All TF heads should populate when models are present.
    if "yamnet" in feat.models_present:
        assert feat.speechiness is not None and 0.0 <= feat.speechiness <= 1.0
        assert feat.liveness is not None and 0.0 <= feat.liveness <= 1.0
    if "emomusic" in feat.models_present:
        assert feat.valence is not None and 0.0 <= feat.valence <= 1.0
        assert feat.arousal is not None and 0.0 <= feat.arousal <= 1.0
