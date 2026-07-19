import json

import pytest

from workspaces.alignment_prototype import lyrics_align


def test_long_audio_defaults_to_cpu(monkeypatch):
    monkeypatch.delenv("ALIGN_WHISPER_DEVICE", raising=False)
    assert lyrics_align._whisper_device(lyrics_align.LONG_AUDIO_CPU_S) == "cpu"


def test_whisper_device_override(monkeypatch):
    monkeypatch.setenv("ALIGN_WHISPER_DEVICE", "mps")
    assert lyrics_align._whisper_device(99_999) == "mps"

    monkeypatch.setenv("ALIGN_WHISPER_DEVICE", "bogus")
    with pytest.raises(ValueError, match="ALIGN_WHISPER_DEVICE"):
        lyrics_align._whisper_device(1)


def test_partial_cache_round_trip_is_atomic(tmp_path):
    final = tmp_path / "transcript.json"
    partial = lyrics_align._partial_cache_file(final)
    payload = {"next_t0": 600.0, "words": [{"w": "hello", "s": 1.0, "e": 1.5}]}

    lyrics_align._write_json_atomic(partial, payload)

    words, next_t0 = lyrics_align._load_partial(final)
    assert next_t0 == 600.0
    assert words == payload["words"]
    assert json.loads(partial.read_text()) == payload
    assert not (tmp_path / ".transcript.partial.json.tmp").exists()
