from __future__ import annotations

import struct
from pathlib import Path

from sensors.mert import run_mert, stub_embedding


def test_stub_mert_is_deterministic(tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF....WAVEfmt demo")
    a = stub_embedding(audio.read_bytes())
    b = stub_embedding(audio.read_bytes())
    assert a == b
    assert len(a) == 64 * 4
    assert struct.unpack("<f", a[:4])


def test_run_mert_writes_sidecar(tmp_path: Path) -> None:
    audio = tmp_path / "t.wav"
    audio.write_bytes(b"RIFF....WAVEfmt demo-bytes")
    out = run_mert(
        audio, tmp_path / "mert", audio_artifact_id="00000000-0000-4000-8000-000000000099"
    )
    assert out["ok"] is True
    assert out["stub"] is True
    assert Path(out["feature_path"]).is_file()
    assert Path(out["mert_json"]).is_file()
    assert out["metadata"]["audio_artifact_id"].endswith("099")
