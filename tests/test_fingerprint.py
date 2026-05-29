"""Tests for the chromaprint variant-identity adapter.

The similarity + classify logic is pure (synthetic fingerprints, deterministic).
The fpcalc round-trip is guarded by fpcalc availability.
"""
from __future__ import annotations

import shutil
import wave

import numpy as np
import pytest

from ingest.adapters import fingerprint as fp


def test_similarity_identical_is_one() -> None:
    a = np.arange(1, 201, dtype=np.uint32)
    assert fp.similarity(a, a) == 1.0


def test_similarity_recovers_shifted_alignment() -> None:
    # Varied (not monotonic) so a wrong offset wouldn't accidentally match.
    a = ((np.arange(1, 201, dtype=np.uint64) * np.uint64(2654435761)) % np.uint64(2**32)).astype(np.uint32)
    b = a[5:]   # b is a shifted by 5 frames
    assert fp.similarity(a, b) == 1.0


def test_similarity_random_near_half() -> None:
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2**32, size=200, dtype=np.uint64).astype(np.uint32)
    b = rng.integers(0, 2**32, size=200, dtype=np.uint64).astype(np.uint32)
    s = fp.similarity(a, b)
    assert 0.4 < s < 0.7   # unrelated 32-bit words agree on ~half their bits


def test_similarity_empty_is_zero() -> None:
    assert fp.similarity(np.array([], dtype=np.uint32), np.arange(5, dtype=np.uint32)) == 0.0


def test_classify_instrumental_bands() -> None:
    assert fp.classify("instrumental", 0.98, 1.00)[0] == "FALLBACK_TO_ORIGINAL"
    assert fp.classify("instrumental", 0.70, 1.00)[0] == "OK"
    assert fp.classify("instrumental", 0.40, 1.00)[0] == "WRONG_SONG"
    assert fp.classify("instrumental", 0.70, 0.50)[0] == "DURATION_MISMATCH"


def test_classify_acappella_is_weak_but_catches_fallback() -> None:
    assert fp.classify("acappella", 0.40, 1.00)[0] == "WEAK_SIGNAL"
    assert fp.classify("acappella", 0.98, 1.00)[0] == "FALLBACK_TO_ORIGINAL"


@pytest.mark.skipif(shutil.which("fpcalc") is None, reason="fpcalc not installed")
def test_fingerprint_file_real_roundtrip(tmp_path) -> None:
    sr = 22050
    rng = np.random.default_rng(1)
    sig = rng.integers(-8000, 8000, size=sr * 8).astype(np.int16)   # 8s of noise
    p = tmp_path / "a.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(sig.tobytes())

    r = fp.fingerprint_file(str(p))
    assert r.is_ok(), r
    res = r.value
    assert res.raw.size > 0 and res.duration_s > 0
    assert fp.similarity(res.raw, res.raw) == 1.0
