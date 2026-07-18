from __future__ import annotations

import numpy as np
import soundfile as sf

from workspaces.alignment_prototype.landmark_fp import (
    SR,
    fingerprint_from_audio,
    fingerprint_from_file_streaming,
)


def _synth(seconds: float = 40.0) -> np.ndarray:
    # A few stepping tones so constellation finds many landmarks across time.
    t = np.arange(int(seconds * SR)) / SR
    y = np.zeros_like(t)
    for i, f in enumerate([220.0, 440.0, 660.0, 880.0, 330.0, 550.0]):
        seg = (t >= i * seconds / 6) & (t < (i + 1) * seconds / 6)
        y[seg] += np.sin(2 * np.pi * f * t[seg])
    y += 0.01 * np.sin(2 * np.pi * 1000.0 * t)  # broadband-ish content
    return y.astype(np.float32)


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def test_streaming_matches_full_signal_fingerprint(tmp_path):
    y = _synth(40.0)
    wav = tmp_path / "m.wav"
    sf.write(
        wav, y, SR, subtype="FLOAT"
    )  # 32-bit lossless; 16-bit PCM clips & shifts STFT grids
    full = fingerprint_from_audio(y)
    # chunk_s=15 + overlap 3 → ~3 chunks over the 40s signal
    streamed = fingerprint_from_file_streaming(wav, chunk_s=15.0, overlap_s=3.0)
    # near-lossless: the interior of each chunk reproduces full-signal hashes; only
    # a few STFT-edge frames per boundary differ. Require high key overlap.
    j = _jaccard(set(full.hashes), set(streamed.hashes))
    assert j > 0.9, f"streaming keys Jaccard {j:.3f} too low"
    assert abs(streamed.duration_s - full.duration_s) < 0.5


def test_overlap_recovers_boundary_pairs(tmp_path):
    y = _synth(40.0)
    wav = tmp_path / "m.wav"
    sf.write(wav, y, SR, subtype="FLOAT")
    full_keys = set(fingerprint_from_audio(y).hashes)
    with_ovl = set(
        fingerprint_from_file_streaming(wav, chunk_s=15.0, overlap_s=3.0).hashes
    )
    no_ovl = set(
        fingerprint_from_file_streaming(wav, chunk_s=15.0, overlap_s=0.0).hashes
    )
    # overlap must recover at least as many true keys as no-overlap
    assert _jaccard(full_keys, with_ovl) >= _jaccard(full_keys, no_ovl)


def test_streaming_never_loads_whole_signal(monkeypatch, tmp_path):
    y = _synth(40.0)
    wav = tmp_path / "m.wav"
    sf.write(wav, y, SR, subtype="FLOAT")
    import librosa
    from workspaces.alignment_prototype import landmark_fp

    seen = []
    real_load = librosa.load

    def spy_load(*args, **kwargs):
        seen.append(kwargs.get("duration"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(
        landmark_fp.librosa if hasattr(landmark_fp, "librosa") else librosa,
        "load",
        spy_load,
        raising=False,
    )
    monkeypatch.setattr(librosa, "load", spy_load)
    fingerprint_from_file_streaming(wav, chunk_s=10.0, overlap_s=3.0)
    # every load bounded a single chunk window; NONE loaded the whole 40s
    assert seen, "expected chunked loads"
    # hop-snapping rounds chunk_s+overlap_s up by at most 2 hops (FHOP/SR ≈ 0.023 s each)
    assert all(d is not None and d <= 10.0 + 3.0 + 2 * (512 / 22050) for d in seen)
