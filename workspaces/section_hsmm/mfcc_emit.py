"""Pooled per-frame MFCC features for the vocal-verification channel.

MFCC (vocal timbre) is the feature the v7 pre-test proved discriminative for
acappella identity (70% retrieval@1 vs chroma's 43% chance). This pools frame-
level MFCC into frame_s bins for the decoder, reusing similarity_probe's
frame-level MFCC cache when present (keys `<x>_mfcc.npy`).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from workspaces.alignment_prototype.refine_ref_offsets import HOP, SR  # noqa: E402
from workspaces.section_hsmm.v0_1_chroma_scorecard import _CACHE  # noqa: E402

FPS = SR / HOP


def _mfcc(y: np.ndarray) -> np.ndarray:
    import librosa
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = librosa.feature.mfcc(y=y, sr=SR, hop_length=HOP, n_mfcc=20)[1:]  # drop energy
    return (m / (np.linalg.norm(m, axis=0, keepdims=True) + 1e-8)).astype(np.float32)


def _l2rows(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def pooled_mfcc(audio_path: Path | None, frame_key: str, pool_key: str,
                frame_s: float) -> np.ndarray:
    """(n, 19) L2-normed MFCC pooled into frame_s bins; reuses frame-level cache.
    Pool cache is `_mfcc`-suffixed to avoid colliding with chroma pooled caches
    that share the same pool_key."""
    pf = _CACHE / f"{pool_key}_mfcc.npy"
    if pf.is_file():
        return np.load(pf)
    ff = _CACHE / f"{frame_key}_mfcc.npy"      # similarity_probe's frame-level cache
    if ff.is_file():
        c = np.load(ff)                         # (19, frames)
    else:
        if audio_path is None or not Path(audio_path).is_file():
            return np.zeros((0, 19), dtype=np.float32)
        import librosa
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
        c = _mfcc(y)
        _CACHE.mkdir(parents=True, exist_ok=True)
        np.save(ff, c)
    w = max(1, int(round(frame_s * FPS)))
    n = c.shape[1] // w
    if n == 0:
        return np.zeros((0, 19), dtype=np.float32)
    pooled = c[:, : n * w].reshape(c.shape[0], n, w).mean(axis=2)
    out = _l2rows(pooled.T).astype(np.float32)
    _CACHE.mkdir(parents=True, exist_ok=True)
    np.save(pf, out)
    return out
