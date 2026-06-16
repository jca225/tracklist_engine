"""Per-frame HuBERT emissions for the overlay channel (A/B vs MERT).

Mirrors mert_emit.py's cache interface so the v3/abstain overlay decode can swap
HuBERT phonetic embeddings in for MERT. The pre-test (similarity_probe) showed
HuBERT L9 separates manipulated acappellas far better than MFCC (rk16->rk1 on
time-stretched spans); this wires the same feature into the actual Viterbi
overlay channel so we can read the abstain precision@coverage curve.

Reuses similarity_probe's frame-level HuBERT cache (`<key>_hubertL<layer>.npy`),
so the hour-long mix_vocals pass is not recomputed. Pools frame-level (SR/HOP
grid) into frame_s bins by mean + L2-norm, exactly as mfcc_emit.pooled_mfcc.
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
from workspaces.section_hsmm.decode_hsmm import Vocab  # noqa: E402
from workspaces.section_hsmm.similarity_probe import _hubert  # noqa: E402
from workspaces.section_hsmm.v0_1_chroma_scorecard import _CACHE  # noqa: E402

FPS = SR / HOP
DIM = 768


def _l2rows(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def cache_path(cache_key: str, frame_s: float, layer: int) -> Path:
    return _CACHE / f"{cache_key}_hubertL{layer}_pool{frame_s}.npy"


def load_pooled_hubert(cache_key: str, frame_s: float, layer: int) -> np.ndarray:
    return np.load(cache_path(cache_key, frame_s, layer))


def ensure_hubert_cache(items: list[tuple[Path, str]], frame_s: float, layer: int) -> None:
    """items: (audio_path, cache_key). Compute + cache any missing pooled HuBERT.
    Reuses the frame-level `<key>_hubertL<layer>.npy` cache when present."""
    todo = [(p, k) for (p, k) in items if not cache_path(k, frame_s, layer).is_file()]
    if not todo:
        return
    print(f"HuBERT: pooling {len(todo)} files (layer {layer}, {frame_s}s frames) …",
          file=sys.stderr)
    import librosa
    w = max(1, int(round(frame_s * FPS)))
    for i, (path, key) in enumerate(todo, 1):
        ff = _CACHE / f"{key}_hubertL{layer}.npy"      # similarity_probe frame cache
        if ff.is_file():
            c = np.load(ff)                             # (768, F)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y, _ = librosa.load(str(path), sr=SR, mono=True)
            c = _hubert(y, layer)
            _CACHE.mkdir(parents=True, exist_ok=True)
            np.save(ff, c)
        n = c.shape[1] // w
        if n == 0:
            np.save(cache_path(key, frame_s, layer), np.zeros((0, DIM), np.float32))
            continue
        pooled = c[:, : n * w].reshape(c.shape[0], n, w).mean(axis=2)
        np.save(cache_path(key, frame_s, layer), _l2rows(pooled.T).astype(np.float32))
        if i % 10 == 0 or i == len(todo):
            print(f"  [{i}/{len(todo)}] cached", file=sys.stderr)


def assemble_vocab(key_of: dict[str, str], frame_s: float, layer: int) -> Vocab:
    """Stack pooled-HuBERT refs into a Vocab (mirror of decode_v3._assemble_vocab)."""
    tids, refs, track_of, ref_frame, slices = [], [], [], [], []
    cur = 0
    for tid, key in key_of.items():
        c = load_pooled_hubert(key, frame_s, layer)
        if c.shape[0] < 2:
            continue
        k = len(tids)
        tids.append(tid)
        refs.append(c)
        slices.append((cur, cur + c.shape[0]))
        track_of.extend([k] * c.shape[0])
        ref_frame.extend(range(c.shape[0]))
        cur += c.shape[0]
    if not refs:
        return Vocab([], np.array([]), np.array([]), [], np.zeros((0, DIM), np.float32))
    return Vocab(tids, np.array(ref_frame, np.int32), np.array(track_of, np.int32),
                 slices, np.concatenate(refs, axis=0).astype(np.float32))
