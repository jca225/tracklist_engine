"""Per-frame MERT emissions for the overlay channel.

v2 showed vocal chroma can't say *which* acappella (overlay 25%). MERT is the
identity feature (~100% in the prior aligner), so the overlay channel decodes
against per-frame MERT instead. This computes per-frame (frame_s-second grid)
single-layer MERT for a set of audio files and caches them, reusing the existing
mert_adapter (Mac MPS). One forward pass per track, pooled into frame bins.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.adapters import audio_io, mert_adapter  # noqa: E402
from workspaces.section_hsmm.v0_1_chroma_scorecard import _CACHE  # noqa: E402


def _l2rows(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def _decode_layer(m, layer: int) -> np.ndarray:
    n_layers = len(m.embedding_bytes) // (2 * m.dim)
    arr = np.frombuffer(m.embedding_bytes, dtype=np.float16).reshape(n_layers, m.dim)
    return arr[layer].astype(np.float32)


def cache_path(cache_key: str, frame_s: float, layer: int) -> Path:
    return _CACHE / f"{cache_key}_mertL{layer}_pool{frame_s}.npy"


def load_pooled_mert(cache_key: str, frame_s: float, layer: int) -> np.ndarray:
    return np.load(cache_path(cache_key, frame_s, layer))


def ensure_mert_cache(items: list[tuple[Path, str]], frame_s: float, layer: int) -> None:
    """items: (audio_path, cache_key). Compute + cache any missing per-frame MERT."""
    todo = [(p, k) for (p, k) in items if not cache_path(k, frame_s, layer).is_file()]
    if not todo:
        return
    print(f"MERT: embedding {len(todo)} files (layer {layer}, {frame_s}s frames) …",
          file=sys.stderr)
    h_r = mert_adapter.load()
    if not h_r.is_ok():
        sys.exit(f"MERT load failed: {h_r.error.detail}")
    h = h_r.value
    _CACHE.mkdir(parents=True, exist_ok=True)
    for i, (path, key) in enumerate(todo, 1):
        wf = audio_io.load_mono(Path(path), target_sr=mert_adapter.MERT_SR)
        if not wf.is_ok():
            print(f"  [{i}/{len(todo)}] {key}: load failed {wf.error.detail}", file=sys.stderr)
            np.save(cache_path(key, frame_s, layer), np.zeros((0, 1024), np.float32))
            continue
        dur = wf.value.samples.size / mert_adapter.MERT_SR
        grid = tuple(float(x) for x in np.arange(0.0, dur, frame_s)) + (float(dur),)
        if len(grid) < 2:
            grid = (0.0, float(dur))
        emb = mert_adapter.embed_track_per_measure(
            h, wf.value.samples, track_audio_id=0, measure_times=grid)
        if not emb.is_ok() or not emb.value:
            print(f"  [{i}/{len(todo)}] {key}: embed failed", file=sys.stderr)
            np.save(cache_path(key, frame_s, layer), np.zeros((0, 1024), np.float32))
            continue
        vecs = _l2rows(np.stack([_decode_layer(m, layer) for m in emb.value], axis=0))
        np.save(cache_path(key, frame_s, layer), vecs.astype(np.float32))
        if i % 10 == 0 or i == len(todo):
            print(f"  [{i}/{len(todo)}] cached", file=sys.stderr)
