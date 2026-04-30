"""MERT measure-level embedding + cache.

The only public surface the alignment pipeline uses is
`_cache_measure_embeddings` (wrapped by `indicators_debug._embed_per_measure`):
compute one L2-normalised 768-dim MERT vector per measure, cache it as
`.npz` under `data/cache/mert/`, and return it.

Caching strategy: key on `(audio file + mtime + size, MERT layer, measure
grid hash, offset, duration)`. A ref's embedding is computed once per
variant and reused across every set that references it; the mix is
computed once per stem and reused across every ref comparison in the set.

The embedding call chain is:
    _cache_measure_embeddings
      → compute_frame_embeddings  (MERT forward pass, ~75 Hz frames)
      → pool_to_measures          (mean-pool frames per measure, L2-normalise)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..result import Err, Ok, Result
from .errors import AlignmentError


# On-disk cache root.
_CACHE_DIR = Path("data/cache/mert")

# MERT-v1 layer 6 balances low-level acoustic (layers 1-3) against
# task-specific signals (top of stack). The MERT paper shows mid-layers
# transfer best to music-ID tasks.
DEFAULT_LAYER: int = 6


# ---------- cache key + path ------------------------------------------------

def _cache_key(audio_path: Path, layer: int) -> str:
    """Stable cache key per (audio_file, layer)."""
    stat = audio_path.stat()
    raw = f"{audio_path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}|layer={layer}"
    return hashlib.blake2b(raw.encode(), digest_size=12).hexdigest()


def _cache_path(key: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{key}.npz"


# ---------- lazy MERT handle ------------------------------------------------
# The model weights are ~400 MB on disk / ~1.5 GB resident. Loading is
# expensive so we do it once per process and reuse across every
# embedding call. Kept private so tests can import this module without
# triggering the load.

_mert_handle: object | None = None


def _get_handle() -> object:
    global _mert_handle
    if _mert_handle is None:
        from ..analysis.adapters import mert_adapter
        r = mert_adapter.load()
        if not r.is_ok():
            raise RuntimeError(f"MERT load failed: {r.error}")
        _mert_handle = r.value
    return _mert_handle


# ---------- forward pass ----------------------------------------------------

def compute_frame_embeddings(
    audio_path: Path,
    *,
    offset_s: float = 0.0,
    duration_s: float | None = None,
    layer: int = DEFAULT_LAYER,
    chunk_s: float = 10.0,
) -> Result[tuple[np.ndarray, int], AlignmentError]:
    """Full-file MERT embeddings at the model's native frame rate.

    Returns `(embeddings, frame_rate_hz)` where embeddings is `(T, D)`
    and `frame_rate_hz` is frames / second (≈75 on MERT-v1-95M). Audio
    is loaded at 24 kHz (MERT-required), chunked (attention is quadratic
    past ~30 s; the model was trained on 10 s clips), and chunk outputs
    concatenated along time.
    """
    try:
        from ..analysis.adapters.mert_adapter import MERT_SR
        import librosa
        import torch
    except ImportError as e:
        return Err(AlignmentError(kind="dtw_failed", detail=f"mert runtime: {e}"))

    handle = _get_handle()
    # librosa's audioread fallback chokes on offset=None → pass 0.0.
    # duration=None is accepted and means "read to end", so keep that.
    try:
        y, _ = librosa.load(
            str(audio_path), sr=MERT_SR, mono=True,
            offset=float(offset_s or 0.0),
            duration=float(duration_s) if duration_s else None,
        )
    except (FileNotFoundError, OSError, ValueError) as e:
        return Err(AlignmentError(kind="dtw_failed", detail=f"mert load: {e}"))
    if y.size < MERT_SR // 10:
        return Err(AlignmentError(kind="dtw_failed", detail="mert: audio too short"))

    chunk_size = int(chunk_s * MERT_SR)
    pieces: list[np.ndarray] = []
    try:
        for i in range(0, y.size, chunk_size):
            chunk = y[i:i + chunk_size]
            if chunk.size < MERT_SR // 10:
                continue
            inputs = handle._processor(chunk, sampling_rate=MERT_SR, return_tensors="pt")
            inputs = {k: v.to(handle.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = handle._model(**inputs, output_hidden_states=True)
            hidden = out.hidden_states[layer]                # (1, T, D)
            pieces.append(hidden.squeeze(0).to("cpu").to(torch.float32).numpy())
    except (RuntimeError, ValueError) as e:
        return Err(AlignmentError(kind="dtw_failed", detail=f"mert inference: {e}"))

    if not pieces:
        return Err(AlignmentError(kind="dtw_failed", detail="mert: no chunks"))

    arr = np.concatenate(pieces, axis=0)
    frame_rate = arr.shape[0] / (y.size / MERT_SR)
    return Ok((arr, int(round(frame_rate))))


# ---------- measure pooling -------------------------------------------------

def pool_to_measures(
    frame_embeddings: np.ndarray,
    frame_rate_hz: int,
    measures: list[tuple[int, float, float, float | None]],
    *,
    offset_s: float = 0.0,
) -> np.ndarray:
    """Mean-pool frame embeddings into per-measure vectors, L2-normalised.

    For each `(measure_idx, start_s, end_s, bpm)` in the grid, average
    the MERT frames whose timestamps fall inside, then L2-normalise so a
    subsequent `A @ B.T` gives cosine similarity directly. Measures past
    the end of the embedded audio get zero rows (norm stays ≈0).
    """
    D = frame_embeddings.shape[1]
    out = np.zeros((len(measures), D), dtype=np.float32)
    base = float(offset_s)
    for i, (_midx, ms, me, _bpm) in enumerate(measures):
        frame_lo = max(0, int((ms - base) * frame_rate_hz))
        frame_hi = min(frame_embeddings.shape[0], int((me - base) * frame_rate_hz))
        if frame_hi <= frame_lo:
            continue
        out[i] = frame_embeddings[frame_lo:frame_hi].mean(axis=0)

    norms = np.linalg.norm(out, axis=1, keepdims=True)
    out = out / (norms + 1e-9)
    return out.astype(np.float32)


# ---------- cached compute --------------------------------------------------

def _cache_measure_embeddings(
    audio_path: Path,
    measures: list[tuple[int, float, float, float | None]],
    layer: int = DEFAULT_LAYER,
    *,
    offset_s: float = 0.0,
    duration_s: float | None = None,
) -> Result[np.ndarray, AlignmentError]:
    """Cached compute: `(N_measures, D)` L2-normalised MERT embeddings
    for one audio file + measure grid + layer combination. Atomic
    write-then-replace so concurrent callers don't corrupt the `.npz`."""
    # Include the measure grid in the key — different grids give
    # different poolings of the same underlying frames.
    grid_hash = hashlib.blake2b(
        np.array([(m[1], m[2]) for m in measures], dtype=np.float64).tobytes(),
        digest_size=8,
    ).hexdigest()
    key = f"{_cache_key(audio_path, layer)}_grid={grid_hash}_off={offset_s:.2f}_dur={duration_s or 0:.2f}"
    path = _cache_path(key)
    if path.exists():
        try:
            npz = np.load(path)
            return Ok(npz["emb"].astype(np.float32))
        except (OSError, ValueError, KeyError):
            pass   # cache corruption — re-compute

    frames_r = compute_frame_embeddings(
        audio_path, offset_s=offset_s, duration_s=duration_s, layer=layer,
    )
    if not frames_r.is_ok():
        return frames_r
    frames, fr_hz = frames_r.value
    embeddings = pool_to_measures(frames, fr_hz, measures, offset_s=offset_s)
    try:
        tmp = path.with_suffix(".npz.tmp")
        np.savez_compressed(tmp, emb=embeddings)
        tmp.replace(path)
    except OSError:
        pass       # cache write failure — compute result is still valid
    return Ok(embeddings)
