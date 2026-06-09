"""Decode per-measure MERT stacks into probe vectors."""
from __future__ import annotations

import numpy as np

from analysis.adapters.mert_adapter import MERT_DEFAULT_LAYER

MERT_PROBE_LAYER: int = MERT_DEFAULT_LAYER


def n_layers_from_blob(embedding_bytes: bytes, dim: int) -> int:
    n_float16 = len(embedding_bytes) // 2
    if n_float16 % dim != 0:
        raise ValueError(f"blob length {n_float16} not divisible by dim={dim}")
    return n_float16 // dim


def decode_measure_stack(embedding_bytes: bytes, dim: int) -> np.ndarray:
    """Return (n_layers, dim) float32 from a persisted measure blob."""
    n_layers = n_layers_from_blob(embedding_bytes, dim)
    stack = np.frombuffer(embedding_bytes, dtype=np.float16).reshape(n_layers, dim)
    return stack.astype(np.float32)


def probe_vector(
    embedding_bytes: bytes,
    dim: int,
    *,
    layer: int = MERT_PROBE_LAYER,
) -> np.ndarray:
    """Single-layer vector used by structure probes (default: layer 6)."""
    stack = decode_measure_stack(embedding_bytes, dim)
    if layer < 0 or layer >= stack.shape[0]:
        raise IndexError(f"layer {layer} out of range for n_layers={stack.shape[0]}")
    return stack[layer]


def probe_vectors_from_matrix(mert: np.ndarray, *, layer: int = MERT_PROBE_LAYER) -> np.ndarray:
    """(n_bars, n_layers, dim) or (n_bars, dim) → (n_bars, dim)."""
    if mert.ndim == 2:
        return mert.astype(np.float32)
    if mert.ndim != 3:
        raise ValueError(f"expected 2d or 3d mert, got shape {mert.shape}")
    return mert[:, layer, :].astype(np.float32)


def l2_normalize(rows: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.maximum(norms, eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine sim between two 1d vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
