"""Stem-domain per-layer MERT extraction (Phase 1B data dependency).

The blind open-set identity LF ([open_set_identity.py]) needs per-measure MERT at
tuned layers over STEM audio (mix vocal/instrumental stems + ref acappellas/
instrumentals) — L3 for vocal identity, L22 for instrumental. The repo's stored
MERT is full-track only ("ALL analysis runs on the ORIGINAL full audio … never
the stems", analysis/CLAUDE.md), so this must run MERT over the stem audio.

This module is the thin, correct core: reuse the analysis MERT adapter's
all-layer per-measure path and slice the layers we want. It is GPU-bound (MERT
auto-selects cuda→mps→cpu) — run it on a rented gpubox or Mac MPS. The set-level
resolution (which stems, which measure grids, which ref pool) is the runner built
+ validated during the extraction run; this core is unit-testable without a GPU.
Consumes the existing MERT channel at tuned layers — NOT a new sensor (module
CLAUDE.md sensor freeze; the open-set findings class L3/L22 as layer tuning).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Stem → identity layer (open_set_acappella_identity_findings.md, decision #22):
# vocal identity = low/acoustic (L3); instrumental/musical = high/abstract (L22).
VOCAL_LAYER = 3
INSTRUMENTAL_LAYER = 22


def _measure_stack(embeddings: tuple, dim: int) -> np.ndarray:
    """(n_measures, n_layers, dim) float32 from adapter MeasureEmbeddings.

    Each MeasureEmbedding.embedding_bytes is a flattened (n_layers, dim) float16
    (same layout export_mert_from_pi.probe slices). Measures are already in order.
    """
    out = []
    for m in embeddings:
        buf = np.frombuffer(m.embedding_bytes, dtype=np.float16)
        n_layers = buf.size // dim
        out.append(buf.reshape(n_layers, dim).astype(np.float32))
    return np.stack(out, axis=0) if out else np.empty((0, 0, dim), dtype=np.float32)


def slice_layers(stack: np.ndarray, layers: tuple[int, ...]) -> dict[int, np.ndarray]:
    """{layer: (n_measures, dim)} from a (n_measures, n_layers, dim) stack.

    Transposed to (dim, n_measures) — the (D, n) chamfer convention of
    open_set_identity.per_query_max_cos.
    """
    return {ell: stack[:, ell, :].T.copy() for ell in layers}


def embed_stem_layers(
    handle,
    samples_24k: np.ndarray,
    measure_times: tuple[float, ...],
    *,
    layers: tuple[int, ...] = (VOCAL_LAYER, INSTRUMENTAL_LAYER),
    track_audio_id: int = -1,
) -> dict[int, np.ndarray] | None:
    """Per-measure MERT for one stem, restricted to ``layers``.

    Returns {layer: (dim, n_measures)} for chamfer, or None on extraction error.
    `handle` is an analysis.adapters.mert_adapter.MertHandle; `measure_times` is
    the stem's beat_this measure grid. Runs one MERT pass over the stem (GPU).
    """
    from analysis.adapters.mert_adapter import embed_track_per_measure
    from core.result import Err, Ok

    match embed_track_per_measure(handle, samples_24k, track_audio_id, measure_times):
        case Err(_):
            return None
        case Ok(embeddings):
            if not embeddings:
                return None
            dim = embeddings[0].dim
            return slice_layers(_measure_stack(embeddings, dim), layers)


def cache_path(cache_dir: Path, audio_path: str, layer: int) -> Path:
    """Content-independent cache key by audio path + layer (mirrors _ensure_feat).

    NB: path-keyed cache — a locator, fine for a derived-feature cache (not
    identity). A byte-content key is the Phase-1 artifact-store upgrade.
    """
    import hashlib

    h = hashlib.sha1(str(audio_path).encode()).hexdigest()[:16]
    return cache_dir / f"{h}_mertL{layer}.npy"
