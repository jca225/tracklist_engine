"""MERT adapter: per-measure, all-layer embeddings via `m-a-p/MERT-v1-330M`.

The track is read from disk once (full mix, never stems), MERT runs in fixed
10s chunks (the model was trained on short clips, so chunking avoids
quadratic-attention blowups), and within each beat_this-derived measure window
the frame embeddings are mean-pooled — **for every hidden-state layer**, not a
single mid-layer pick.

Output: one (n_layers, dim) float16 tensor per measure → `MeasureEmbedding`.
For MERT-v1-330M that is (25, 1024): 24 transformer layers + the input
embedding. Persisting every layer (rather than the old single layer-6 pick) is
what lets a downstream learned weighted-sum / variant classifier choose
per-task which layers matter — the SUPERB / s3prl probing pattern — without
re-embedding. `embedding_bytes` is the flattened (n_layers, dim) array; `dim`
is the per-layer dimension (1024) and n_layers is recoverable as
`len(embedding_bytes) // (2 * dim)` (float16 == 2 bytes), so no schema column is
needed to describe the stack.

Pooling streams chunk-by-chunk and accumulates per-measure sums, so memory
stays bounded even for hour-long DJ mixes (materializing all layers
frame-level for a 1h mix at ~75 Hz would be ~14 GB).

The legacy `embed_section` (single-layer, raw frames) is kept for ad-hoc
inspection; the production pipeline uses `embed_track_per_measure`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.result import Err, Ok, Result
from ..errors import MertError
from ..models import MeasureEmbedding, SectionEmbedding

MERT_SR: int = 24000
MERT_MODEL: str = "m-a-p/MERT-v1-330M"
MERT_CHUNK_S: float = 10.0     # MERT-v1 was trained on 5s clips; 10s is a safe batch size.

# Legacy single-layer pick — used only by `embed_section` (ad-hoc inspection).
# The production per-measure path keeps ALL hidden states; the layer choice
# moves to a learned weighted sum co-trained with the downstream head.
MERT_DEFAULT_LAYER: int = 6


@dataclass(frozen=True)
class MertHandle:
    _model: object                 # transformers model
    _processor: object             # transformers feature extractor
    device: str
    version: str


def load(model_name: str = MERT_MODEL, device: str = "auto") -> Result[MertHandle, MertError]:
    try:
        import torch
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
    except ImportError as e:
        return Err(MertError(kind="model_load", detail=f"transformers import: {e}"))

    dev = device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    try:
        processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(dev).eval()
    except (OSError, RuntimeError, ValueError) as e:
        return Err(MertError(kind="model_load", detail=str(e)))
    return Ok(MertHandle(_model=model, _processor=processor, device=dev, version=model_name))


def embed_section(
    h: MertHandle,
    samples_24k: np.ndarray,
    track_audio_id: int,
    section_idx: int,
    start_s: float,
    end_s: float,
    layer: int = MERT_DEFAULT_LAYER,
    chunk_s: float = MERT_CHUNK_S,
) -> Result[SectionEmbedding, MertError]:
    """Legacy: embed one section at a single layer, keeping raw frames.

    Kept for ad-hoc inspection. The production path is
    `embed_track_per_measure` (all layers, mean-pooled per measure).
    """
    if samples_24k.size < MERT_SR // 10:   # < 0.1 s — not enough for MERT
        return Err(MertError(kind="empty_section", detail=f"{samples_24k.size} samples"))

    try:
        import torch
    except ImportError as e:
        return Err(MertError(kind="model_load", detail=f"torch runtime: {e}"))

    chunk_size = max(1, int(chunk_s * MERT_SR))
    pieces: list[np.ndarray] = []
    try:
        for i in range(0, samples_24k.size, chunk_size):
            chunk = samples_24k[i:i + chunk_size]
            if chunk.size < MERT_SR // 10:
                continue
            inputs = h._processor(chunk, sampling_rate=MERT_SR, return_tensors="pt")
            inputs = {k: v.to(h.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = h._model(**inputs, output_hidden_states=True)
            hidden = out.hidden_states[layer]      # (1, T_chunk, D)
            pieces.append(hidden.squeeze(0).to("cpu").to(torch.float16).numpy())
    except (RuntimeError, ValueError) as e:
        return Err(MertError(kind="inference", detail=str(e)))

    if not pieces:
        return Err(MertError(kind="empty_section", detail="no chunks embedded"))

    arr = np.concatenate(pieces, axis=0)
    return Ok(SectionEmbedding(
        track_audio_id=track_audio_id,
        section_idx=section_idx,
        start_s=start_s,
        end_s=end_s,
        n_frames=int(arr.shape[0]),
        dim=int(arr.shape[1]),
        dtype="float16",
        embedding_bytes=arr.tobytes(),
    ))


def _accumulate_chunk(
    sums: np.ndarray,          # (n_measures, n_layers, dim) float32 — mutated
    counts: np.ndarray,        # (n_measures,) int64 — mutated
    chunk_layers: np.ndarray,  # (n_layers, T_chunk, dim) one chunk, all layers
    chunk_start_s: float,      # global time (s) of this chunk's first sample
    frames_per_s: float,
    boundaries: np.ndarray,    # (n_measures + 1,) measure boundary times (s)
) -> None:
    """Bin one chunk's frames into per-measure sums/counts by frame time.

    Pure (no model dependency) so the pooling math is unit-testable without
    loading MERT. Each frame's centre time maps to the measure interval
    [boundaries[i], boundaries[i+1]); frames outside all measures are dropped.
    """
    t_chunk = chunk_layers.shape[1]
    times = chunk_start_s + (np.arange(t_chunk) + 0.5) / frames_per_s
    bins = np.searchsorted(boundaries, times, side="right") - 1
    n_measures = len(boundaries) - 1
    in_range = (bins >= 0) & (bins < n_measures)
    if not in_range.any():
        return
    for b in np.unique(bins[in_range]):
        mask = bins == b   # b is in-range, so all such frames are valid
        sums[b] += chunk_layers[:, mask, :].sum(axis=1).astype(np.float32)
        counts[b] += int(mask.sum())


def embed_track_per_measure(
    h: MertHandle,
    samples_24k: np.ndarray,
    track_audio_id: int,
    measure_times: tuple[float, ...],
    chunk_s: float = MERT_CHUNK_S,
) -> Result[tuple[MeasureEmbedding, ...], MertError]:
    """Run MERT once over the full track, mean-pooling ALL hidden-state layers
    per measure.

    `measure_times` is the beat_this-derived sequence of measure boundaries in
    seconds; we produce up to N-1 measure embeddings (one per (m[i], m[i+1])
    interval). Each is an (n_layers, dim) float16 tensor, flattened into
    `embedding_bytes` with `dim` = the per-layer dimension. Empty measures (no
    frames overlap) are skipped; the gap is recorded by the missing measure_idx.

    Streams chunk-by-chunk and accumulates per-measure sums so peak memory is
    bounded by the (n_measures, n_layers, dim) accumulator, not the full
    frame-level tensor.
    """
    if len(measure_times) < 2:
        return Err(MertError(kind="empty_section",
                             detail=f"need >=2 measure boundaries, got {len(measure_times)}"))
    if samples_24k.size < MERT_SR // 10:
        return Err(MertError(kind="empty_section", detail=f"{samples_24k.size} samples"))

    try:
        import torch
    except ImportError as e:
        return Err(MertError(kind="model_load", detail=f"torch runtime: {e}"))

    boundaries = np.asarray(measure_times, dtype=np.float64)
    n_measures = len(boundaries) - 1
    chunk_size = max(1, int(chunk_s * MERT_SR))

    sums: np.ndarray | None = None      # lazily sized once we know (n_layers, dim)
    counts = np.zeros(n_measures, dtype=np.int64)
    frames_per_s: float | None = None
    dim: int | None = None

    try:
        for i in range(0, samples_24k.size, chunk_size):
            chunk = samples_24k[i:i + chunk_size]
            if chunk.size < MERT_SR // 10:
                continue
            inputs = h._processor(chunk, sampling_rate=MERT_SR, return_tensors="pt")
            inputs = {k: v.to(h.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = h._model(**inputs, output_hidden_states=True)
            # Stack every hidden state: (n_layers, T_chunk, dim).
            layers = torch.stack(out.hidden_states, dim=0).squeeze(1)
            layers_np = layers.to("cpu").to(torch.float16).numpy()
            del out, layers
            n_layers, t_chunk, d = layers_np.shape
            if sums is None:
                dim = d
                frames_per_s = t_chunk / (chunk.size / MERT_SR)
                sums = np.zeros((n_measures, n_layers, d), dtype=np.float32)
            _accumulate_chunk(sums, counts, layers_np, i / MERT_SR, frames_per_s, boundaries)
    except (RuntimeError, ValueError) as e:
        return Err(MertError(kind="inference", detail=str(e)))

    if sums is None:
        return Err(MertError(kind="empty_section", detail="no chunks embedded"))

    out_list: list[MeasureEmbedding] = []
    for idx in range(n_measures):
        if counts[idx] == 0:
            continue
        pooled = sums[idx] / counts[idx]
        pooled = np.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
        pooled = pooled.astype(np.float16)
        out_list.append(MeasureEmbedding(
            track_audio_id=track_audio_id,
            measure_idx=idx,
            start_s=float(boundaries[idx]),
            end_s=float(boundaries[idx + 1]),
            dim=int(dim),
            dtype="float16",
            embedding_bytes=pooled.tobytes(),
        ))

    if not out_list:
        return Err(MertError(kind="empty_section", detail="no measures produced"))
    return Ok(tuple(out_list))
