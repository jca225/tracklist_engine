"""MERT adapter: per-measure mean-pooled embeddings via `m-a-p/MERT-v1-95M`.

The track is read from disk once (full mix, never stems), MERT runs in
fixed 10s chunks (the model was trained on short clips, so chunking
avoids quadratic-attention blowups), per-chunk frame embeddings are
concatenated along the time axis, and finally mean-pooled within each
beat_this-derived measure window.

Output: one 768-d float16 vector per measure → `MeasureEmbedding`.
The legacy `embed_section` is kept for ad-hoc inspection but the
production pipeline uses `embed_track_per_measure`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.result import Err, Ok, Result
from ..errors import MertError
from ..models import MeasureEmbedding, SectionEmbedding

MERT_SR: int = 24000
MERT_MODEL: str = "m-a-p/MERT-v1-95M"
MERT_CHUNK_S: float = 10.0     # MERT-v1 was trained on 5s clips; 10s is a safe batch size.

# Mid-layer (6 of 12). The MERT paper shows mid-layers transfer best to
# music-ID / structural matching tasks — low layers are too acoustic,
# top layers too tagging-oriented. Matches `DEFAULT_LAYER` in
# `audio_pipeline/alignment/mert_align.py` so the analysis-side cache
# is reusable by alignment without re-embedding.
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
    """Embed one section by chunking at `chunk_s` seconds.

    MERT-v1 is a transformer trained on short clips; feeding a full section
    directly triggers quadratic attention blowups (GB-sized activations).
    We split into fixed chunks, embed each, and concatenate the per-chunk
    hidden states along the time axis so the section keeps its full
    resolution with sane memory.
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


def _embed_full_track_frames(
    h: MertHandle,
    samples_24k: np.ndarray,
    layer: int = MERT_DEFAULT_LAYER,
    chunk_s: float = MERT_CHUNK_S,
) -> Result[np.ndarray, MertError]:
    """Run MERT over the full track in 10s chunks, return concatenated
    frame-level hidden states of shape (n_frames_total, dim).

    Frame rate is determined by the MERT feature extractor (~75 Hz for
    MERT-v1-95M at 24 kHz input), so callers can map frame index to
    seconds via `frame_idx / (n_frames_total / total_duration_s)`.
    """
    if samples_24k.size < MERT_SR // 10:
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
    return Ok(np.concatenate(pieces, axis=0))


def embed_track_per_measure(
    h: MertHandle,
    samples_24k: np.ndarray,
    track_audio_id: int,
    measure_times: tuple[float, ...],
    layer: int = MERT_DEFAULT_LAYER,
    chunk_s: float = MERT_CHUNK_S,
) -> Result[tuple[MeasureEmbedding, ...], MertError]:
    """Run MERT once over the full track, then mean-pool per measure.

    `measure_times` is the beat_this-derived sequence of measure boundaries
    in seconds. We produce N-1 measure embeddings (one per (m[i], m[i+1])
    interval). Frames whose timestamp falls within a measure window are
    averaged into that measure's 768-d vector.

    Empty measures (no MERT frames overlap them — only happens if a
    measure is shorter than ~13 ms) are skipped silently rather than
    erroring; the gap is recorded by the missing measure_idx.
    """
    if len(measure_times) < 2:
        return Err(MertError(kind="empty_section",
                             detail=f"need >=2 measure boundaries, got {len(measure_times)}"))

    frames_r = _embed_full_track_frames(h, samples_24k, layer=layer, chunk_s=chunk_s)
    match frames_r:
        case Err(_):
            return frames_r
        case Ok(frames):
            pass

    total_duration_s = samples_24k.size / MERT_SR
    n_frames_total = int(frames.shape[0])
    if n_frames_total == 0 or total_duration_s <= 0:
        return Err(MertError(kind="empty_section", detail="zero MERT frames"))
    frames_per_s = n_frames_total / total_duration_s
    dim = int(frames.shape[1])

    out: list[MeasureEmbedding] = []
    for idx in range(len(measure_times) - 1):
        start_s = float(measure_times[idx])
        end_s = float(measure_times[idx + 1])
        f_start = max(0, int(round(start_s * frames_per_s)))
        f_end = min(n_frames_total, int(round(end_s * frames_per_s)))
        if f_end <= f_start:
            continue   # measure shorter than MERT frame resolution
        # Mean-pool in float32 for numerical stability, cast back to float16.
        pooled = frames[f_start:f_end].astype(np.float32, copy=False).mean(axis=0).astype(np.float16)
        out.append(MeasureEmbedding(
            track_audio_id=track_audio_id,
            measure_idx=idx,
            start_s=start_s,
            end_s=end_s,
            dim=dim,
            dtype="float16",
            embedding_bytes=pooled.tobytes(),
        ))

    if not out:
        return Err(MertError(kind="empty_section", detail="no measures produced"))
    return Ok(tuple(out))
