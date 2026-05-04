"""MERT adapter: section-level embeddings via HuggingFace `m-a-p/MERT-v1-95M`.

Embeddings are stored per cue-delimited section (not per measure) — this
matches the scoped `track_mert_sections` table. The model is loaded once
and reused across tracks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...result import Err, Ok, Result
from ..errors import MertError
from ..models import SectionEmbedding

MERT_SR: int = 24000
MERT_MODEL: str = "m-a-p/MERT-v1-95M"
MERT_CHUNK_S: float = 10.0     # MERT-v1 was trained on 5s clips; 10s is a safe batch size.


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
    layer: int = -1,
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
