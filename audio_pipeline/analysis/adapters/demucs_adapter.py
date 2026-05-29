"""Demucs adapter: 2-stem split (vocals / instrumental) via `htdemucs_ft`.

The model still internally computes 4 sources (vocals/drums/bass/other) —
that's how htdemucs_ft works, no faster way to get just two. We just
persist only `vocals.flac` (direct) and `instrumental.flac` (drums + bass +
other summed sample-accurately). DJs split on this axis and there's no
use case in the rest of the pipeline for keeping the individual
non-vocal stems.

Stems are written as 16-bit FLAC (was WAV until 2026-05-06). On Vast →
pi-storage rsync this saves ~50% bandwidth (~25 s/track wall) without
quality loss — Demucs internally outputs float32, but downstream
consumers (MERT resampled to 24 kHz mono, browser_daw playback) don't
benefit from 24-bit, and lossless FLAC at 16-bit fits the use case.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.result import Err, Ok, Result
from ..errors import StemError
from ..models import STEM_NAMES, _STEM_RECIPES, StemAsset, StemSet


@dataclass(frozen=True)
class DemucsHandle:
    _model: object                 # demucs.apply.BagOfModels or Model
    device: str
    version: str                   # 'htdemucs_ft'


def load(model_name: str = "htdemucs_ft", device: str = "auto") -> Result[DemucsHandle, StemError]:
    try:
        import torch
        from demucs.pretrained import get_model
    except ImportError as e:
        return Err(StemError(kind="model_load", detail=f"demucs import: {e}"))
    dev = device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    try:
        model = get_model(model_name)
        model.to(dev).eval()
    except (FileNotFoundError, RuntimeError, OSError, ValueError) as e:
        return Err(StemError(kind="model_load", detail=str(e)))
    return Ok(DemucsHandle(_model=model, device=dev, version=model_name))


def separate(
    h: DemucsHandle,
    audio_path: Path,
    out_dir: Path,
    track_audio_id: int,
) -> Result[StemSet, StemError]:
    """Split one file into 4 stems, write as WAV into `out_dir/track_audio_id/`."""
    try:
        import torch
        import torchaudio
        from demucs.apply import apply_model
        from demucs.audio import convert_audio
    except ImportError as e:
        return Err(StemError(kind="model_load", detail=f"demucs runtime import: {e}"))

    try:
        wav, sr = torchaudio.load(str(audio_path))
        wav = convert_audio(wav, sr, h._model.samplerate, h._model.audio_channels)
        with torch.no_grad():
            sources = apply_model(h._model, wav[None], device=h.device, split=True, overlap=0.25)[0]
    except (FileNotFoundError, RuntimeError, OSError, ValueError) as e:
        return Err(StemError(kind="inference", detail=str(e)))

    model_sources = list(h._model.sources)
    dest = out_dir / str(track_audio_id)
    dest.mkdir(parents=True, exist_ok=True)
    assets: list[StemAsset] = []
    try:
        # Cache raw demucs source tensors so we can compose persisted
        # stems from them without re-running the model.
        raw_sources: dict[str, object] = {}
        for src_name in set().union(*(_STEM_RECIPES[n] for n in STEM_NAMES)):
            if src_name not in model_sources:
                return Err(StemError(kind="inference", detail=f"model missing source {src_name!r}"))
            raw_sources[src_name] = sources[model_sources.index(src_name)].cpu()

        for stem_name in STEM_NAMES:
            recipe = _STEM_RECIPES[stem_name]
            mix = raw_sources[recipe[0]].clone()
            for s in recipe[1:]:
                mix = mix + raw_sources[s]
            path = dest / f"{stem_name}.flac"
            # 16-bit FLAC: ~50% smaller than 16-bit WAV (PCM) for music,
            # lossless. Demucs outputs float32 internally; torchaudio.save
            # without `bits_per_sample` defaults to 16 for FLAC, which
            # downstream (MERT @ 24 kHz mono, browser playback) doesn't
            # exceed. Encoding adds ~3-5s CPU per stem on Vast — still
            # net-faster wall-clock than shipping uncompressed WAV.
            torchaudio.save(str(path), mix, h._model.samplerate, format="flac")
            assets.append(StemAsset(
                track_audio_id=track_audio_id,
                stem_name=stem_name,
                path=str(path),
                codec="flac",
            ))
    except (OSError, RuntimeError) as e:
        return Err(StemError(kind="disk", detail=str(e)))

    return Ok(StemSet(track_audio_id=track_audio_id, stems=tuple(assets)))
