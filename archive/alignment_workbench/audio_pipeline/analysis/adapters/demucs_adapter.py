"""Demucs adapter: 4-stem split using `htdemucs_ft`.

Model loads once via `load()`, then `separate(handle, audio_path, out_dir,
track_audio_id)` writes four WAVs to disk and returns `StemAsset`s.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...result import Err, Ok, Result
from ..errors import StemError
from ..models import DERIVED_STEM_NAMES, STEM_NAMES, StemAsset, StemSet


# Map of derived stem name → which demucs sources to sum. Kept here
# (not in models.py) because it's an implementation detail of how we
# construct derived stems from the raw demucs output.
_DERIVED_RECIPES: dict[str, tuple[str, ...]] = {
    "instrumental": ("drums", "bass", "other"),
}


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
    source_tensors: dict[str, object] = {}
    try:
        for name in STEM_NAMES:
            if name not in model_sources:
                return Err(StemError(kind="inference", detail=f"model missing stem {name!r}"))
            idx = model_sources.index(name)
            src = sources[idx].cpu()
            source_tensors[name] = src
            path = dest / f"{name}.wav"
            torchaudio.save(str(path), src, h._model.samplerate)
            assets.append(StemAsset(
                track_audio_id=track_audio_id,
                stem_name=name,
                path=str(path),
                codec="wav",
            ))

        # Derived stems (instrumental = drums + bass + other). Written
        # alongside raw stems so alignment can treat the 3-stem
        # instrumental hypothesis as a single-file CCC/DTW target — big
        # speedup on tagged rows and avoids re-summing in the render
        # path. Sample-accurate addition in torch before one
        # `torchaudio.save` call.
        for derived_name in DERIVED_STEM_NAMES:
            recipe = _DERIVED_RECIPES.get(derived_name, ())
            tensors = [source_tensors[s] for s in recipe if s in source_tensors]
            if len(tensors) != len(recipe):
                continue
            mix = tensors[0].clone()
            for t in tensors[1:]:
                mix = mix + t
            path = dest / f"{derived_name}.wav"
            torchaudio.save(str(path), mix, h._model.samplerate)
            assets.append(StemAsset(
                track_audio_id=track_audio_id,
                stem_name=derived_name,
                path=str(path),
                codec="wav",
            ))
    except (OSError, RuntimeError) as e:
        return Err(StemError(kind="disk", detail=str(e)))

    return Ok(StemSet(track_audio_id=track_audio_id, stems=tuple(assets)))
