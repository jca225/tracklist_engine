"""beat_this adapter: beat + downbeat tracking.

Loads the pretrained checkpoint once via `load()`, then `run(handle, path)`
is called per track. Downbeats → measure times are derived by assuming a
4/4 time signature with one measure per downbeat (beat_this already emits
per-measure downbeats).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.result import Err, Ok, Result
from ..errors import BeatError


@dataclass(frozen=True)
class BeatThisHandle:
    """Opaque handle carrying the loaded model."""
    _file_pipeline: object         # beat_this.inference.File2Beats
    version: str


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load(checkpoint: str = "final0", device: str = "auto") -> Result[BeatThisHandle, BeatError]:
    try:
        from beat_this.inference import File2Beats
    except ImportError as e:
        return Err(BeatError(kind="model_load", detail=f"beat_this import: {e}"))
    try:
        f2b = File2Beats(checkpoint_path=checkpoint, device=_resolve_device(device))
    except (FileNotFoundError, RuntimeError, OSError) as e:
        return Err(BeatError(kind="model_load", detail=str(e)))
    return Ok(BeatThisHandle(_file_pipeline=f2b, version=checkpoint))


def predict(
    h: BeatThisHandle, audio_path: Path
) -> Result[tuple[tuple[float, ...], tuple[float, ...]], BeatError]:
    """Returns (beat_times, downbeat_times) in seconds."""
    try:
        beats, downbeats = h._file_pipeline(str(audio_path))
    except (FileNotFoundError, RuntimeError) as e:
        return Err(BeatError(kind="inference", detail=str(e)))
    return Ok((tuple(float(t) for t in beats), tuple(float(t) for t in downbeats)))


def estimate_bpm(beat_times: tuple[float, ...]) -> float:
    """Median inter-beat interval → bpm. Returns 0.0 if fewer than 2 beats."""
    if len(beat_times) < 2:
        return 0.0
    import numpy as np

    intervals = np.diff(beat_times)
    if intervals.size == 0:
        return 0.0
    median_ibi = float(np.median(intervals))
    return 60.0 / median_ibi if median_ibi > 0 else 0.0


def measure_times(downbeat_times: tuple[float, ...]) -> tuple[float, ...]:
    """For 4/4 EDM each downbeat is a measure. If another time-sig is
    detected later this is the place to adjust (e.g. grouping every N)."""
    return downbeat_times
