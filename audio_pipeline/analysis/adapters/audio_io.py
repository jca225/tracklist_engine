"""librosa / soundfile adapter for loading PCM off disk."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...result import Err, Ok, Result
from ..errors import AudioIoError


@dataclass(frozen=True)
class Waveform:
    samples: np.ndarray            # 1-D float32, mono
    sample_rate: int


def load_mono(path: Path, target_sr: int | None = None) -> Result[Waveform, AudioIoError]:
    """Load a file as mono float32, optionally resampling to `target_sr`.

    Catches librosa/soundfile's documented exceptions for missing or corrupt
    files and converts them to `AudioIoError`.
    """
    import librosa
    import soundfile as sf  # noqa: F401 — librosa routes through it

    p = Path(path)
    if not p.exists():
        return Err(AudioIoError(kind="not_found", path=str(p), detail=""))
    try:
        y, sr = librosa.load(str(p), sr=target_sr, mono=True)
    except FileNotFoundError as e:
        return Err(AudioIoError(kind="not_found", path=str(p), detail=str(e)))
    except (sf.LibsndfileError, EOFError, ValueError) as e:
        return Err(AudioIoError(kind="decode", path=str(p), detail=str(e)))
    return Ok(Waveform(samples=y.astype(np.float32, copy=False), sample_rate=int(sr)))
