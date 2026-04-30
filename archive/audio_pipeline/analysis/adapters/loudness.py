"""pyloudnorm adapter: integrated LUFS per ITU-R BS.1770."""
from __future__ import annotations

import numpy as np

from ...result import Err, Ok, Result
from ..errors import LoudnessError


def integrated_lufs(samples: np.ndarray, sample_rate: int) -> Result[float, LoudnessError]:
    """Returns integrated loudness in LUFS. Expects mono float32 ≥ ~1s of audio."""
    import pyloudnorm as pyln

    if samples.size < sample_rate:
        return Err(LoudnessError(kind="signal_too_short", detail=f"{samples.size} samples"))
    meter = pyln.Meter(sample_rate)
    try:
        value = float(meter.integrated_loudness(samples.astype(np.float64, copy=False)))
    except ValueError as e:
        return Err(LoudnessError(kind="nan", detail=str(e)))
    if np.isnan(value) or np.isinf(value):
        return Err(LoudnessError(kind="nan", detail="non-finite loudness"))
    return Ok(value)
