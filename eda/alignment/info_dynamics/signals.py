"""Per-frame information signals + light smoothing.

A ``SignalSet`` is just a named bag of length-``n_frames`` float arrays, one per
information measure a model emits. ``np.nan`` marks frames where the signal is
undefined (cold-start, warmup) — excluded from scoring and NLL.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SignalSet:
    model: str
    n_frames: int
    signals: dict[str, np.ndarray]
    # Optional scalar: mean prequential surprisal (nats) over valid+labeled frames.
    prequential_nll: float = float("nan")

    def names(self) -> list[str]:
        return list(self.signals.keys())

    def get(self, name: str) -> np.ndarray:
        return self.signals[name]


def smooth(signal: np.ndarray, *, window: int) -> np.ndarray:
    """Centered moving average, NaN-aware. window<=1 is a no-op."""
    s = np.asarray(signal, dtype=np.float64)
    if window <= 1:
        return s.copy()
    out = np.full_like(s, np.nan)
    half = window // 2
    for i in range(len(s)):
        lo = max(0, i - half)
        hi = min(len(s), i + half + 1)
        chunk = s[lo:hi]
        finite = chunk[np.isfinite(chunk)]
        if finite.size:
            out[i] = float(finite.mean())
    return out
