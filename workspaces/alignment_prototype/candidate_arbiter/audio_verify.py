"""Pinned audio-pair verification features for a proposed alignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairVerification:
    match_mean: float
    match_p10: float
    margin: float
    continuity: float
    support_bins: int
    background_count: int


def _window(
    values: np.ndarray,
    start: int,
    length: int,
) -> np.ndarray | None:
    if start < 0 or length <= 0 or start + length > values.shape[0]:
        return None
    return np.asarray(values[start : start + length], dtype=np.float32)


def _diagonal_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = left / (np.linalg.norm(left, axis=1, keepdims=True) + 1e-9)
    right_norm = right / (np.linalg.norm(right, axis=1, keepdims=True) + 1e-9)
    return np.sum(left_norm * right_norm, axis=1)


def verify_alignment(
    mix_features: np.ndarray,
    ref_features: np.ndarray,
    *,
    mix_start_bin: int,
    ref_start_bin: int,
    window_bins: int,
    negative_shift_bins: tuple[int, ...] = (-40, -20, 20, 40),
    continuity_floor: float = 0.5,
) -> PairVerification | None:
    """Verify one pinned mix↔reference window against nearby negative shifts.

    Feature matrices are ``(time, dimensions)``. The verifier does no search:
    it scores the proposed diagonal and compares it with the same reference
    window at nearby mix positions. This is deliberately easier than proposing
    an alignment and cannot change a timeline by itself.
    """
    mix_window = _window(mix_features, mix_start_bin, window_bins)
    ref_window = _window(ref_features, ref_start_bin, window_bins)
    if mix_window is None or ref_window is None:
        return None
    if mix_window.shape[1] != ref_window.shape[1]:
        raise ValueError("mix and reference feature dimensions differ")

    scores = _diagonal_similarity(mix_window, ref_window)
    backgrounds: list[float] = []
    for shift in negative_shift_bins:
        negative = _window(mix_features, mix_start_bin + shift, window_bins)
        if negative is not None:
            backgrounds.append(float(_diagonal_similarity(negative, ref_window).mean()))
    background_max = max(backgrounds) if backgrounds else float(scores.mean())
    mean = float(scores.mean())
    return PairVerification(
        match_mean=mean,
        match_p10=float(np.quantile(scores, 0.1)),
        margin=mean - background_max,
        continuity=float(np.mean(scores >= continuity_floor)),
        support_bins=int(scores.size),
        background_count=len(backgrounds),
    )
