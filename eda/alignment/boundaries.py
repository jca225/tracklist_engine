"""Boundary extraction, peak picking, and GT scoring."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from labeling.ground_truth.schema import GroundTruthSet


@dataclass(frozen=True)
class BoundaryScore:
    tolerance_bars: int
    n_gt: int
    n_pred: int
    tp: int
    precision: float
    recall: float
    f1: float


def gt_section_starts_s(gt: GroundTruthSet, *, merge_tol_s: float = 0.5) -> tuple[float, ...]:
    """Unique mix entry times from GT span rows (merged within tolerance)."""
    starts = sorted(t.set_start_s for t in gt.tracks)
    if not starts:
        return ()
    merged: list[float] = [starts[0]]
    for s in starts[1:]:
        if s - merged[-1] > merge_tol_s:
            merged.append(s)
    return tuple(merged)


def seconds_to_bar_indices(
    times_s: tuple[float, ...] | list[float],
    bar_start_s: np.ndarray,
) -> tuple[int, ...]:
    """Map timestamps to nearest bar index."""
    out: list[int] = []
    for t in times_s:
        idx = int(np.argmin(np.abs(bar_start_s - t)))
        out.append(idx)
    return tuple(sorted(set(out)))


def pick_peaks(
    signal: np.ndarray,
    *,
    min_distance: int = 8,
    percentile: float = 90.0,
) -> tuple[int, ...]:
    """Local-max peaks above a global percentile threshold."""
    if signal.size == 0:
        return ()
    thresh = float(np.percentile(signal, percentile))
    return _pick_local_maxima(signal, thresh=thresh, min_distance=min_distance)


def pick_local_peaks(
    signal: np.ndarray,
    *,
    window: int = 32,
    z_threshold: float = 1.5,
    min_distance: int = 4,
) -> tuple[int, ...]:
    """Peaks where signal exceeds a rolling baseline by z_threshold std devs."""
    if signal.size == 0:
        return ()
    s = np.asarray(signal, dtype=np.float64)
    half = max(window // 2, 1)
    z = np.zeros_like(s)
    for i in range(len(s)):
        lo = max(0, i - half)
        hi = min(len(s), i + half + 1)
        local = s[lo:hi]
        med = float(np.median(local))
        std = float(np.std(local))
        z[i] = (s[i] - med) / (std + 1e-8)
    return _pick_local_maxima(z, thresh=z_threshold, min_distance=min_distance)


def _pick_local_maxima(
    signal: np.ndarray,
    *,
    thresh: float,
    min_distance: int,
) -> tuple[int, ...]:
    peaks: list[int] = []
    for i in range(1, signal.size - 1):
        if signal[i] < thresh:
            continue
        if signal[i] >= signal[i - 1] and signal[i] >= signal[i + 1]:
            if peaks and i - peaks[-1] < min_distance:
                if signal[i] > signal[peaks[-1]]:
                    peaks[-1] = i
            else:
                peaks.append(i)
    return tuple(peaks)


def score_boundaries(
    pred_bars: tuple[int, ...],
    gt_bars: tuple[int, ...],
    *,
    tolerance_bars: int = 1,
) -> BoundaryScore:
    if not gt_bars:
        return BoundaryScore(
            tolerance_bars=tolerance_bars,
            n_gt=0,
            n_pred=len(pred_bars),
            tp=0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
        )
    matched_gt: set[int] = set()
    tp = 0
    for p in pred_bars:
        for g in gt_bars:
            if abs(p - g) <= tolerance_bars and g not in matched_gt:
                tp += 1
                matched_gt.add(g)
                break
    precision = tp / len(pred_bars) if pred_bars else 0.0
    recall = tp / len(gt_bars)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return BoundaryScore(
        tolerance_bars=tolerance_bars,
        n_gt=len(gt_bars),
        n_pred=len(pred_bars),
        tp=tp,
        precision=precision,
        recall=recall,
        f1=f1,
    )
