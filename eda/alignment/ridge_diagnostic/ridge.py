"""GT diagonal overlay and ridge contrast for the placement ridge diagnostic."""

from __future__ import annotations

import numpy as np

_EPS = 1e-8


def gt_diagonal_mask(
    shape: tuple[int, int],
    *,
    offsets_s: tuple[float, ...],
    bin_s: float,
    band_bins: int = 1,
    mix_origin_s: float = 0.0,
    ref_origin_s: float = 0.0,
) -> np.ndarray:
    """True on cells within band_bins of any GT diagonal.

    offset_s semantics: mix_time ≈ ref_time + offset
    where offset = gt_audible_onset_s - gt_ref_start_s for a straight span.
    For each ref_segment (mix_start, ref_start, ref_end), add
    offset = mix_start - ref_start.
    """
    tm, tr = shape
    mask = np.zeros((tm, tr), dtype=bool)
    origin_shift_bins = (ref_origin_s - mix_origin_s) / bin_s

    mix_bins = np.arange(tm, dtype=np.float64)
    ref_bins = np.arange(tr, dtype=np.float64)

    for offset_s in offsets_s:
        expected_ref = mix_bins + origin_shift_bins + offset_s / bin_s
        dist = np.abs(ref_bins[np.newaxis, :] - expected_ref[:, np.newaxis])
        mask |= dist <= float(band_bins)

    return mask


def ridge_contrast(M: np.ndarray, mask: np.ndarray) -> float:
    """mean(M[mask]) / (mean(M[~mask]) + eps). Higher = clearer true ridge."""
    if not np.any(mask):
        return 0.0
    on = float(np.mean(M[mask]))
    off_mask = ~mask
    off = float(np.mean(M[off_mask])) if np.any(off_mask) else 0.0
    return on / (off + _EPS)
