"""HuBERT cosine similarity → sparse time correspondences for the vocal lane."""

from __future__ import annotations

import numpy as np

from workspaces.alignment_prototype.trajectory.features import (
    BIN_S,
    FPS_MATCH,
    HUBERT_LAYER,
    pool_bins,
)

from .schema import LandmarkMatch

# Frozen from synthetic planted-ridge / flat-noise unit gates. Do not retune
# on BB11/BB12.
DEFAULT_PEAK_FRAC = 0.5
DEFAULT_NEIGHBORHOOD = 3
DEFAULT_MAX_PEAKS = 256
DEFAULT_MIN_CONTRAST = 0.15


def cosine_similarity(mix_bins: np.ndarray, ref_bins: np.ndarray) -> np.ndarray:
    """(Tm,D)+(Tr,D) L2-normalized rows → (Tm,Tr) cosine."""
    if mix_bins.ndim != 2 or ref_bins.ndim != 2:
        raise ValueError("mix_bins and ref_bins must be 2-D")
    if mix_bins.shape[1] != ref_bins.shape[1]:
        raise ValueError("mix/ref feature dimensions must match")
    if mix_bins.shape[0] == 0 or ref_bins.shape[0] == 0:
        return np.zeros((mix_bins.shape[0], ref_bins.shape[0]), dtype=np.float32)
    return (mix_bins @ ref_bins.T).astype(np.float32)


def sparsify_similarity(
    matrix: np.ndarray,
    *,
    peak_frac: float = DEFAULT_PEAK_FRAC,
    neighborhood: int = DEFAULT_NEIGHBORHOOD,
    max_peaks: int = DEFAULT_MAX_PEAKS,
    min_contrast: float = DEFAULT_MIN_CONTRAST,
) -> tuple[tuple[int, int, float], ...]:
    """Keep local maxima above a relative peak floor.

    Returns ``(mix_bin, ref_bin, weight)`` tuples sorted by descending weight.
    Flat / low-contrast matrices abstain.
    """
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-D")
    if not 0.0 < peak_frac <= 1.0:
        raise ValueError("peak_frac must be in (0, 1]")
    if neighborhood < 1 or neighborhood % 2 == 0:
        raise ValueError("neighborhood must be a positive odd integer")
    if max_peaks < 1:
        raise ValueError("max_peaks must be positive")
    if min_contrast < 0.0:
        raise ValueError("min_contrast must be non-negative")

    m = np.asarray(matrix, dtype=np.float32)
    if m.size == 0:
        return ()
    peak = float(np.max(m))
    if not np.isfinite(peak) or peak <= 0.0:
        return ()
    contrast = peak - float(np.median(m))
    if contrast < min_contrast:
        return ()

    threshold = peak_frac * peak
    radius = neighborhood // 2
    tm, tr = m.shape
    candidates: list[tuple[float, int, int]] = []
    for i in range(tm):
        for j in range(tr):
            value = float(m[i, j])
            if value < threshold or not np.isfinite(value):
                continue
            i0, i1 = max(0, i - radius), min(tm, i + radius + 1)
            j0, j1 = max(0, j - radius), min(tr, j + radius + 1)
            if value < float(np.max(m[i0:i1, j0:j1])):
                continue
            candidates.append((value, i, j))

    candidates.sort(reverse=True)
    return tuple((i, j, w) for w, i, j in candidates[:max_peaks])


def matches_from_similarity(
    matrix: np.ndarray,
    *,
    recording_id: str,
    ref_stem: str,
    mix_channel: str,
    bin_s: float = BIN_S,
    peak_frac: float = DEFAULT_PEAK_FRAC,
    neighborhood: int = DEFAULT_NEIGHBORHOOD,
    max_peaks: int = DEFAULT_MAX_PEAKS,
    min_contrast: float = DEFAULT_MIN_CONTRAST,
) -> tuple[LandmarkMatch, ...]:
    """Map sparsified similarity peaks to ``LandmarkMatch`` correspondences."""
    if bin_s <= 0.0 or not np.isfinite(bin_s):
        raise ValueError("bin_s must be positive and finite")
    peaks = sparsify_similarity(
        matrix,
        peak_frac=peak_frac,
        neighborhood=neighborhood,
        max_peaks=max_peaks,
        min_contrast=min_contrast,
    )
    half = 0.5 * bin_s
    return tuple(
        LandmarkMatch(
            recording_id=recording_id,
            ref_stem=ref_stem,
            mix_channel=mix_channel,
            mix_time_s=float(mi) * bin_s + half,
            ref_time_s=float(ri) * bin_s + half,
            weight=float(weight),
            hash_frequency=1,
        )
        for mi, ri, weight in peaks
    )


def retrieve_hubert_matches(
    mix_feat: np.ndarray,
    ref_feat: np.ndarray,
    *,
    recording_id: str,
    ref_stem: str,
    mix_channel: str,
    bin_s: float = BIN_S,
    already_pooled: bool = False,
    peak_frac: float = DEFAULT_PEAK_FRAC,
    neighborhood: int = DEFAULT_NEIGHBORHOOD,
    max_peaks: int = DEFAULT_MAX_PEAKS,
    min_contrast: float = DEFAULT_MIN_CONTRAST,
) -> tuple[LandmarkMatch, ...]:
    """Build sparse HuBERT correspondences from feature matrices.

    When ``already_pooled`` is false, ``mix_feat`` / ``ref_feat`` are raw
    ``(D, T)`` HuBERT frames on the aligner grid. When true, they are
    ``(T, D)`` L2-normalized pooled bins.
    """
    if already_pooled:
        mix_bins = np.asarray(mix_feat, dtype=np.float32)
        ref_bins = np.asarray(ref_feat, dtype=np.float32)
    else:
        mix_bins = pool_bins(np.asarray(mix_feat, dtype=np.float32), FPS_MATCH, bin_s)
        ref_bins = pool_bins(np.asarray(ref_feat, dtype=np.float32), FPS_MATCH, bin_s)
    matrix = cosine_similarity(mix_bins, ref_bins)
    return matches_from_similarity(
        matrix,
        recording_id=recording_id,
        ref_stem=ref_stem,
        mix_channel=mix_channel,
        bin_s=bin_s,
        peak_frac=peak_frac,
        neighborhood=neighborhood,
        max_peaks=max_peaks,
        min_contrast=min_contrast,
    )


def load_hubert_feat(
    audio_path: str | object, *, layer: int = HUBERT_LAYER
) -> np.ndarray:
    """Load cached HuBERT ``(D, T)`` features for an audio path."""
    from workspaces.alignment_prototype.path_decode import _ensure_feat

    path = _ensure_feat(str(audio_path), str(audio_path), "hubert", layer)
    return np.load(path)
