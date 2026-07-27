"""Instrumental chroma cosine similarity → sparse segment correspondences."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from alignment.trajectory.features import (
    BIN_S,
    FPS_MATCH,
    pool_bins,
)

from .hubert_retrieve import matches_from_similarity, retrieve_hubert_matches
from .schema import LandmarkMatch


def load_chroma_feat(audio_path: str | Path) -> np.ndarray:
    """Load cached chroma ``(D, T)`` features for an audio path."""
    from alignment.path_decode import _ensure_feat

    path = _ensure_feat(str(audio_path), str(audio_path), "chroma", 9)
    return np.load(path)


def retrieve_chroma_matches(
    mix_feat: np.ndarray,
    ref_feat: np.ndarray,
    *,
    recording_id: str,
    ref_stem: str,
    mix_channel: str,
    bin_s: float = BIN_S,
    already_pooled: bool = False,
) -> tuple[LandmarkMatch, ...]:
    """Same peak-sparsified cosine path as HuBERT, on chroma features."""
    # Reuse the HuBERT retrieve body: pooling + cosine + sparsify are feature-agnostic.
    return retrieve_hubert_matches(
        mix_feat,
        ref_feat,
        recording_id=recording_id,
        ref_stem=ref_stem,
        mix_channel=mix_channel,
        bin_s=bin_s,
        already_pooled=already_pooled,
    )


def pooled_chroma(audio_path: str | Path, *, bin_s: float = BIN_S) -> np.ndarray:
    """Convenience: load chroma frames and pool to ``(T, D)`` bins."""
    feat = load_chroma_feat(audio_path)
    return pool_bins(np.asarray(feat, dtype=np.float32), FPS_MATCH, bin_s)


# Re-export for callers that only need the matrix→matches step.
__all__ = [
    "load_chroma_feat",
    "matches_from_similarity",
    "pooled_chroma",
    "retrieve_chroma_matches",
]
