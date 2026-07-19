"""Chroma observation reuses the peak-sparsified cosine path."""

from __future__ import annotations

import numpy as np

from workspaces.alignment_prototype.fp_segments.chroma_retrieve import (
    retrieve_chroma_matches,
)
from workspaces.alignment_prototype.fp_segments.run import _resolve_observation


def test_retrieve_chroma_matches_from_synthetic_features():
    d = 12
    tm, tr = 16, 10
    mix = np.zeros((tm, d), dtype=np.float32)
    ref = np.zeros((tr, d), dtype=np.float32)
    for i in range(tm):
        mix[i, i % d] = 1.0
    for j in range(tr):
        ref[j, j % d] = 1.0
    matches = retrieve_chroma_matches(
        mix,
        ref,
        recording_id="r-chroma",
        ref_stem="instrumental",
        mix_channel="instrumental",
        bin_s=0.5,
        already_pooled=True,
    )
    assert matches
    assert all(m.mix_channel == "instrumental" for m in matches)


def test_instrumental_accepts_chroma_observation():
    assert _resolve_observation("instrumental", "chroma") == "chroma"
    assert _resolve_observation("instrumental", None) == "landmark"
