"""Tests for landmark fingerprint serialize + offset."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("librosa")  # landmark_fp computes the STFT via librosa at runtime

from workspaces.alignment_prototype.landmark_fp import (
    LandmarkFingerprint,
    fingerprint_from_audio,
    fp_offset,
    vote_sharpness,
)


def test_landmark_roundtrip_blob() -> None:
    rng = np.random.default_rng(0)
    y = rng.standard_normal(22050 * 8).astype(np.float32) * 0.05
    fp = fingerprint_from_audio(y)
    back = LandmarkFingerprint.from_blob(fp.to_blob())
    assert back.duration_s == fp.duration_s
    assert len(back.hashes) == len(fp.hashes)


def test_vote_sharpness_single_peak() -> None:
    assert vote_sharpness({3: 10}) == 10.0


def test_fp_offset_self_match_has_votes() -> None:
    rng = np.random.default_rng(1)
    y = rng.standard_normal(22050 * 20).astype(np.float32) * 0.08
    off, votes, _stretch, sharp = fp_offset(y, y)
    assert votes > 0
    assert sharp >= 1.0
    assert abs(off) < 2.0
