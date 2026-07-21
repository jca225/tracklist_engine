"""Tests for the learned + ensemble identity verifier."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.identity_learned import (
    LearnedSongVerifier,
    LearnedVersionVerifier,
    RecordingFeatures,
    build_song_pairs,
    build_version_pairs,
    ensemble_verify_version,
    extract_pair_features,
)
from analysis.identity_verify import AxisVerdict, EssentiaFeatures, VersionVerdict


def _rec(
    recording_id: str,
    work_id: str | None,
    version: str = "original",
    artist: str = "Artist",
    title: str = "Title",
    mert_seed: int = 0,
) -> RecordingFeatures:
    rng = np.random.default_rng(mert_seed)
    return RecordingFeatures(
        recording_id=recording_id,
        work_id=work_id,
        artist=artist,
        title=title,
        version=version,
        mert_mean=rng.normal(size=32).astype(np.float32),
    )


def test_extract_pair_features_computes_mert_cosine():
    a = _rec("r1", "w1", mert_seed=1)
    b = _rec("r2", "w1", mert_seed=1)
    # Same seed -> very similar (but not identical random) vectors.
    f = extract_pair_features(a, b)
    assert f.mert_cosine is not None
    assert -1.0 <= f.mert_cosine <= 1.0


def test_extract_pair_features_text_and_version():
    a = _rec("r1", "w1", version="remix", artist="Don Diablo", title="Momentum")
    b = _rec("r2", "w1", version="original", artist="Don Diablo", title="Momentum")
    f = extract_pair_features(a, b)
    assert f.version_match == 0
    assert f.artist_overlap > 0
    assert f.title_overlap > 0


def test_build_version_pairs_excludes_different_work():
    recordings = [
        _rec("r1", "w1", version="original"),
        _rec("r2", "w1", version="remix"),
        _rec("r3", "w2", version="original"),
    ]
    X, y = build_version_pairs(recordings)
    assert X.shape[0] == 1
    assert y[0] == 0  # same work, different version


def test_build_song_pairs_includes_different_work():
    recordings = [
        _rec("r1", "w1", version="original"),
        _rec("r2", "w1", version="remix"),
        _rec("r3", "w2", version="original"),
    ]
    X, y = build_song_pairs(recordings)
    assert X.shape[0] == 3
    assert sorted(y.tolist()) == [0, 0, 1]


def test_learned_version_verifier_trains_and_predicts():
    recordings = [
        _rec("r1", "w1", version="original", mert_seed=1),
        _rec("r2", "w1", version="remix", mert_seed=2),
        _rec("r3", "w1", version="original", mert_seed=1),
        _rec("r4", "w1", version="remix", mert_seed=2),
        _rec("r5", "w2", version="original", mert_seed=3),
    ]
    X, y = build_version_pairs(recordings)
    assert X.shape[0] >= 3
    verifier = LearnedVersionVerifier().fit(X, y)

    ref = _rec("r_ref", "w1", version="original", mert_seed=1)
    cand_same = _rec("r_same", "w1", version="original", mert_seed=1)
    cand_diff = _rec("r_diff", "w1", version="remix", mert_seed=2)

    verdict_same = verifier.verify(ref, cand_same)
    verdict_diff = verifier.verify(ref, cand_diff)
    assert verdict_same in (VersionVerdict.SAME_VERSION, VersionVerdict.AMBIGUOUS)
    assert verdict_diff in (VersionVerdict.DIFFERENT_VERSION, VersionVerdict.AMBIGUOUS)


def test_learned_song_verifier_gates_different_song():
    recordings = [
        _rec("r1", "w1", mert_seed=1),
        _rec("r2", "w1", mert_seed=1),
        _rec("r3", "w2", mert_seed=3),
    ]
    X, y = build_song_pairs(recordings)
    song_verifier = LearnedSongVerifier().fit(X, y)

    ref = _rec("r_ref", "w1", mert_seed=1)
    same = _rec("r_same", "w1", mert_seed=1)
    diff = _rec("r_diff", "w2", mert_seed=3)

    assert song_verifier.verify(ref, same) in (AxisVerdict.SAME, AxisVerdict.AMBIGUOUS)
    assert song_verifier.verify(ref, diff) in (
        AxisVerdict.DIFFERENT,
        AxisVerdict.AMBIGUOUS,
    )


def test_ensemble_without_model_is_ambiguous():
    ref = _rec("r1", "w1", version="original")
    cand = _rec("r2", "w1", version="remix")
    assert ensemble_verify_version(ref, cand) is VersionVerdict.AMBIGUOUS


def test_untrained_verifier_returns_ambiguous():
    v = LearnedVersionVerifier()
    ref = _rec("r1", "w1", version="original")
    cand = _rec("r2", "w1", version="remix")
    assert v.verify(ref, cand) is VersionVerdict.AMBIGUOUS


def test_missing_features_are_usable_if_any_signal_present():
    a = RecordingFeatures(recording_id="r1", mert_mean=np.ones(4, dtype=np.float32))
    b = RecordingFeatures(recording_id="r2", mert_mean=np.ones(4, dtype=np.float32))
    f = extract_pair_features(a, b)
    assert f.is_usable() is True
    assert f.mert_cosine == pytest.approx(1.0, abs=1e-5)


def test_completely_missing_features_are_not_usable():
    a = RecordingFeatures(recording_id="r1")
    b = RecordingFeatures(recording_id="r2")
    f = extract_pair_features(a, b)
    assert f.is_usable() is False
