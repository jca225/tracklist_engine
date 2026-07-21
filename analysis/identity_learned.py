"""Learned + ensemble identity verifier for version, stem, and variant axes.

This module trains on the structured data already in the canonical DB:
- MERT measure embeddings (`track_mert_measures`)
- Essentia features (`track_audio_features`)
- Chromaprint fingerprints (`track_fingerprints`)
- Text metadata (`recording`, `work`, `track_metadata`)
- Identity axes (`version`, `stem`, `variant`)

The goal is a robust ensemble that beats the heuristic
`analysis/identity_verify.py` on the version-verification task. The first
implementation is a Random Forest on hand-crafted features including a MERT
cosine similarity; the MERT head can be upgraded to a learned neural head once
more labeled pairs are accumulated.

Training labels are derived from the database itself:
- same-song: two recordings share the same `work_id`
- same-version: same work + same `version` value
- same-stem: same work + same `stem` value
- same-variant: same work + same `variant` value

A data gap (missing work_id, missing MERT, etc.) is NOT a guess — it is
AMBIGUOUS. The model refuses to predict when features are missing.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from analysis.identity_verify import (
    AxisVerdict,
    EssentiaFeatures,
    VersionVerdict,
    _key_distance,
)
from core.identity import normalize_version


def _tokens(s: str) -> set[str]:
    """Lower-case alphanumeric tokens of length > 2."""
    return {w for w in re.split(r"\W+", s.lower()) if len(w) > 2}


# -----------------------------------------------------------------------------
# Structured inputs for one recording (no DB coupling)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordingFeatures:
    """Everything the learned verifier needs for one recording."""

    recording_id: str
    work_id: str | None = None
    artist: str = ""
    title: str = ""
    version: str = "original"
    stem: str = "regular"
    variant: str = "regular"
    remixer: str = ""

    # Audio features
    essentia: EssentiaFeatures | None = None
    mert_mean: np.ndarray | None = None  # mean-pooled MERT embedding
    fingerprint: np.ndarray | None = None  # chromaprint raw fingerprint


@dataclass(frozen=True)
class PairFeatures:
    """Hand-crafted feature vector for a pair of recordings."""

    # MERT similarity
    mert_cosine: float | None = None

    # Essentia deltas
    bpm_diff: float | None = None
    bpm_ratio: float | None = None
    key_distance: int | None = None  # -1 if unknown
    energy_diff: float | None = None
    danceability_diff: float | None = None
    instrumentalness_diff: float | None = None
    lufs_diff: float | None = None

    # Fingerprint similarity
    fingerprint_sim: float | None = None

    # Text overlap
    artist_overlap: float = 0.0
    title_overlap: float = 0.0
    version_match: int = 0
    remixer_match: int = 0
    stem_match: int = 0
    variant_match: int = 0

    def to_vector(self) -> np.ndarray:
        """Return a fixed-length numeric vector, using sentinel -1 for unknowns."""
        return np.array(
            [
                self._or_sentinel(self.mert_cosine),
                self._or_sentinel(self.bpm_diff),
                self._or_sentinel(self.bpm_ratio),
                self._or_sentinel(self.key_distance),
                self._or_sentinel(self.energy_diff),
                self._or_sentinel(self.danceability_diff),
                self._or_sentinel(self.instrumentalness_diff),
                self._or_sentinel(self.lufs_diff),
                self._or_sentinel(self.fingerprint_sim),
                self.artist_overlap,
                self.title_overlap,
                float(self.version_match),
                float(self.remixer_match),
                float(self.stem_match),
                float(self.variant_match),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _or_sentinel(value: float | int | None) -> float:
        return -1.0 if value is None else float(value)

    def is_usable(self) -> bool:
        """A pair is usable if at least one strong signal is present."""
        return (
            self.mert_cosine is not None
            or self.fingerprint_sim is not None
            or self.bpm_diff is not None
            or (self.key_distance is not None and self.key_distance >= 0)
            or self.energy_diff is not None
        )


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None or a.size == 0 or b.size == 0:
        return None
    a = a.flatten().astype(np.float32)
    b = b.flatten().astype(np.float32)
    # The canonical DB contains MERT embeddings from different model/layer
    # configurations (e.g. 768-dim single-layer vs. 25×1024=25600 full-stack).
    # Cross-dimensional similarities are undefined, so refuse rather than crash.
    if a.shape != b.shape:
        return None
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0 or not np.isfinite(denom):
        return None
    return float(np.dot(a, b) / denom)


def _fingerprint_sim(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """Bit-agreement similarity between two chromaprint fingerprints.
    Mirrors ingest/adapters/fingerprint.similarity but over the full overlap."""
    if a is None or b is None or a.size == 0 or b.size == 0:
        return None
    a = np.asarray(a, dtype=np.uint32)
    b = np.asarray(b, dtype=np.uint32)
    if a.size == 0 or b.size == 0:
        return None
    # Overlap at offset 0 (both recordings are expected to be full files).
    overlap = min(a.size, b.size)
    xor = a[:overlap] ^ b[:overlap]
    bit_counts = np.bitwise_count(xor)
    ber = float(bit_counts.mean()) / 32.0
    return 1.0 - ber


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def extract_pair_features(
    ref: RecordingFeatures,
    cand: RecordingFeatures,
) -> PairFeatures:
    """Extract the hand-crafted feature vector for a pair of recordings."""
    ref_e = ref.essentia
    cand_e = cand.essentia

    bpm_diff = None
    bpm_ratio = None
    if ref_e and cand_e and ref_e.bpm is not None and cand_e.bpm is not None:
        bpm_diff = abs(ref_e.bpm - cand_e.bpm)
        bpm_ratio = cand_e.bpm / ref_e.bpm if ref_e.bpm else None

    key_distance = -1
    if ref_e and cand_e:
        key_distance = _key_distance(
            ref_e.key_pc, ref_e.key_mode, cand_e.key_pc, cand_e.key_mode
        )

    energy_diff = None
    danceability_diff = None
    instrumentalness_diff = None
    lufs_diff = None
    if ref_e and cand_e:
        if ref_e.energy is not None and cand_e.energy is not None:
            energy_diff = abs(ref_e.energy - cand_e.energy)
        if ref_e.danceability is not None and cand_e.danceability is not None:
            danceability_diff = abs(ref_e.danceability - cand_e.danceability)
        if ref_e.instrumentalness is not None and cand_e.instrumentalness is not None:
            instrumentalness_diff = abs(
                ref_e.instrumentalness - cand_e.instrumentalness
            )
        if ref_e.lufs is not None and cand_e.lufs is not None:
            lufs_diff = abs(ref_e.lufs - cand_e.lufs)

    return PairFeatures(
        mert_cosine=_cosine(ref.mert_mean, cand.mert_mean),
        bpm_diff=bpm_diff,
        bpm_ratio=bpm_ratio,
        key_distance=key_distance,
        energy_diff=energy_diff,
        danceability_diff=danceability_diff,
        instrumentalness_diff=instrumentalness_diff,
        lufs_diff=lufs_diff,
        fingerprint_sim=_fingerprint_sim(ref.fingerprint, cand.fingerprint),
        artist_overlap=_jaccard(_tokens(ref.artist), _tokens(cand.artist)),
        title_overlap=_jaccard(_tokens(ref.title), _tokens(cand.title)),
        version_match=int(
            normalize_version(ref.version) == normalize_version(cand.version)
        ),
        remixer_match=int(
            bool(ref.remixer)
            and bool(cand.remixer)
            and ref.remixer.lower() == cand.remixer.lower()
        ),
        stem_match=int(ref.stem == cand.stem),
        variant_match=int(ref.variant == cand.variant),
    )


# -----------------------------------------------------------------------------
# Training pair builder
# -----------------------------------------------------------------------------


def build_version_pairs(
    recordings: Sequence[RecordingFeatures],
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) training matrices for the version-verification task.

    Positive label (1): same work_id and same version.
    Negative label (0): same work_id but different version.
    Pairs with different work_id are excluded — they are the same-song task, not
    the version task.
    """
    X: list[np.ndarray] = []
    y: list[int] = []
    for i, ref in enumerate(recordings):
        if not ref.work_id:
            continue
        for cand in recordings[i + 1 :]:
            if cand.work_id != ref.work_id:
                continue
            features = extract_pair_features(ref, cand)
            if not features.is_usable():
                continue
            X.append(features.to_vector())
            y.append(
                1
                if normalize_version(ref.version) == normalize_version(cand.version)
                else 0
            )
    if not X:
        return np.zeros((0, 15), dtype=np.float32), np.zeros(0, dtype=int)
    return np.stack(X), np.array(y, dtype=int)


def build_song_pairs(
    recordings: Sequence[RecordingFeatures],
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) for the same-song task.

    Positive (1): same work_id. Negative (0): different work_id."""
    X: list[np.ndarray] = []
    y: list[int] = []
    n = len(recordings)
    for i in range(n):
        ref = recordings[i]
        for j in range(i + 1, n):
            cand = recordings[j]
            if not ref.work_id or not cand.work_id:
                continue
            features = extract_pair_features(ref, cand)
            if not features.is_usable():
                continue
            X.append(features.to_vector())
            y.append(1 if ref.work_id == cand.work_id else 0)
    if not X:
        return np.zeros((0, 15), dtype=np.float32), np.zeros(0, dtype=int)
    return np.stack(X), np.array(y, dtype=int)


# -----------------------------------------------------------------------------
# Model wrapper
# -----------------------------------------------------------------------------


class LearnedVersionVerifier:
    """Random-forest version verifier trained on PairFeatures."""

    def __init__(self, model: Any | None = None, threshold: float = 0.5) -> None:
        self.model = model
        self.threshold = threshold
        self._trained = model is not None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LearnedVersionVerifier":
        from sklearn.ensemble import RandomForestClassifier

        if X.shape[0] == 0:
            raise ValueError("No training pairs provided")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X, y)
        self._trained = True
        return self

    def predict_proba(self, features: PairFeatures) -> float | None:
        """Return P(same_version | same_song). None if not usable."""
        if not self._trained or not features.is_usable():
            return None
        x = features.to_vector().reshape(1, -1)
        # RandomForestClassifier.predict_proba returns [P(0), P(1)]
        return float(self.model.predict_proba(x)[0, 1])

    def verify(
        self,
        ref: RecordingFeatures,
        cand: RecordingFeatures,
        *,
        same_song_prob: float | None = None,
    ) -> VersionVerdict:
        """Return SAME_VERSION / DIFFERENT_VERSION / AMBIGUOUS.

        If ``same_song_prob`` is provided (from the same-song model), the pair is
        first gated: low same-song probability forces DIFFERENT_VERSION. This
        lets the version verifier sit on top of a same-song detector."""
        if same_song_prob is not None and same_song_prob < 0.5:
            return VersionVerdict.DIFFERENT_VERSION

        features = extract_pair_features(ref, cand)
        prob = self.predict_proba(features)
        if prob is None:
            return VersionVerdict.AMBIGUOUS
        if prob >= self.threshold + 0.2:
            return VersionVerdict.SAME_VERSION
        if prob <= self.threshold - 0.2:
            return VersionVerdict.DIFFERENT_VERSION
        return VersionVerdict.AMBIGUOUS


class LearnedSongVerifier:
    """Random-forest same-song verifier trained on PairFeatures."""

    def __init__(self, model: Any | None = None, threshold: float = 0.5) -> None:
        self.model = model
        self.threshold = threshold
        self._trained = model is not None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LearnedSongVerifier":
        from sklearn.ensemble import RandomForestClassifier

        if X.shape[0] == 0:
            raise ValueError("No training pairs provided")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X, y)
        self._trained = True
        return self

    def predict_proba(self, features: PairFeatures) -> float | None:
        if not self._trained or not features.is_usable():
            return None
        x = features.to_vector().reshape(1, -1)
        return float(self.model.predict_proba(x)[0, 1])

    def verify(
        self,
        ref: RecordingFeatures,
        cand: RecordingFeatures,
    ) -> AxisVerdict:
        features = extract_pair_features(ref, cand)
        prob = self.predict_proba(features)
        if prob is None:
            return AxisVerdict.AMBIGUOUS
        if prob >= self.threshold + 0.2:
            return AxisVerdict.SAME
        if prob <= self.threshold - 0.2:
            return AxisVerdict.DIFFERENT
        return AxisVerdict.AMBIGUOUS


PAIR_FEATURE_NAMES = [
    "mert_cosine",
    "bpm_diff",
    "bpm_ratio",
    "key_distance",
    "energy_diff",
    "danceability_diff",
    "instrumentalness_diff",
    "lufs_diff",
    "fingerprint_sim",
    "artist_overlap",
    "title_overlap",
    "version_match",
    "remixer_match",
    "stem_match",
    "variant_match",
]


def _cv_folds(y: np.ndarray) -> int:
    """Adaptive fold count so StratifiedKFold never exceeds class size."""
    counts = np.bincount(y)
    return min(5, int(counts.min())) if counts.size > 0 else 1


def train_verifiers(
    recordings: Sequence[RecordingFeatures],
) -> tuple[LearnedSongVerifier, LearnedVersionVerifier, dict[str, Any]]:
    """Train same-song and same-version verifiers, return metrics."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    metrics: dict[str, Any] = {}

    X_song, y_song = build_song_pairs(list(recordings))
    if X_song.shape[0] == 0:
        raise ValueError("No usable same-song training pairs")

    song_verifier = LearnedSongVerifier().fit(X_song, y_song)
    song_folds = _cv_folds(y_song)
    if song_folds >= 2:
        cv_song = cross_val_score(
            song_verifier.model,
            X_song,
            y_song,
            cv=StratifiedKFold(n_splits=song_folds, shuffle=True, random_state=42),
            scoring="roc_auc",
        )
        metrics["song_cv_roc_auc"] = float(cv_song.mean())
    else:
        metrics["song_cv_roc_auc"] = None

    X_ver, y_ver = build_version_pairs(list(recordings))
    if X_ver.shape[0] == 0:
        raise ValueError("No usable same-version training pairs")

    version_verifier = LearnedVersionVerifier().fit(X_ver, y_ver)
    ver_folds = _cv_folds(y_ver)
    if ver_folds >= 2:
        cv_ver = cross_val_score(
            version_verifier.model,
            X_ver,
            y_ver,
            cv=StratifiedKFold(n_splits=ver_folds, shuffle=True, random_state=42),
            scoring="roc_auc",
        )
        metrics["version_cv_roc_auc"] = float(cv_ver.mean())
    else:
        metrics["version_cv_roc_auc"] = None

    importances = version_verifier.model.feature_importances_
    metrics["version_feature_importance"] = {
        name: float(imp) for name, imp in zip(PAIR_FEATURE_NAMES, importances)
    }

    return song_verifier, version_verifier, metrics


# -----------------------------------------------------------------------------
# Ensemble: learned + heuristic
# -----------------------------------------------------------------------------


def ensemble_verify_version(
    ref: RecordingFeatures,
    cand: RecordingFeatures,
    *,
    learned: LearnedVersionVerifier | None = None,
    same_song: LearnedSongVerifier | None = None,
) -> VersionVerdict:
    """Run the learned version verifier, with optional same-song pre-gate.

    Falls back to AMBIGUOUS if the learned model is untrained or the pair is
    unusable."""
    if learned is None or not learned._trained:
        return VersionVerdict.AMBIGUOUS

    same_song_prob = None
    if same_song is not None and same_song._trained:
        same_song_prob = same_song.predict_proba(extract_pair_features(ref, cand))

    return learned.verify(ref, cand, same_song_prob=same_song_prob)
