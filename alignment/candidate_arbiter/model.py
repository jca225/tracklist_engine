"""Small baseline-aware candidate critic and deterministic feature schema."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import CandidateExample
from .schema import PlacementCandidate

SOURCES = (
    "cue_prior",
    "fp",
    "instr_fp",
    "lyrics",
    "mert_decode",
    "stem_hubert",
    "chroma_refine",
    "surprise",
    "other",
)
STEMS = ("regular", "acappella", "instrumental")
BASELINE_SOURCES = (
    "mert",
    "fp",
    "lyrics",
    "instr_fp",
    "agentic",
    "synthetic_coarse",
    "other",
)
OPTIONAL_NUMERIC = (
    "native_confidence",
    "ridge_peak",
    "ridge_second",
    "ridge_margin",
    "ridge_prominence",
    "ridge_support_s",
    "vote_count",
    "vote_density",
    "runner_up_ratio",
    "cue_delta_s",
    "mert_delta_s",
    "neighbor_gap_left_s",
    "neighbor_gap_right_s",
    "active_layer_count",
    "repeat_ambiguity",
    "verify_match_mean",
    "verify_match_p10",
    "verify_margin",
    "verify_continuity",
    "verify_support_bins",
    "verify_background_count",
)
REQUIRED_NUMERIC = (
    "baseline_delta_s",
    "agreeing_source_count",
    "independent_groups",
)


class CandidateVectorizer:
    """Fixed feature mapping; missing sensor values get explicit mask columns."""

    def __init__(self) -> None:
        self.feature_names = (
            *REQUIRED_NUMERIC,
            *OPTIONAL_NUMERIC,
            *(f"missing:{name}" for name in OPTIONAL_NUMERIC),
            *(f"source:{source}" for source in SOURCES),
            *(f"baseline_source:{source}" for source in BASELINE_SOURCES),
            *(f"stem:{stem}" for stem in STEMS),
        )

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @staticmethod
    def _optional(candidate: PlacementCandidate, name: str) -> float | None:
        if name == "native_confidence":
            return candidate.native_confidence
        return getattr(candidate.evidence, name)

    def transform(
        self,
        candidates: tuple[PlacementCandidate, ...],
    ) -> np.ndarray:
        rows: list[list[float]] = []
        for candidate in candidates:
            evidence = candidate.evidence
            row = [
                float(evidence.baseline_delta_s),
                float(len(evidence.agreeing_sources)),
                float(evidence.independent_groups),
            ]
            values = [
                self._optional(candidate, name) for name in OPTIONAL_NUMERIC
            ]
            row.extend(float(value) if value is not None else 0.0 for value in values)
            row.extend(1.0 if value is None else 0.0 for value in values)
            source = candidate.source if candidate.source in SOURCES else "other"
            row.extend(1.0 if source == name else 0.0 for name in SOURCES)
            raw_baseline_source = candidate.evidence.baseline_source or "other"
            baseline_source = next(
                (
                    name
                    for name in BASELINE_SOURCES
                    if name != "other" and name in raw_baseline_source
                ),
                "other",
            )
            row.extend(
                1.0 if baseline_source == name else 0.0
                for name in BASELINE_SOURCES
            )
            row.extend(
                1.0 if candidate.claimed_stem == stem else 0.0 for stem in STEMS
            )
            rows.append(row)
        if not rows:
            return np.zeros((0, self.n_features), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)


@dataclass(frozen=True)
class CandidateCritic:
    vectorizer: CandidateVectorizer
    estimator: Pipeline

    def predict_proba(
        self,
        candidates: tuple[PlacementCandidate, ...],
    ) -> np.ndarray:
        values = self.estimator.predict_proba(
            self.vectorizer.transform(candidates)
        )
        return np.asarray(values[:, 1], dtype=np.float32)


def train_critic(
    examples: tuple[CandidateExample, ...],
    *,
    false_accept_weight: float = 4.0,
) -> CandidateCritic:
    """Fit the smallest precision-first baseline: scaled logistic regression."""
    if not examples:
        raise ValueError("cannot train critic without examples")
    labels = np.asarray(
        [example.label.improves_baseline for example in examples],
        dtype=np.int64,
    )
    if np.unique(labels).size != 2:
        raise ValueError("critic training requires positive and negative examples")
    vectorizer = CandidateVectorizer()
    estimator = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    random_state=0,
                    max_iter=1000,
                    solver="liblinear",
                ),
            ),
        ]
    )
    weights = np.where(labels == 0, false_accept_weight, 1.0)
    estimator.fit(
        vectorizer.transform(tuple(example.candidate for example in examples)),
        labels,
        model__sample_weight=weights,
    )
    return CandidateCritic(vectorizer=vectorizer, estimator=estimator)
