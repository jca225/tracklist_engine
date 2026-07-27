from __future__ import annotations

import numpy as np

from alignment.candidate_arbiter.dataset import (
    CandidateExample,
    CandidateLabel,
)
from alignment.candidate_arbiter.model import (
    CandidateVectorizer,
    train_critic,
)
from alignment.candidate_arbiter.schema import (
    CandidateEvidence,
    PlacementCandidate,
)


def _example(source: str, delta: float, improves: bool) -> CandidateExample:
    candidate = PlacementCandidate(
        set_id="set",
        slot_label=str(abs(int(delta)) + 1),
        recording_id=f"rid-{source}-{delta}",
        claimed_stem="instrumental",
        source=source,
        rank=0,
        set_start_s=100.0 + delta,
        ref_start_s=None,
        native_confidence=0.8,
        evidence=CandidateEvidence(
            baseline_delta_s=delta,
            ridge_prominence=4.0 if improves else None,
            independent_groups=2 if improves else 1,
            verify_match_mean=0.9 if improves else 0.1,
            verify_margin=0.7 if improves else -0.1,
        ),
    )
    return CandidateExample(
        candidate=candidate,
        label=CandidateLabel(
            candidate_error_s=1.0 if improves else 20.0,
            baseline_error_s=10.0,
            improvement_s=9.0 if improves else -10.0,
            improves_baseline=improves,
            within_accept_tolerance=improves,
        ),
    )


def test_vectorizer_emits_missingness_mask() -> None:
    vectorizer = CandidateVectorizer()
    example = _example("fp", 10.0, improves=False)

    vector = vectorizer.transform((example.candidate,))

    assert vector.shape == (1, vectorizer.n_features)
    assert np.isfinite(vector).all()
    prominence = vectorizer.feature_names.index("ridge_prominence")
    missing = vectorizer.feature_names.index("missing:ridge_prominence")
    assert vector[0, prominence] == 0.0
    assert vector[0, missing] == 1.0


def test_logistic_critic_learns_separable_examples() -> None:
    examples = tuple(
        [_example("fp", float(i), improves=True) for i in range(1, 7)]
        + [_example("mert_decode", float(i), improves=False) for i in range(7, 13)]
    )

    critic = train_critic(examples)
    probabilities = critic.predict_proba(
        tuple(example.candidate for example in examples)
    )

    assert probabilities[:6].mean() > probabilities[6:].mean()
