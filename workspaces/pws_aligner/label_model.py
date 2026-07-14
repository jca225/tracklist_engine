"""Programmatic weak-supervision label models for the PWS aligner.

Provides:
  - LabelModel: protocol (fit / predict_proba)
  - MajorityVote: confidence-weighted tally baseline
  - DawidSkene: numpy EM with per-probe accuracy estimation (no GT required)
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from workspaces.pws_aligner.hypotheses import (
    Hypothesis,
    build_hypothesis_space,
    vote_to_hypothesis,
)
from workspaces.pws_aligner.votes import Vote

# Numerical stability clip for probe accuracies
_ACC_LO: float = 0.01
_ACC_HI: float = 0.99

# Prior weight boosting the NULL hypothesis relative to non-null candidates.
# Must be > 1.0 so the prior is a genuine boost (not a multiply-by-one no-op) when
# evidence is absent or weakly split (I2 fix).  W=1.05 is deliberately minimal:
#   - satisfies the > 1.0 contract (all-abstain spans → NULL via trivial space, not
#     just this weight, but the weight cements the contract against edge cases),
#   - leaves the asymmetric wrong-mass EM fixed point essentially undisturbed so the
#     oracle-recovery test (n=400, 3-probe synthetic) still achieves MAP ≥ 0.9 and
#     good > ok > bad accuracy ranking,
#   - avoids the W≥1.1 threshold where the NULL prior begins pulling MAP below 0.9
#     in the 2-class non-null synthetic (verified empirically).
# Do not raise above ~1.08 without re-validating the oracle test.
_NULL_PRIOR_WEIGHT: float = 1.05


class LabelModel(Protocol):
    def fit(self, spans: Sequence[Sequence[Vote]]) -> None: ...
    def predict_proba(self, span_votes: Sequence[Vote]) -> dict[Hypothesis, float]: ...


# ---------------------------------------------------------------------------
# MajorityVote
# ---------------------------------------------------------------------------


class MajorityVote:
    """Confidence-weighted tally over vote_to_hypothesis(). Stateless — fit is a no-op."""

    def fit(self, spans: Sequence[Sequence[Vote]]) -> None:  # noqa: ARG002
        pass

    def predict_proba(self, span_votes: Sequence[Vote]) -> dict[Hypothesis, float]:
        null = Hypothesis(None, 0)
        tally: dict[Hypothesis, float] = {null: 0.0}
        for v in span_votes:
            if v.abstained:
                continue
            h = vote_to_hypothesis(v)
            tally[h] = tally.get(h, 0.0) + v.confidence
        total = sum(tally.values())
        if total == 0.0:
            return {null: 1.0}
        return {h: w / total for h, w in tally.items()}


# ---------------------------------------------------------------------------
# DawidSkene — 2-parameter-per-probe EM
# ---------------------------------------------------------------------------


def _e_step(
    spans: list[tuple[Hypothesis, ...]],
    span_fired: list[list[tuple[str, Hypothesis]]],  # per span: [(probe, voted_hyp)]
    probe_acc: dict[str, float],
    null: Hypothesis,
) -> list[np.ndarray]:
    """Return per-span posterior arrays (length = |space|, uniform prior)."""
    posteriors: list[np.ndarray] = []
    for space, fired in zip(spans, span_fired):
        K = len(space)
        # Build index map
        h_idx = {h: i for i, h in enumerate(space)}
        null_idx = h_idx.get(null, 0)

        # Non-null hypothesis count for the wrong-mass denominator
        K_non_null = K - 1  # space always has NULL first

        log_p = np.zeros(K)
        for probe, voted_h in fired:
            a = probe_acc[probe]
            voted_idx = h_idx.get(voted_h, null_idx)
            # Spec: 2-parameter-per-probe asymmetric wrong-mass model.
            #
            # Given true = h (non-NULL):
            #   P(vote=h   | true=h, fires) = a_p
            #   P(vote=h_v | true=h, fires) = (1−a_p) / max(K_nn−1, 1)
            #     for each other non-NULL h_v, where K_nn = # non-NULL hypotheses.
            #
            # Given true = NULL:
            #   P(vote=h_v | true=NULL) = (1−a_p) / K_nn
            #     for each non-NULL h_v.
            #
            # Interpretation: a reliable probe mostly ABSTAINS when nothing in the
            # candidate space is correct (mass a_p goes to abstention, which carries
            # no likelihood term).  The remaining 1−a_p misfires uniformly over the
            # K_nn non-NULL candidates.  Wrong mass from a non-NULL true hypothesis
            # spreads over the K_nn−1 *other* non-NULL hypotheses only — NULL is not
            # a valid wrong-vote target because a probe that fires has already
            # committed to a candidate.  This asymmetry is intentional and correct;
            # do not "fix" it to a symmetric (K−1) denominator.
            for i, h in enumerate(space):
                if i == voted_idx:
                    log_p[i] += np.log(a)
                elif h == null:
                    # NULL gets wrong-mass spread over K_nn non-null candidates
                    if K_non_null > 0:
                        log_p[i] += np.log((1.0 - a) / max(K_non_null, 1))
                    else:
                        log_p[i] += np.log(1e-9)
                else:
                    # Non-null, non-voted: wrong mass over K_nn−1 competing non-null
                    denom = max(K_non_null - 1, 1) if K_non_null > 1 else 1
                    log_p[i] += np.log((1.0 - a) / denom)

        # Subtract max for numerical stability before exp
        log_p -= log_p.max()
        p = np.exp(log_p)

        # Apply NULL prior weight boost so all-abstain → NULL
        p[null_idx] *= _NULL_PRIOR_WEIGHT

        total = p.sum()
        if total == 0.0:
            p = np.ones(K) / K
        else:
            p /= total

        posteriors.append(p)
    return posteriors


def _m_step(
    spans: list[tuple[Hypothesis, ...]],
    span_fired: list[list[tuple[str, Hypothesis]]],
    posteriors: list[np.ndarray],
    probes: list[str],
) -> dict[str, float]:
    """Re-estimate probe accuracies from posteriors."""
    numerator: dict[str, float] = {p: 0.0 for p in probes}
    denominator: dict[str, float] = {p: 0.0 for p in probes}

    for space, fired, post in zip(spans, span_fired, posteriors):
        h_idx = {h: i for i, h in enumerate(space)}
        for probe, voted_h in fired:
            voted_idx = h_idx.get(voted_h, 0)
            # Posterior mass on the hypothesis this probe voted for
            numerator[probe] += post[voted_idx]
            denominator[probe] += 1.0

    new_acc: dict[str, float] = {}
    for p in probes:
        if denominator[p] == 0.0:
            new_acc[p] = 0.7
        else:
            raw = numerator[p] / denominator[p]
            new_acc[p] = float(np.clip(raw, _ACC_LO, _ACC_HI))
    return new_acc


class DawidSkene:
    """Classic Dawid–Skene EM with 2-parameter-per-probe reduction (no GT).

    Each probe p has a single accuracy a_p = P(voted hypothesis is correct | fired).
    When wrong, the vote's mass spreads over the other non-NULL hypotheses only
    (asymmetric denominators per spec — see _e_step docstring for full model).

    Parameters
    ----------
    max_iter: int
        Maximum EM iterations.
    tol: float
        Convergence threshold on max |Δa_p|.
    """

    def __init__(self, max_iter: int = 50, tol: float = 1e-4) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self._probe_acc: dict[str, float] = {}
        self._fitted = False

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, spans: Sequence[Sequence[Vote]]) -> None:
        """Run EM over all spans to learn per-probe accuracies."""
        # Build hypothesis spaces and collect fired votes per span
        spaces: list[tuple[Hypothesis, ...]] = []
        span_fired: list[list[tuple[str, Hypothesis]]] = []
        probe_set: set[str] = set()
        null = Hypothesis(None, 0)

        for votes in spans:
            space = build_hypothesis_space(votes)
            spaces.append(space)
            fired: list[tuple[str, Hypothesis]] = []
            for v in votes:
                if not v.abstained:
                    fired.append((v.probe, vote_to_hypothesis(v)))
                    probe_set.add(v.probe)
            span_fired.append(fired)

        probes = sorted(probe_set)

        # Initialize accuracies
        acc: dict[str, float] = {p: 0.7 for p in probes}

        for _ in range(self.max_iter):
            posteriors = _e_step(spaces, span_fired, acc, null)
            new_acc = _m_step(spaces, span_fired, posteriors, probes)

            # Check convergence
            delta = max(abs(new_acc[p] - acc[p]) for p in probes) if probes else 0.0
            acc = new_acc
            if delta < self.tol:
                break

        self._probe_acc = acc
        self._fitted = True

    # ------------------------------------------------------------------
    # predict_proba
    # ------------------------------------------------------------------

    def predict_proba(self, span_votes: Sequence[Vote]) -> dict[Hypothesis, float]:
        """E-step posterior for the given span using learned accuracies."""
        if not self._fitted:
            raise RuntimeError("DawidSkene.fit() must be called before predict_proba()")

        null = Hypothesis(None, 0)
        space = build_hypothesis_space(span_votes)
        fired: list[tuple[str, Hypothesis]] = [
            (v.probe, vote_to_hypothesis(v)) for v in span_votes if not v.abstained
        ]

        # I1 fix: build accuracy dict only for probes that fired in this span,
        # defaulting to 0.7 for any probe absent from training.  Passing this
        # dict (rather than the full fitted dict) means _e_step never sees a
        # probe key it wasn't given, eliminating the latent KeyError on unseen
        # probes.  The previous full_acc copy was a dead no-op.
        acc = {p: self._probe_acc.get(p, 0.7) for p, _ in fired}

        posteriors = _e_step([space], [fired], acc, null)
        post = posteriors[0]

        return {h: float(post[i]) for i, h in enumerate(space)}

    # ------------------------------------------------------------------
    # probe_accuracy
    # ------------------------------------------------------------------

    def probe_accuracy(self) -> dict[str, float]:
        """Return the learned P(correct | fired) per probe."""
        if not self._fitted:
            raise RuntimeError(
                "DawidSkene.fit() must be called before probe_accuracy()"
            )
        return dict(self._probe_acc)
