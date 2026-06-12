"""M0 (memoryless) and M1 (adaptive Markov) rungs of the ladder.

Both are exactly prequential by construction: every readout at frame ``t``
depends only on frames ``< t``.
"""
from __future__ import annotations

import numpy as np

from eda.alignment.adaptive_markov import AdaptiveMarkovChain

from .data import StudyData, normalized_mert
from .signals import SignalSet


def run_m0(data: StudyData, *, alpha: float = 0.5) -> SignalSet:
    """Memoryless null: online marginal token model + embedding persistence.

    - ``surprisal`` / ``entropy``: running marginal (add-alpha) over tokens seen
      so far — no order, no context. This is the genuine null hypothesis.
    - ``persist_cosdist``: 1 - cos(x_t, x_{t-1}). A purely local novelty detector
      ("predict t = t-1" in embedding space); strong-but-dumb boundary baseline.
    """
    tokens = data.tokens
    n = len(tokens)
    K = data.n_tokens

    counts = np.full(K, alpha, dtype=np.float64)
    surprisal = np.full(n, np.nan)
    entropy = np.full(n, np.nan)
    for t in range(n):
        probs = counts / counts.sum()
        surprisal[t] = -np.log(max(float(probs[tokens[t]]), 1e-12))
        entropy[t] = float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))))
        counts[tokens[t]] += 1.0

    x = normalized_mert(data)
    persist = np.full(n, np.nan)
    cos = np.sum(x[1:] * x[:-1], axis=1)
    persist[1:] = 1.0 - cos

    return SignalSet(
        model="M0",
        n_frames=n,
        signals={
            "surprisal": surprisal,
            "entropy": entropy,
            "persist_cosdist": persist,
        },
    )


def run_m1(
    data: StudyData,
    *,
    alpha: float = 0.5,
    beta: float = 0.01,
) -> SignalSet:
    """Adaptive first-order Markov chain (Abdallah & Plumbley replication).

    Reuses :class:`AdaptiveMarkovChain`. Adds the *exact* predictive
    information ``I(x|z) = D(p(next|x,z) || p(next|z))`` (the chain's own
    ``_predictive_information``, computed pre-update so it stays causal), which
    the stock readout does not expose.
    """
    tokens = data.tokens
    n = len(tokens)
    chain = AdaptiveMarkovChain(data.n_tokens, alpha=alpha, beta=beta)

    surprisal = np.full(n, np.nan)
    entropy = np.full(n, np.nan)
    mir = np.full(n, np.nan)
    pir_proxy = np.full(n, np.nan)
    pred_info = np.full(n, np.nan)

    for t in range(n):
        sym = int(tokens[t])
        prev = chain._prev
        if prev is not None:
            probs = chain._column_probs(prev)
            pred_info[t] = chain._predictive_information(prev, sym, probs)
        r = chain.observe(sym)
        if prev is not None:  # frame 0 has no context — leave NaN
            surprisal[t] = r.surprisingness
            entropy[t] = r.predictive_uncertainty
            mir[t] = r.model_information_rate
            pir_proxy[t] = r.predictive_information_rate

    return SignalSet(
        model="M1",
        n_frames=n,
        signals={
            "surprisal": surprisal,
            "entropy": entropy,
            "mir": mir,
            "pir_proxy": pir_proxy,
            "pred_info": pred_info,
        },
    )
