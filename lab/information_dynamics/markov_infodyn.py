"""Closed-form information-dynamic measures for a first-order Markov chain.

Faithful port of the measures in Abdallah & Plumbley, *Information Dynamics:
Patterns of expectation and surprise in the perception of music* (Connection
Science 2009) — see ``papers/Information Dynamics ...pdf``, eqs (5), (7), (8).

Convention (matches the paper): the transition matrix ``a`` is COLUMN-stochastic,

    a[i, j] = p(S_{t+1} = i | S_t = j),      sum_i a[i, j] = 1.

So column ``j`` is the outgoing (next-state) distribution from state ``j``.

Measures
--------
entropy_rate  Ḣ(a)  = Σ_i π_i · H(column_i)              (eq 5)
    the observer's average per-step uncertainty about the next symbol.

pir           Ṫ     = Ḣ(a²) − Ḣ(a)  = I(S_t, S_{t+1} | S_{t-1})   (eq 8)
    predictive information rate: how much, on average, the present tells you
    about the future *beyond* what the past already told you. This is the
    quantity the paper conjectures peaks (inverted-U / Wundt curve) for
    aesthetically "just right" — structured *and* surprising — sequences.

redundancy    ρ     = H(marginal) − Ḣ(a)                 (Attneave; §2.6)
    how predictable the chain is from its own past (0 = iid, high = repetitive).

Surprise under an external ("subjective") prior model is in ``surprise_under``.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def stationary(a: np.ndarray) -> np.ndarray:
    """Stationary distribution π of a column-stochastic matrix ``a`` (aπ = π).

    Returns the normalised left Perron vector. Falls back to the uniform
    distribution if the chain is degenerate.
    """
    n = a.shape[0]
    vals, vecs = np.linalg.eig(a)
    # eigenvector for eigenvalue closest to 1
    idx = int(np.argmin(np.abs(vals - 1.0)))
    pi = np.real(vecs[:, idx])
    pi = np.abs(pi)
    s = pi.sum()
    if s < EPS or not np.isfinite(s):
        return np.full(n, 1.0 / n)
    return pi / s


def _col_entropy(a: np.ndarray) -> np.ndarray:
    """Per-column (per current-state) entropy in nats. Returns (n,)."""
    a_c = np.clip(a, EPS, None)
    # -Σ_i a[i,j] log a[i,j] over rows i, for each column j
    return -np.sum(a * np.log(a_c), axis=0)


def entropy_rate(a: np.ndarray, pi: np.ndarray | None = None) -> float:
    """Ḣ(a) = Σ_j π_j · H(column_j)  (eq 5), in nats/step."""
    if pi is None:
        pi = stationary(a)
    return float(np.dot(pi, _col_entropy(a)))


def pir(a: np.ndarray, pi: np.ndarray | None = None) -> float:
    """Predictive information rate Ṫ = Ḣ(a²) − Ḣ(a)  (eq 8), in nats/step.

    a² is the two-step transition matrix; it shares a's stationary π
    (a²π = a(aπ) = aπ = π), so we reuse π for both entropy rates.
    """
    if pi is None:
        pi = stationary(a)
    a2 = a @ a
    return float(entropy_rate(a2, pi) - entropy_rate(a, pi))


def redundancy(a: np.ndarray, pi: np.ndarray | None = None) -> float:
    """ρ = H(π) − Ḣ(a): reduction in per-symbol entropy from knowing the past."""
    if pi is None:
        pi = stationary(a)
    pic = np.clip(pi, EPS, None)
    h_marg = float(-np.sum(pi * np.log(pic)))
    return h_marg - entropy_rate(a, pi)


def fit_transition(seq: np.ndarray, n_states: int, alpha: float = 1.0) -> np.ndarray:
    """Column-stochastic transition matrix from a token sequence, Laplace-smoothed.

    ``alpha`` is the Dirichlet pseudo-count (1.0 = add-one). Smoothing keeps the
    chain ergodic so the stationary distribution and entropy rates are defined
    even for short per-stimulus sequences.
    """
    counts = np.full((n_states, n_states), alpha, dtype=np.float64)
    if len(seq) >= 2:
        cur = seq[:-1]
        nxt = seq[1:]
        # counts[next, cur] += 1  → column = current state
        np.add.at(counts, (nxt, cur), 1.0)
    counts /= counts.sum(axis=0, keepdims=True)
    return counts


def surprise_under(seq: np.ndarray, a_prior: np.ndarray) -> dict[str, float]:
    """Info-dynamics of a token sequence UNDER an external prior model.

    This is the *subjective* reading: an observer whose expectations are
    ``a_prior`` (e.g. a BB-native listener) hears ``seq`` and we measure how
    surprising / informative it is *to them*.

        mean_surprise : mean_t −log p_prior(s_t | s_{t-1})   (eq 1, L(x|z))
        max_surprise  : peak of that trajectory
        var_surprise  : dynamic range of surprise over time
    """
    if len(seq) < 2:
        return {"mean_surprise": 0.0, "max_surprise": 0.0, "var_surprise": 0.0}
    cur = seq[:-1]
    nxt = seq[1:]
    p = np.clip(a_prior[nxt, cur], EPS, None)  # p(next | cur) under prior
    s = -np.log(p)  # per-transition surprise, nats
    return {
        "mean_surprise": float(np.mean(s)),
        "max_surprise": float(np.max(s)),
        "var_surprise": float(np.var(s)),
    }


# ---------------------------------------------------------------------------
# Self-test: verify the closed forms against cases with known answers.
# ---------------------------------------------------------------------------


def _selftest() -> None:
    rng = np.random.default_rng(0)
    n = 5

    # 1. Deterministic permutation chain: no uncertainty, no predictive info.
    perm = np.roll(np.eye(n), 1, axis=0)  # column-stochastic permutation
    assert abs(entropy_rate(perm)) < 1e-9, "det chain entropy rate must be 0"
    assert abs(pir(perm)) < 1e-9, "det chain PIR must be 0"

    # 2. Uniform iid: max entropy, ZERO predictive info (future ⟂ present | past).
    uni = np.full((n, n), 1.0 / n)
    assert abs(entropy_rate(uni) - np.log(n)) < 1e-9, "iid entropy rate = log N"
    assert abs(pir(uni)) < 1e-9, "iid PIR must be 0"
    assert abs(redundancy(uni)) < 1e-9, "iid redundancy must be 0"

    # 3. Intermediate structured chain: PIR strictly positive, entropy in (0,logN).
    a = rng.dirichlet(np.ones(n) * 0.5, size=n).T  # each column a Dirichlet draw
    a /= a.sum(axis=0, keepdims=True)
    h = entropy_rate(a)
    p = pir(a)
    assert 0.0 < h < np.log(n), f"intermediate entropy {h} out of range"
    assert p > 1e-6, f"intermediate PIR should be > 0, got {p}"

    # 4. Stationarity: aπ = π.
    pi = stationary(a)
    assert np.allclose(a @ pi, pi, atol=1e-8), "stationary vector check"

    # 5. fit_transition recovers structure: a near-deterministic sequence has
    #    low entropy rate and low PIR (predictable/boring end of the arch).
    seq = np.array([0, 1, 2, 3, 4] * 40)  # perfectly cyclic
    a_fit = fit_transition(seq, n, alpha=0.01)
    assert entropy_rate(a_fit) < 0.2, "cyclic seq should have low entropy rate"

    # 6. surprise_under: a sequence drawn FROM a_prior is less surprising than
    #    an adversarial one that always makes the low-probability transition.
    a_prior = fit_transition(seq, n, alpha=0.01)
    faithful = surprise_under(seq, a_prior)
    adversarial = surprise_under(np.array([0, 2, 4, 1, 3] * 40), a_prior)
    assert adversarial["mean_surprise"] > faithful["mean_surprise"], (
        "off-model sequence must be more surprising under the prior"
    )

    print("markov_infodyn self-test: ALL PASS")
    print(f"  intermediate chain: entropy_rate={h:.3f} nats  PIR={p:.4f} nats")


if __name__ == "__main__":
    _selftest()
