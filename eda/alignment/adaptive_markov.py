"""Adaptive first-order Markov chain — information-dynamics readouts.

Implements the model-based observer of Abdallah & Plumbley (Connection Science
2008) over a discrete symbol alphabet. Pure numpy; no torch. Every readout at
frame t is causal — computed from the belief state *before* the frame-t count
update.

Measures (paper reference in brackets):

- ``surprisingness`` — L(x|z) = −log p(x|z)                          [eq. 1]
- ``predictive_uncertainty`` — H(X|Z=z), entropy of the prediction   [§2.1]
- ``model_information_rate`` — Bayesian surprise, KL(posterior ‖ prior)
  over the context column's Dirichlet belief                         [§2.3]
- ``predictive_information`` — exact I(x|z) = Σₖ a_kx log(a_kx/[a²]_kz),
  the KL between the next-step forecast after vs before seeing x     [eq. A11]
- ``expected_predictive_information`` — E_{x~p(·|z)} I(x|z), the paper's
  pre-event attention signal Ī(z)                                    [eq. 8]
- ``predictive_information_rate`` — Ḣ(a²) − Ḣ(a) of the current expected
  transition matrix; a slow-varying model property                   [eq. 8]
- ``uncertainty_delta`` — H(X|z_{t−1}) − H(X|z_t). NOT a paper measure;
  retained as the legacy "pir_proxy" signal for comparability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _digamma(x: np.ndarray) -> np.ndarray:
    """Digamma ψ(x). Uses scipy when available; Stirling fallback otherwise."""
    try:
        from scipy.special import digamma as sp_digamma  # type: ignore[import-untyped]

        return sp_digamma(x).astype(np.float64)
    except ImportError:
        x = np.maximum(x, 1e-8)
        return np.log(x) - 1.0 / (2.0 * x)


def _log_beta(alpha: np.ndarray) -> float:
    return sum(math.lgamma(float(a)) for a in alpha) - math.lgamma(float(np.sum(alpha)))


def dirichlet_kl(alpha_q: np.ndarray, alpha_p: np.ndarray) -> float:
    """KL(Dir(q) || Dir(p)) for parameter vectors of equal length."""
    alpha_q = np.asarray(alpha_q, dtype=np.float64)
    alpha_p = np.asarray(alpha_p, dtype=np.float64)
    s_q = float(np.sum(alpha_q))
    s_p = float(np.sum(alpha_p))
    psi_q = _digamma(alpha_q)
    psi_sq = float(_digamma(np.array([s_q]))[0])
    return (
        _log_beta(alpha_p)
        - _log_beta(alpha_q)
        + np.sum((alpha_q - alpha_p) * (psi_q - psi_sq))
    )


@dataclass(frozen=True)
class MarkovReadout:
    symbol: int
    prev_symbol: int | None
    surprisingness: float
    predictive_uncertainty: float
    model_information_rate: float
    predictive_information: float
    expected_predictive_information: float
    predictive_information_rate: float
    uncertainty_delta: float


@dataclass(frozen=True)
class MarkovTrace:
    symbols: tuple[int, ...]
    readouts: tuple[MarkovReadout, ...]

    def series(self, name: str) -> np.ndarray:
        key = {
            "surprisingness": lambda r: r.surprisingness,
            "predictive_uncertainty": lambda r: r.predictive_uncertainty,
            "model_information_rate": lambda r: r.model_information_rate,
            "predictive_information": lambda r: r.predictive_information,
            "expected_predictive_information": lambda r: (
                r.expected_predictive_information
            ),
            "predictive_information_rate": lambda r: r.predictive_information_rate,
            "uncertainty_delta": lambda r: r.uncertainty_delta,
            "mir": lambda r: r.model_information_rate,
            "pred_info": lambda r: r.predictive_information,
            "expected_pi": lambda r: r.expected_predictive_information,
            "pir": lambda r: r.predictive_information_rate,
        }.get(name)
        if key is None:
            raise KeyError(name)
        return np.array([key(r) for r in self.readouts], dtype=np.float64)


class AdaptiveMarkovChain:
    """Online adaptive Markov observer over a discrete symbol alphabet."""

    def __init__(
        self,
        n_symbols: int,
        *,
        alpha: float = 0.5,
        beta: float = 0.01,
    ) -> None:
        if n_symbols < 2:
            raise ValueError("n_symbols must be >= 2")
        self.n_symbols = n_symbols
        self.alpha = alpha
        self.beta = beta
        self.theta = np.full((n_symbols, n_symbols), alpha, dtype=np.float64)
        self._prev: int | None = None
        self._prev_uncertainty: float | None = None
        self._pi: np.ndarray = np.full(n_symbols, 1.0 / n_symbols, dtype=np.float64)

    def _column_probs(self, j: int) -> np.ndarray:
        col = self.theta[:, j]
        total = col.sum()
        if total <= 0:
            return np.full(self.n_symbols, 1.0 / self.n_symbols)
        return col / total

    def _transition_matrix(self) -> np.ndarray:
        """Expected transition matrix a_ij = p(next=i | prev=j) — column-normalized θ."""
        totals = self.theta.sum(axis=0, keepdims=True)
        return self.theta / np.maximum(totals, 1e-12)

    def _predictive_information_all(self, j: int, a: np.ndarray) -> np.ndarray:
        """Eq. A11 for every candidate symbol i: I(i|j) = Σₖ a_ki log(a_ki / [a²]_kj).

        Numerator column belongs to the *observed* symbol i; the denominator is
        column j of the matrix square of the normalized transition matrix (NOT
        of the raw counts — squaring counts skews paths by per-context data
        volume).
        """
        log_a = np.log(np.maximum(a, 1e-12))
        neg_h = np.sum(a * log_a, axis=0)  # Σₖ a_ki log a_ki, per column i
        a2_col = a @ a[:, j]  # [a²]_{:,j}
        log_a2 = np.log(np.maximum(a2_col, 1e-12))
        return np.maximum(neg_h - a.T @ log_a2, 0.0)

    def _stationary(self, a: np.ndarray) -> np.ndarray:
        """Equilibrium distribution aπ = π by warm-started power iteration."""
        pi = self._pi
        for _ in range(200):
            nxt = a @ pi
            total = nxt.sum()
            if total <= 0:
                return np.full(self.n_symbols, 1.0 / self.n_symbols)
            nxt = nxt / total
            if float(np.max(np.abs(nxt - pi))) < 1e-10:
                pi = nxt
                break
            pi = nxt
        self._pi = pi
        return pi

    @staticmethod
    def _entropy_rate(a: np.ndarray, pi: np.ndarray) -> float:
        """Ḣ(a) = Σⱼ πⱼ H(a_:j) — paper eq. 5."""
        log_a = np.log(np.maximum(a, 1e-12))
        col_h = -np.sum(a * log_a, axis=0)
        return float(np.dot(col_h, pi))

    def _predictive_information_rate(self, a: np.ndarray) -> float:
        """Exact model PIR Ḣ(a²) − Ḣ(a) — paper eq. 8; ≥ 0, same π for both."""
        pi = self._stationary(a)
        return max(self._entropy_rate(a @ a, pi) - self._entropy_rate(a, pi), 0.0)

    def observe(self, symbol: int) -> MarkovReadout:
        if not 0 <= symbol < self.n_symbols:
            raise IndexError(f"symbol {symbol} out of range [0, {self.n_symbols})")

        prev = self._prev
        surprisingness = 0.0
        uncertainty = 0.0
        mir = 0.0
        pred_info = 0.0
        expected_pi = 0.0
        pir = 0.0
        uncertainty_delta = 0.0

        if prev is not None:
            a = self._transition_matrix()
            probs = a[:, prev]
            p = max(float(probs[symbol]), 1e-12)
            surprisingness = -math.log(p)
            uncertainty = float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))))

            pi_all = self._predictive_information_all(prev, a)
            pred_info = float(pi_all[symbol])
            expected_pi = float(np.dot(probs, pi_all))
            pir = self._predictive_information_rate(a)

            prior_col = self.theta[:, prev].copy()
            post_col = prior_col.copy()
            post_col[symbol] += 1.0
            mir = dirichlet_kl(post_col, prior_col)

            if self._prev_uncertainty is not None:
                uncertainty_delta = self._prev_uncertainty - uncertainty

            self.theta[symbol, prev] += 1.0
            if self.beta > 0:
                self.theta /= 1.0 + self.beta * self.theta

        self._prev = symbol
        self._prev_uncertainty = uncertainty if prev is not None else None

        return MarkovReadout(
            symbol=symbol,
            prev_symbol=prev,
            surprisingness=surprisingness,
            predictive_uncertainty=uncertainty,
            model_information_rate=mir,
            predictive_information=pred_info,
            expected_predictive_information=expected_pi,
            predictive_information_rate=pir,
            uncertainty_delta=uncertainty_delta,
        )

    def run(self, symbols: np.ndarray | list[int]) -> MarkovTrace:
        syms = tuple(int(s) for s in symbols)
        readouts = tuple(self.observe(s) for s in syms)
        return MarkovTrace(symbols=syms, readouts=readouts)
