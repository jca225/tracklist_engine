"""Adaptive first-order Markov chain — information-dynamics readouts.

Faithful enough to Abdallah & Plumbley (Connection Science 2008) for exploratory
boundary detection. Pure numpy; no torch.
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
        _log_beta(alpha_p) - _log_beta(alpha_q)
        + np.sum((alpha_q - alpha_p) * (psi_q - psi_sq))
    )


@dataclass(frozen=True)
class MarkovReadout:
    symbol: int
    prev_symbol: int | None
    surprisingness: float
    predictive_uncertainty: float
    model_information_rate: float
    predictive_information_rate: float


@dataclass(frozen=True)
class MarkovTrace:
    symbols: tuple[int, ...]
    readouts: tuple[MarkovReadout, ...]

    def series(self, name: str) -> np.ndarray:
        key = {
            "surprisingness": lambda r: r.surprisingness,
            "predictive_uncertainty": lambda r: r.predictive_uncertainty,
            "model_information_rate": lambda r: r.model_information_rate,
            "predictive_information_rate": lambda r: r.predictive_information_rate,
            "mir": lambda r: r.model_information_rate,
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

    def _column_probs(self, j: int) -> np.ndarray:
        col = self.theta[:, j]
        total = col.sum()
        if total <= 0:
            return np.full(self.n_symbols, 1.0 / self.n_symbols)
        return col / total

    def _predictive_information(self, j: int, i: int, probs: np.ndarray) -> float:
        """Eq. A11 style term — info symbol i carries about the future given context j."""
        a2 = self.theta @ self.theta
        denom_col = a2[:, j]
        total = denom_col.sum()
        if total <= 0:
            return 0.0
        a2_norm = denom_col / total
        out = 0.0
        for k in range(self.n_symbols):
            akj = probs[k]
            if akj <= 1e-12:
                continue
            denom = a2_norm[k]
            if denom <= 1e-12:
                continue
            out += akj * math.log(akj / denom)
        return out

    def observe(self, symbol: int) -> MarkovReadout:
        if not 0 <= symbol < self.n_symbols:
            raise IndexError(f"symbol {symbol} out of range [0, {self.n_symbols})")

        prev = self._prev
        surprisingness = 0.0
        uncertainty = 0.0
        mir = 0.0
        pir = 0.0

        if prev is not None:
            probs = self._column_probs(prev)
            p = max(float(probs[symbol]), 1e-12)
            surprisingness = -math.log(p)
            uncertainty = float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))))

            prior_col = self.theta[:, prev].copy()
            post_col = prior_col.copy()
            post_col[symbol] += 1.0
            mir = dirichlet_kl(post_col, prior_col)

            if self._prev_uncertainty is not None:
                pir = self._prev_uncertainty - uncertainty

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
            predictive_information_rate=pir,
        )

    def run(self, symbols: np.ndarray | list[int]) -> MarkovTrace:
        syms = tuple(int(s) for s in symbols)
        readouts = tuple(self.observe(s) for s in syms)
        return MarkovTrace(symbols=syms, readouts=readouts)
