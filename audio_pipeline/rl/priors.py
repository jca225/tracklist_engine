"""Demographic-specific `UserContext` constructors.

Each constructor returns a `UserContext` with:
  - Han et al. universal Bigaussian params (or a gender-specific variant)
  - Genre-weight dict calibrated to that demographic's literature findings

Adding a new demographic = one function. Keep numbers here, not scattered.
"""
from __future__ import annotations

from types import MappingProxyType

import numpy as np

from .context import HAN_ALL_USERS, HAN_FEMALE, HAN_MALE, UserContext


# -----------------------------------------------------------------------
# Mellander et al. 2018, Table 2: US-metro income × genre correlations.
# Positive coefficient = genre over-represented in high-income metros.
# Numbers are *signed scalars*, not probabilities — use as additive bias.
# Han confirms the US-sophisticated-income link; US-specific rap/soul/reggae
# link comes from Mellander and is the bit that does NOT transfer to China.
# These values are the ones to refine when we get the paper's raw table
# (currently approximate — flagged for PIR-validation-time retuning).
# -----------------------------------------------------------------------
_MELLANDER_US_INCOME_GENRE: dict[str, float] = {
    # Sophisticated (income +)
    "jazz":      0.35, "classical":   0.30, "world_music":  0.28,
    "blues":     0.22, "folk":        0.18,
    # Contemporary (US-specific income +)
    "rap":       0.22, "soul":        0.20, "reggae":       0.18,
    "r&b":       0.18, "hip_hop":     0.15,
    # Mainstream baseline
    "pop":       0.00, "rock":        0.05,
    # Income-negative in US metros
    "country":  -0.15, "christian":  -0.10,
    "metal":    -0.05, "electronica": 0.05,
}


def _with_jitter(d: dict[str, float], rng: np.random.Generator, sd: float = 0.05) -> dict[str, float]:
    return {k: v + float(rng.normal(0.0, sd)) for k, v in d.items()}


def han_universal(age: int) -> UserContext:
    """No genre bias, just the universal age-sensitivity curve. Useful as a
    control for ablating the demographic component during PIR validation."""
    return UserContext(age=age, **HAN_ALL_USERS, genre_weights=MappingProxyType({}))


def han_male(age: int) -> UserContext:
    return UserContext(age=age, **HAN_MALE, genre_weights=MappingProxyType({}))


def han_female(age: int) -> UserContext:
    return UserContext(age=age, **HAN_FEMALE, genre_weights=MappingProxyType({}))


def wealthy_ne_american(
    age: int | None = None,
    *,
    rng: np.random.Generator | None = None,
    jitter_sd: float = 0.05,
) -> UserContext:
    """Sample a listener context calibrated to wealthy US Northeast adults.

    - Age: drawn from Bonneville-Roussy 2013's US/UK adult streaming cohort,
      biased to 22-45 (peak ~28) when `age` is None.
    - Genre weights: Mellander 2018 US-metro income coefficients, with small
      Gaussian jitter so repeated calls sample individual variation within
      the demographic rather than returning the same vector.
    """
    rng = rng or np.random.default_rng()

    if age is None:
        ages = np.arange(22, 46)
        # Rough lognormal-ish bias toward late 20s / early 30s
        pmf = np.exp(-0.5 * ((ages - 28) / 6.0) ** 2)
        pmf = pmf / pmf.sum()
        age = int(rng.choice(ages, p=pmf))

    genre = _with_jitter(_MELLANDER_US_INCOME_GENRE, rng, sd=jitter_sd)
    return UserContext(
        age=age,
        **HAN_ALL_USERS,          # universal curve; US-replicated by Stephens-Davidowitz
        genre_weights=MappingProxyType(genre),
    )
