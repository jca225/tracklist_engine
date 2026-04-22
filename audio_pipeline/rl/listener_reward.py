"""Scalar listener reward = α·age_sensitivity + β·genre_affinity + γ·playlist_prior.

Pure functions of `(UserContext, TrackFeatures)`. No I/O. Deterministic
given the context. Use this as the reward head during SAC training of
the DJ policy (with α/β/γ tuned on Spotify MPD retention signals in the
PIR-validation todo).
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .context import TrackFeatures, UserContext


CURRENT_YEAR: int = 2026


def age_sensitivity(c: UserContext, track_release_year: int, now_year: int = CURRENT_YEAR) -> float:
    """Han Bigaussian evaluated at the listener's age when the track came out.

    Negative `age_when_released` (track released before listener's birth)
    is handled naturally by the Bigaussian's left tail — older tracks a
    listener couldn't have heard in real time still get positive mass via
    cultural inheritance / canon listening.
    """
    age_when_released = c.age - (now_year - track_release_year)
    delta = age_when_released - c.peak_sensitivity_age
    w = c.pre_peak_width if delta < 0.0 else c.post_peak_width
    return c.sensitivity_baseline + c.peak_height * math.exp(-0.5 * (delta / w) ** 2)


def genre_affinity(c: UserContext, tags: dict[str, float] | Iterable[tuple[str, float]]) -> float:
    """Weighted dot-product of the track's (tag → strength) dict against the
    listener's `genre_weights`. Unknown tags contribute 0 (no penalty)."""
    items = tags.items() if isinstance(tags, dict) else tags
    return float(sum(c.genre_weights.get(tag, 0.0) * strength for tag, strength in items))


def playlist_prior_score(c: UserContext, track_mert_mean: np.ndarray | None) -> float:
    """Cosine similarity between the listener's MPD-derived playlist prior
    and the track's mean-pooled MERT embedding. Returns 0 in Phase 1 when
    either side is missing."""
    if c.playlist_prior is None or track_mert_mean is None:
        return 0.0
    a, b = c.playlist_prior, track_mert_mean
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0


def listener_retention(c: UserContext, track: TrackFeatures) -> float:
    """Scalar reward — larger = higher predicted retention from this listener
    on this track. Linear combination; all three components are pure."""
    s = age_sensitivity(c, track.release_year)
    g = genre_affinity(c, dict(track.tags))
    p = playlist_prior_score(c, track.mert_mean)
    return c.alpha_sensitivity * s + c.beta_genre * g + c.gamma_prior * p
