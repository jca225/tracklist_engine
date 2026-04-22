"""Frozen records for listener reward conditioning.

`UserContext` parameterises the listener reward function. `TrackFeatures`
is the minimum slice of a track the reward needs to score.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np


# Han et al. 2022, §3.2 Bigaussian fit for all NCM users.
# Peak age xc = 12.88; replicated on US Spotify (Stephens-Davidowitz 2018).
HAN_ALL_USERS: dict[str, float] = {
    "peak_sensitivity_age": 12.88,
    "pre_peak_width": 13.18,       # w1 (ages 0..peak)
    "post_peak_width": 7.26,       # w2 (peak..upper)
    "peak_height": 0.87,           # H
    "sensitivity_baseline": 0.43,  # y0
}

HAN_MALE: dict[str, float] = {
    "peak_sensitivity_age": 12.76, "pre_peak_width": 14.25,
    "post_peak_width": 7.19,       "peak_height": 0.85,
    "sensitivity_baseline": 0.41,
}

HAN_FEMALE: dict[str, float] = {
    "peak_sensitivity_age": 13.79, "pre_peak_width": 12.83,
    "post_peak_width": 7.60,       "peak_height": 0.83,
    "sensitivity_baseline": 0.50,
}


@dataclass(frozen=True)
class UserContext:
    """Everything the listener reward is conditioned on.

    Bigaussian defaults come from Han "all users" fit — the universal
    claim in the paper. Gender-specific variants are available via
    `priors.han_male` / `priors.han_female`.
    """
    age: int
    peak_sensitivity_age: float = HAN_ALL_USERS["peak_sensitivity_age"]
    pre_peak_width: float       = HAN_ALL_USERS["pre_peak_width"]
    post_peak_width: float      = HAN_ALL_USERS["post_peak_width"]
    peak_height: float          = HAN_ALL_USERS["peak_height"]
    sensitivity_baseline: float = HAN_ALL_USERS["sensitivity_baseline"]

    # Demographic-specific genre affinities. Default is empty → no genre bias.
    # Populated by constructors in `priors.py` (e.g. wealthy_ne_american).
    genre_weights: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({})
    )

    # Phase 2 slot: a 128-d playlist-prior embedding averaged over a slice of
    # the Spotify Million Playlist Dataset (MPD) matching this demographic.
    # `None` in Phase 1 — the playlist-prior component of the reward is then
    # disabled (contributes 0), leaving age-sensitivity + genre-affinity.
    playlist_prior: np.ndarray | None = None

    # Retention-head combination weights (α·sensitivity + β·genre + γ·prior).
    # Tuned empirically during PIR-reward validation against MPD.
    alpha_sensitivity: float = 1.0
    beta_genre: float        = 0.5
    gamma_prior: float       = 0.5


@dataclass(frozen=True)
class TrackFeatures:
    """Minimum track info the listener reward needs to score a candidate."""
    track_id: str
    release_year: int
    tags: Mapping[str, float]                # genre/emotion/theme → normalized strength
    mert_mean: np.ndarray | None = None      # (768,) mean across MERT sections; Phase 2
    bpm: float | None = None
    duration_s: float | None = None
