"""Tests for Phase 1 listener reward model."""
from __future__ import annotations

import math

import numpy as np
import pytest

from audio_pipeline.rl import priors
from audio_pipeline.rl.context import HAN_ALL_USERS, TrackFeatures, UserContext
from audio_pipeline.rl.listener_reward import (
    age_sensitivity,
    genre_affinity,
    listener_retention,
    playlist_prior_score,
)


# ---- Bigaussian age-sensitivity --------------------------------------------

def test_bigaussian_peaks_at_age_13_for_listener_who_was_13_at_release() -> None:
    # Listener age 30 in 2026 → was 13 in 2009. A track from 2009 should peak.
    c = priors.han_universal(age=30)
    peak = age_sensitivity(c, track_release_year=2009, now_year=2026)
    off_by_five = age_sensitivity(c, track_release_year=2004, now_year=2026)
    assert peak > off_by_five


def test_bigaussian_asymmetry_matches_han_paper() -> None:
    """Han's w1=13.18, w2=7.26 → pre-peak tail is wider than post-peak tail,
    i.e. sensitivity falls off faster for tracks released after age-of-peak."""
    c = priors.han_universal(age=25)
    # For a 25yo in 2026, age-of-peak (13) was in 2014.
    five_before_peak = age_sensitivity(c, 2009)   # listener age 8 at release
    five_after_peak  = age_sensitivity(c, 2019)   # listener age 18 at release
    assert five_before_peak > five_after_peak


def test_bigaussian_peak_value_matches_paper_constants() -> None:
    c = priors.han_universal(age=HAN_ALL_USERS["peak_sensitivity_age"].__ceil__())
    # Evaluate exactly at the peak year
    peak = age_sensitivity(c, track_release_year=2026, now_year=2026)
    # Should equal baseline + H at the peak, minus tiny fractional-age offset
    expected_upper = HAN_ALL_USERS["sensitivity_baseline"] + HAN_ALL_USERS["peak_height"]
    assert peak <= expected_upper + 1e-9
    assert peak > HAN_ALL_USERS["sensitivity_baseline"] + 0.5


# ---- Genre affinity --------------------------------------------------------

def test_genre_affinity_is_weighted_dot_product() -> None:
    c = UserContext(age=30, genre_weights={"jazz": 0.35, "pop": 0.0, "country": -0.15})
    # Track that's 70% jazz, 30% country
    score = genre_affinity(c, {"jazz": 0.7, "country": 0.3})
    assert score == pytest.approx(0.35 * 0.7 + -0.15 * 0.3)


def test_genre_affinity_ignores_unknown_tags() -> None:
    c = UserContext(age=30, genre_weights={"jazz": 1.0})
    assert genre_affinity(c, {"dubstep": 0.9, "unknown_tag": 0.5}) == 0.0


def test_genre_affinity_empty_weights_returns_zero() -> None:
    c = priors.han_universal(age=30)
    assert genre_affinity(c, {"pop": 1.0, "jazz": 0.5}) == 0.0


# ---- Playlist prior --------------------------------------------------------

def test_playlist_prior_is_zero_when_prior_missing() -> None:
    c = priors.han_universal(age=30)
    assert playlist_prior_score(c, np.ones(128)) == 0.0


def test_playlist_prior_cosine_between_two_unit_vectors() -> None:
    v = np.zeros(128); v[0] = 1.0
    c = UserContext(age=30, playlist_prior=v)
    assert playlist_prior_score(c, v) == pytest.approx(1.0)
    opposite = -v
    assert playlist_prior_score(c, opposite) == pytest.approx(-1.0)


# ---- Demographic priors ----------------------------------------------------

def test_wealthy_ne_american_returns_stable_structure() -> None:
    rng = np.random.default_rng(42)
    c = priors.wealthy_ne_american(age=30, rng=rng, jitter_sd=0.0)
    # Mellander: jazz, classical, rap all income-positive in US metros
    assert c.genre_weights["jazz"] > 0
    assert c.genre_weights["classical"] > 0
    assert c.genre_weights["rap"] > 0
    # Country income-negative in US
    assert c.genre_weights["country"] < 0


def test_wealthy_ne_american_age_distribution_sits_in_adult_range() -> None:
    rng = np.random.default_rng(0)
    ages = [priors.wealthy_ne_american(rng=rng).age for _ in range(200)]
    assert min(ages) >= 22 and max(ages) <= 45
    assert 25 <= sum(ages) / len(ages) <= 32   # mean near 28


def test_wealthy_ne_american_jitter_varies_across_draws() -> None:
    rng = np.random.default_rng(1)
    a = priors.wealthy_ne_american(age=30, rng=rng, jitter_sd=0.1).genre_weights["jazz"]
    b = priors.wealthy_ne_american(age=30, rng=rng, jitter_sd=0.1).genre_weights["jazz"]
    assert a != b


# ---- End-to-end retention scalar ------------------------------------------

def test_listener_retention_is_pure_linear_combination() -> None:
    c = UserContext(
        age=30,
        genre_weights={"jazz": 0.5},
        alpha_sensitivity=1.0, beta_genre=2.0, gamma_prior=0.0,
    )
    t = TrackFeatures(track_id="t", release_year=2015, tags={"jazz": 1.0})
    s = age_sensitivity(c, 2015)
    g = genre_affinity(c, {"jazz": 1.0})
    assert listener_retention(c, t) == pytest.approx(1.0 * s + 2.0 * g)


def test_listener_retention_higher_for_demographic_match_track() -> None:
    """A wealthy-NE-American listener should score a jazz track higher than
    a country track of identical release year, all else equal."""
    c = priors.wealthy_ne_american(age=30, rng=np.random.default_rng(0), jitter_sd=0.0)
    jazz = TrackFeatures("j", release_year=2015, tags={"jazz": 1.0})
    country = TrackFeatures("c", release_year=2015, tags={"country": 1.0})
    assert listener_retention(c, jazz) > listener_retention(c, country)
