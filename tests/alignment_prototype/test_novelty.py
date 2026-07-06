"""Tests for the Foote-novelty surprise action (agentic/novelty.py)."""

from __future__ import annotations

import numpy as np

from workspaces.alignment_prototype.agentic.novelty import (
    foote_novelty,
    surprise_runner,
)


def _two_regime_mert(n: int = 120, boundary: int = 60, dim: int = 16) -> np.ndarray:
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.05, size=(boundary, dim)) + np.eye(dim)[0]
    b = rng.normal(0.0, 0.05, size=(n - boundary, dim)) + np.eye(dim)[1]
    return np.vstack([a, b]).astype(np.float32)


def test_foote_novelty_peaks_at_regime_boundary():
    nov = foote_novelty(_two_regime_mert(), kernel_bars=6)
    assert int(np.argmax(nov)) in range(56, 65)


def test_foote_novelty_survives_nan_bar():
    x = _two_regime_mert()
    x[10] = np.nan
    nov = foote_novelty(x, kernel_bars=6)
    assert np.all(np.isfinite(nov))
    assert int(np.argmax(nov)) in range(56, 65)


def _curve(boundary_s: float = 120.0):
    # flat novelty with one clear peak at boundary_s (2 s bars)
    times = [2.0 * i for i in range(100)]
    vals = [1.0 + (0.1 if i % 7 == 0 else 0.0) for i in range(100)]
    vals[int(boundary_s / 2.0)] = 9.0
    return tuple(zip(times, vals))


def test_surprise_runner_snaps_cue_to_nearest_peak():
    run = surprise_runner(_curve(120.0), band_s=45.0)
    obs = run({"cue_anchor_s": 130.0, "slot_label": "12"})
    assert not obs.abstained
    assert obs.probe == "surprise"
    assert abs(obs.set_start_s - 120.0) < 2.1


def test_surprise_runner_abstains_without_independent_center():
    run = surprise_runner(_curve(), band_s=45.0)
    assert run({"cue_anchor_s": 0.0, "slot_label": "3w1"}).abstained  # fake w-row cue
    assert run({"cue_anchor_s": None, "slot_label": "3"}).abstained


def test_surprise_runner_abstains_when_no_peak_in_band():
    run = surprise_runner(_curve(120.0), band_s=10.0)
    assert run({"cue_anchor_s": 190.0, "slot_label": "5"}).abstained
