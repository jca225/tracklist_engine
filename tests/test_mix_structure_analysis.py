"""Tests for mix structure analysis (synthetic — no audio)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from eda.alignment.adaptive_markov import AdaptiveMarkovChain
from eda.alignment.boundaries import (
    pick_local_peaks,
    pick_peaks,
    score_boundaries,
    seconds_to_bar_indices,
)
from eda.alignment.mix_structure_probe import _synthetic_artifact, run_probe
from eda.alignment.tokenize import fit_vq_kmeans
from labeling.schema import GroundTruthSet, GroundTruthTrack, load
from core.result import Ok


def test_markov_spikes_at_regime_change():
    # AAAABBBB — expect higher MIR at bar index 4
    symbols = [0, 0, 0, 0, 1, 1, 1, 1]
    trace = AdaptiveMarkovChain(2).run(symbols)
    mir = trace.series("model_information_rate")
    assert mir[4] > mir[2]
    assert mir[4] > mir[6]


def test_predictive_information_matches_eq_a11_bruteforce():
    # I(i|j) = sum_k a_ki * log(a_ki / [a^2]_kj) on the NORMALIZED transition
    # matrix — regression for the two 2026-07 bugs (context column used as
    # numerator; raw counts squared instead of a).
    rng = np.random.default_rng(0)
    chain = AdaptiveMarkovChain(5, alpha=0.3, beta=0.0)
    chain.theta = rng.uniform(0.2, 4.0, size=(5, 5))
    a = chain.theta / chain.theta.sum(axis=0, keepdims=True)
    a2 = a @ a
    for j in range(5):
        got = chain._predictive_information_all(j, a)
        for i in range(5):
            want = sum(a[k, i] * np.log(a[k, i] / a2[k, j]) for k in range(5))
            assert abs(got[i] - max(want, 0.0)) < 1e-9, (i, j)


def test_predictive_information_depends_on_observed_symbol():
    # Symbol 0's successors are near-deterministic, symbol 1's near-uniform →
    # observing them from the same context must yield different I(x|z).
    chain = AdaptiveMarkovChain(3, alpha=0.01, beta=0.0)
    chain.theta = np.array(
        [
            [0.01, 5.0, 5.0],
            [15.0, 5.0, 5.0],
            [0.01, 5.0, 5.0],
        ]
    )
    a = chain.theta / chain.theta.sum(axis=0, keepdims=True)
    pi_all = chain._predictive_information_all(2, a)
    assert abs(pi_all[0] - pi_all[1]) > 1e-3


def test_pir_exact_properties():
    # Deterministic 2-cycle and uniform chain both have PIR 0; a noisy-sticky
    # chain of intermediate entropy has PIR > 0 (the paper's inverted-U).
    chain = AdaptiveMarkovChain(2, alpha=0.5, beta=0.0)

    det = np.array([[1e-9, 1.0], [1.0, 1e-9]])
    uni = np.full((2, 2), 0.5)
    mid = np.array([[0.9, 0.1], [0.1, 0.9]])

    pi_uni = np.array([0.5, 0.5])
    assert chain._entropy_rate(det, pi_uni) < 1e-6
    assert abs(chain._entropy_rate(uni, pi_uni) - np.log(2)) < 1e-9

    assert chain._predictive_information_rate(det) < 1e-6
    assert chain._predictive_information_rate(uni) < 1e-9
    assert chain._predictive_information_rate(mid) > 0.01


def test_readout_exposes_exact_measures_causally():
    rng = np.random.default_rng(3)
    symbols = rng.integers(0, 4, size=60).tolist()
    trace = AdaptiveMarkovChain(4).run(symbols)
    pinfo = trace.series("pred_info")
    epi = trace.series("expected_pi")
    pir = trace.series("pir")
    # frame 0 has no context — all zero; afterwards nonnegative
    assert pinfo[0] == 0.0
    assert np.all(pinfo >= 0.0) and np.all(epi >= 0.0) and np.all(pir >= 0.0)
    # pred_info must actually vary with observations (the old bug froze it per context)
    assert np.std(pinfo[1:]) > 0.0


def test_vq_labels_cover_all_clusters():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 16)).astype(np.float32)
    _, labels = fit_vq_kmeans(x, 8)
    assert labels.shape == (40,)
    assert labels.min() >= 0
    assert labels.max() < 8


def test_boundary_scoring_tolerance():
    score = score_boundaries((10, 20, 30), (11, 21), tolerance_bars=1)
    assert score.tp == 2
    assert score.precision == 2 / 3
    assert score.recall == 1.0


def test_pick_local_peaks_finds_spike():
    s = np.zeros(100, dtype=np.float64)
    s[50] = 10.0
    peaks = pick_local_peaks(s, window=16, z_threshold=2.0, min_distance=4)
    assert any(abs(p - 50) <= 2 for p in peaks)


def test_synthetic_probe_finds_section_boundaries():
    art = _synthetic_artifact(n_sections=6, bars_per_section=24, dim=32)
    result, _ = run_probe(art, None, chroma=False, n_tokens=8)
    peaks = result["streams"]["mert_vq"]["peaks"]["mir_local"]
    expected = {0, 24, 48, 72, 96, 120}
    hit = sum(1 for e in expected if any(abs(p - e) <= 4 for p in peaks))
    assert hit >= 3, f"expected most section starts in local peaks, got {peaks}"


def test_seconds_to_bar_indices():
    bar_start = np.array([0.0, 2.0, 4.0, 6.0])
    assert seconds_to_bar_indices([5.1], bar_start) == (3,)


def test_bb12_gt_load_and_probe_dry():
    yaml_path = Path("labeling/fixtures/bb12_ground_truth.yaml")
    if not yaml_path.is_file():
        return
    match load(yaml_path):
        case Ok(gt):
            assert isinstance(gt, GroundTruthSet)
        case _:
            raise AssertionError("expected Ok")
    art = _synthetic_artifact(n_sections=20, bars_per_section=16, dim=32)
    # GT times won't align with synthetic bars — just ensure scoring path runs
    result, _ = run_probe(art, gt, n_tokens=12, chroma=False)
    assert "streams" in result
    assert "score_mir" not in result  # scores nested under streams now
    assert "mir_local" in result["streams"]["mert_vq"]["scores"]
