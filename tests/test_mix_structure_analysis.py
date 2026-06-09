"""Tests for mix structure analysis (synthetic — no audio)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from eda.alignment.adaptive_markov import AdaptiveMarkovChain
from eda.alignment.boundaries import pick_local_peaks, pick_peaks, score_boundaries, seconds_to_bar_indices
from eda.alignment.mix_structure_probe import _synthetic_artifact, run_probe
from eda.alignment.tokenize import fit_vq_kmeans
from labeling.ground_truth.schema import GroundTruthSet, GroundTruthTrack, load
from core.result import Ok


def test_markov_spikes_at_regime_change():
    # AAAABBBB — expect higher MIR at bar index 4
    symbols = [0, 0, 0, 0, 1, 1, 1, 1]
    trace = AdaptiveMarkovChain(2).run(symbols)
    mir = trace.series("model_information_rate")
    assert mir[4] > mir[2]
    assert mir[4] > mir[6]


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
