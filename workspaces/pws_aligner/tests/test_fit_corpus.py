from __future__ import annotations

import json
from pathlib import Path

from workspaces.pws_aligner.continuous_model import ProbeNoise
from workspaces.pws_aligner.fit_corpus import (
    ablation_compare,
    fit_cohort,
    load_cohort,
    pool_set_votes,
    probe_noise_dict,
)

_COHORT = Path(__file__).resolve().parents[1] / "cohorts" / "big_bootie.json"


def _fake_votes_file(path: Path, span_ids: list[str]) -> None:
    """Write a minimal probe_votes.json with one probe per span."""
    docs = []
    for sid in span_ids:
        docs.append(
            {
                "span_id": sid,
                "slot_label": sid,
                "recording_id": "r_" + sid,
                "probes": [
                    {
                        "probe": "fp",
                        "recording_id": "r_" + sid,
                        "offset_s": 10.0,
                        "confidence": 0.8,
                        "abstain": False,
                        "features": [],
                    }
                ],
            }
        )
    path.write_text(json.dumps(docs))


def test_load_cohort_big_bootie():
    c = load_cohort(_COHORT)
    assert c.name == "big_bootie"
    assert "2nvzlh2k" in c.set_ids and "1fsnxchk" in c.set_ids
    assert len(c.set_ids) == 27
    assert c.grade_sets == ("2nvzlh2k", "1fsnxchk")


def test_pool_reports_missing_sets(tmp_path):
    _fake_votes_file(tmp_path / "setA_probe_votes.json", ["1", "2"])
    pooled, missing = pool_set_votes(["setA", "setB"], tmp_path)
    assert len(pooled) == 2  # from setA
    assert missing == ["setB"]  # setB has no votes file


def test_pool_concatenates_across_sets(tmp_path):
    _fake_votes_file(tmp_path / "setA_probe_votes.json", ["1", "2"])
    _fake_votes_file(tmp_path / "setB_probe_votes.json", ["1", "2", "3"])
    pooled, missing = pool_set_votes(["setA", "setB"], tmp_path)
    assert len(pooled) == 5  # 2 + 3, concatenated
    assert missing == []


def test_fit_cohort_learns_probe_noise(tmp_path):
    _fake_votes_file(tmp_path / "setA_probe_votes.json", [str(i) for i in range(20)])
    pooled, _ = pool_set_votes(["setA"], tmp_path)
    model = fit_cohort(pooled)
    noise = probe_noise_dict(model)
    assert "fp" in noise
    assert set(noise["fp"]) == {"accuracy", "sigma_s", "inlier"}


def test_ablation_compare_flags_sigma_reordering():
    noise_a = {
        "fp": ProbeNoise(0.9, 0.3, 0.9),
        "chroma": ProbeNoise(0.8, 8.0, 0.8),
    }
    # Arm B inverts the σ ordering (fp now looser than chroma).
    noise_b = {
        "fp": ProbeNoise(0.9, 9.0, 0.9),
        "chroma": ProbeNoise(0.8, 2.0, 0.8),
    }
    a_dict = {
        p: {"accuracy": n.accuracy, "sigma_s": n.sigma_s, "inlier": n.inlier}
        for p, n in noise_a.items()
    }
    b_dict = {
        p: {"accuracy": n.accuracy, "sigma_s": n.sigma_s, "inlier": n.inlier}
        for p, n in noise_b.items()
    }
    cmp = ablation_compare(a_dict, b_dict)
    assert cmp["sigma_order_changed"] is True
    assert "fp" in cmp["per_probe"]
    assert cmp["per_probe"]["fp"]["sigma_a"] == 0.3
    assert cmp["per_probe"]["fp"]["sigma_b"] == 9.0
