"""Leakage-guarded --train-yaml wiring for the E1 flywheel."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from alignment.trajectory.train import load_pseudo_train_yaml


def _pseudo_yaml(
    tmp_path: Path,
    *,
    set_id: str = "w1mgcjt",
    provenance: dict | None = {"source": "test"},
    tracks: list[dict] | None = None,
) -> Path:
    if tracks is None:
        tracks = [
            {
                "slot_label": "1",
                "track_id": "t1",
                "recording_id": "r1",
                "claimed_stem": "regular",
                "set_start_s": 0.0,
                "set_end_s": 10.0,
                "ref_start_s": 0.0,
                "ref_end_s": 10.0,
                "tempo_ratio": 1.0,
                "ref_segments": [
                    {"mix_start_s": 0.0, "ref_start_s": 0.0, "ref_end_s": 10.0}
                ],
                "pseudo_label": True,
            }
        ]
    doc: dict = {"set_id": set_id, "tracks": tracks}
    if provenance is not None:
        doc["provenance"] = provenance
    path = tmp_path / f"{set_id}_pseudo_gt.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def test_load_pseudo_train_yaml_rejects_eval_leakage(tmp_path):
    path = _pseudo_yaml(tmp_path, set_id="2nvzlh2k")
    with pytest.raises(ValueError, match="same set"):
        load_pseudo_train_yaml(path, eval_set="2nvzlh2k")


def test_load_pseudo_train_yaml_requires_provenance_and_rows(tmp_path):
    path = _pseudo_yaml(tmp_path, provenance=None, tracks=[])
    with pytest.raises(ValueError, match="provenance|tracks"):
        load_pseudo_train_yaml(path, eval_set="2nvzlh2k")


def test_load_pseudo_train_yaml_rejects_non_pseudo_rows(tmp_path):
    path = _pseudo_yaml(
        tmp_path,
        tracks=[
            {
                "slot_label": "1",
                "track_id": "t1",
                "set_start_s": 0.0,
                "set_end_s": 10.0,
                "pseudo_label": False,
            }
        ],
    )
    with pytest.raises(ValueError, match="pseudo_label"):
        load_pseudo_train_yaml(path, eval_set="2nvzlh2k")


def test_load_pseudo_train_yaml_ok(tmp_path):
    path = _pseudo_yaml(tmp_path, set_id="w1mgcjt")
    sid, yaml_path = load_pseudo_train_yaml(path, eval_set="2nvzlh2k")
    assert sid == "w1mgcjt"
    assert yaml_path == path


def test_main_rejects_same_set_before_dataset(tmp_path, monkeypatch):
    from alignment.trajectory import train as train_mod

    called = {"ds": 0}

    class Boom:
        def __init__(self, *a, **k):
            called["ds"] += 1
            raise AssertionError("dataset must not be built on leakage")

    monkeypatch.setattr(train_mod, "TrajectorySpanDataset", Boom)
    path = _pseudo_yaml(tmp_path, set_id="2nvzlh2k")
    with pytest.raises(ValueError, match="same set"):
        train_mod.main(
            [
                "--split",
                "set",
                "--train-yaml",
                str(path),
                "--eval-set",
                "2nvzlh2k",
                "--epochs",
                "1",
            ]
        )
    assert called["ds"] == 0
