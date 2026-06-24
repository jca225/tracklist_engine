"""Tests for UnmixDB label parsing and checkpoint round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")  # checkpoint/mert_model import torch; CI excludes it

from workspaces.alignment_prototype.external.checkpoint import (
    PretrainMeta,
    load_head,
    save_head,
)
from workspaces.alignment_prototype.external.unmixdb import (
    labels_to_targets,
    parse_labels,
)
from workspaces.alignment_prototype.mert_model import MertAlignHead, TrainConfig

_FIXTURE = (
    Path(__file__).resolve().parents[1] / "labeling/fixtures/unmixdb_sample.labels.txt"
)


def test_parse_labels_three_tracks() -> None:
    spans = parse_labels(_FIXTURE)
    assert len(spans) == 3
    assert spans[0].track_idx == 1
    assert spans[0].set_start_s == pytest.approx(0.0)
    assert spans[0].set_end_s == pytest.approx(38.9979138322, rel=1e-4)
    assert spans[0].tempo_ratio == pytest.approx(1.0)
    assert spans[1].tempo_ratio == pytest.approx(1.015)
    assert spans[2].tempo_ratio == pytest.approx(0.956)


def test_labels_to_targets_recording_ids() -> None:
    from workspaces.alignment_prototype.external.unmixdb import UnmixMix

    spans = parse_labels(_FIXTURE)
    mix = UnmixMix(
        mix_id="sample",
        mix_audio=_FIXTURE,
        labels_path=_FIXTURE,
        track_audio={sp.track_idx: _FIXTURE for sp in spans},
        spans=spans,
    )
    targets = labels_to_targets(mix)
    assert len(targets) == 3
    assert targets[0].recording_id == "07_Sr_Click_-_Tibetan_-_Antiritmo024.excerpt40"
    assert targets[0].claimed_stem == "regular"
    assert targets[0].slot_label == "samplet1"


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    head = MertAlignHead(12)
    ckpt = tmp_path / "pretrain.pt"
    meta = PretrainMeta(
        feature_kind="chroma",
        dim=12,
        n_heads=1,
        n_examples=99,
        n_mixes=33,
    )
    save_head(head, ckpt, meta=meta, cfg=TrainConfig(n_heads=1))
    loaded, loaded_meta = load_head(ckpt, expected_dim=12)
    assert loaded_meta.n_examples == 99
    for p0, p1 in zip(head.parameters(), loaded.parameters()):
        assert torch.allclose(p0, p1)
