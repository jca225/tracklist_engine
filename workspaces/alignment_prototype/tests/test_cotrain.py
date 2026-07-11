"""Tests for co-training dataset loading."""

from pathlib import Path
from workspaces.alignment_prototype.dataset import load_set
from core.result import Ok

_REPO = Path(__file__).resolve().parents[3]
BB12 = _REPO / "labeling/fixtures/bb12_ground_truth.yaml"


def test_load_set_stamps_set_id_on_every_target():
    match load_set(BB12):
        case Ok((gt, targets)):
            assert len(targets) > 0
            assert all(t.set_id == gt.set_id for t in targets)
            assert gt.set_id  # non-empty
        case _:
            raise AssertionError("bb12 fixture failed to load")
