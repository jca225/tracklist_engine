"""Tests for anchor check and aligner dataset."""
from __future__ import annotations

from pathlib import Path

from labeling.anchor_check import _fmt_time, compare_anchors
from labeling.ground_truth.schema import GroundTruthSet, GroundTruthTrack, load
from core.result import Ok
from workspaces.alignment_prototype.dataset import load_set, track_to_target
from workspaces.alignment_prototype.losses import batch_loss
from workspaces.alignment_prototype.model import CopyGTBaseline


def test_fmt_time():
    assert _fmt_time(64.5) == "1:04.500"
    assert _fmt_time(3729.0).startswith("62:")


def test_compare_anchors_identical():
    t = GroundTruthTrack(
        label="Test", track_id="abc", claimed_stem="regular",
        set_start_s=10.0, set_end_s=20.0, ref_start_s=0.0, ref_end_s=5.0,
        slot_label="002",
    )
    gt = GroundTruthSet(set_id="x", tracks=(t,))
    results = compare_anchors(gt, gt, ("002",))
    assert len(results) == 1
    assert results[0].ok
    assert results[0].delta_start_s == 0.0


def test_load_bb12_fixture():
    yaml_path = Path("labeling/fixtures/bb12_ground_truth.yaml")
    if not yaml_path.is_file():
        return
    match load_set(yaml_path):
        case Ok((gt, targets)):
            assert gt.set_id == "1fsnxchk"
            assert len(targets) > 100
        case _:
            raise AssertionError("expected Ok")


def test_copy_gt_baseline_zero_loss():
    yaml_path = Path("labeling/fixtures/bb12_ground_truth.yaml")
    if not yaml_path.is_file():
        return
    match load(yaml_path):
        case Ok(gt):
            targets = tuple(track_to_target(t) for t in gt.tracks)
            preds = CopyGTBaseline().predict(targets)
            assert batch_loss(preds, targets) == 0.0


def test_held_out_eval_copy_gt_zero():
    yaml_path = Path("labeling/fixtures/bb12_ground_truth.yaml")
    if not yaml_path.is_file():
        return
    from workspaces.alignment_prototype.eval import evaluate
    from workspaces.alignment_prototype.split import split_targets

    match load_set(yaml_path):
        case Ok((_gt, targets)):
            train, eval_ = split_targets(targets, eval_fraction=0.2)
            assert len(train) + len(eval_) == len(targets)
            assert len(eval_) >= 20
            preds = CopyGTBaseline().predict(eval_)
            report = evaluate(preds, eval_)
            assert report.batch_loss == 0.0
            assert report.identity_accuracy == 1.0
