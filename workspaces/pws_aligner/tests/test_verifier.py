"""Tests for workspaces.pws_aligner.verifier."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from workspaces.pws_aligner.continuous_model import ProbeNoise
from workspaces.pws_aligner.verifier import (
    JointEstimate,
    calibration_report,
    continuous_calibration_report,
    prune_confident_errors,
    sigma_rank_inversions,
)
from workspaces.pws_aligner.votes import AbstainReason, Vote


# ---------------------------------------------------------------------------
# prune_confident_errors
# ---------------------------------------------------------------------------


def test_prunes_injected_label_errors():
    # 100 spans; soft label = argmax; inject 10 flips the model is confident are wrong
    soft = {f"s{i}": ("A" if i % 2 == 0 else "B") for i in range(100)}
    proba = {
        sid: ({"A": 0.9, "B": 0.1} if lbl == "A" else {"A": 0.1, "B": 0.9})
        for sid, lbl in soft.items()
    }
    # flip 10 labels against a confident model prediction
    for i in range(0, 20, 2):
        soft[f"s{i}"] = "B"  # label says B but proba says A strongly
    to_prune, _ = prune_confident_errors(soft, proba)
    assert {f"s{i}" for i in range(0, 20, 2)} <= to_prune


def test_prune_returns_joint_estimate():
    soft = {"a": "A", "b": "B", "c": "A"}
    proba = {
        "a": {"A": 0.9, "B": 0.1},
        "b": {"A": 0.1, "B": 0.9},
        "c": {"A": 0.8, "B": 0.2},
    }
    to_prune, est = prune_confident_errors(soft, proba)
    assert isinstance(est, JointEstimate)
    assert est.classes == ("A", "B")
    # 2x2 matrices
    assert len(est.raw_counts) == 2
    assert len(est.calibrated_joint) == 2
    assert len(est.raw_counts[0]) == 2


def test_prune_empty_input():
    to_prune, est = prune_confident_errors({}, {})
    assert to_prune == set()
    assert est.classes == ()


def test_no_errors_when_labels_consistent():
    """When every label matches the argmax, nothing should be pruned."""
    soft = {f"s{i}": ("A" if i % 2 == 0 else "B") for i in range(20)}
    proba = {
        sid: ({"A": 0.9, "B": 0.1} if lbl == "A" else {"A": 0.1, "B": 0.9})
        for sid, lbl in soft.items()
    }
    to_prune, _ = prune_confident_errors(soft, proba)
    assert len(to_prune) == 0


# ---------------------------------------------------------------------------
# calibration_report
# ---------------------------------------------------------------------------


def _write_votes_file(path: Path, spans: list[dict]) -> None:
    path.write_text(json.dumps(spans, indent=2))


def _write_accuracy_file(path: Path, acc: dict[str, float]) -> None:
    path.write_text(json.dumps(acc, indent=2))


def test_calibration_report_basic(tmp_path: Path) -> None:
    """Two probes: one whose learned ≈ gt_measured, one that diverges >0.15."""
    # Build a synthetic votes file with 4 spans.
    # span 1: recording_id="r1", set_start=10, probe "fp" votes r1@offset 5.0,
    #         probe "mert" votes r1@offset 5.0 (both correct)
    # span 2: recording_id="r2", set_start=100, probe "fp" votes r2@offset 10.0 (correct)
    #         probe "mert" votes r1@offset 5.0 (wrong recording_id → incorrect)
    # span 3: recording_id="r1", set_start=200, probe "fp" abstains
    #         probe "mert" votes r1@offset 5.0 (correct)
    # span 4: recording_id="r2", set_start=300, probe "fp" votes r2@offset 10.0 (correct)
    #         probe "mert" votes r2@offset 999.0 (wrong offset bin → incorrect)

    spans = [
        {
            "span_id": "1",
            "slot_label": "1",
            "recording_id": "r1",
            "set_start_s": 10.0,
            "set_end_s": 50.0,
            "ref_start_s": 15.0,
            "ref_end_s": 55.0,
            "confidence": 0.8,
            "name": "Track 1",
            "probes": [
                {
                    "probe": "fp",
                    "recording_id": "r1",
                    "offset_s": 5.0,
                    "confidence": 0.8,
                    "abstain": False,
                    "features": [],
                },
                {
                    "probe": "mert",
                    "recording_id": "r1",
                    "offset_s": 5.0,
                    "confidence": 0.7,
                    "abstain": False,
                    "features": [],
                },
            ],
        },
        {
            "span_id": "2",
            "slot_label": "2",
            "recording_id": "r2",
            "set_start_s": 100.0,
            "set_end_s": 140.0,
            "ref_start_s": 110.0,
            "ref_end_s": 150.0,
            "confidence": 0.8,
            "name": "Track 2",
            "probes": [
                {
                    "probe": "fp",
                    "recording_id": "r2",
                    "offset_s": 10.0,
                    "confidence": 0.8,
                    "abstain": False,
                    "features": [],
                },
                {
                    "probe": "mert",
                    "recording_id": "r1",
                    "offset_s": 5.0,
                    "confidence": 0.6,
                    "abstain": False,
                    "features": [],
                },
            ],
        },
        {
            "span_id": "3",
            "slot_label": "3",
            "recording_id": "r1",
            "set_start_s": 200.0,
            "set_end_s": 240.0,
            "ref_start_s": 205.0,
            "ref_end_s": 245.0,
            "confidence": 0.7,
            "name": "Track 3",
            "probes": [
                {
                    "probe": "fp",
                    "recording_id": None,
                    "offset_s": 0.0,
                    "confidence": 0.0,
                    "abstain": True,
                    "features": [],
                },
                {
                    "probe": "mert",
                    "recording_id": "r1",
                    "offset_s": 5.0,
                    "confidence": 0.7,
                    "abstain": False,
                    "features": [],
                },
            ],
        },
        {
            "span_id": "4",
            "slot_label": "4",
            "recording_id": "r2",
            "set_start_s": 300.0,
            "set_end_s": 340.0,
            "ref_start_s": 310.0,
            "ref_end_s": 350.0,
            "confidence": 0.7,
            "name": "Track 4",
            "probes": [
                {
                    "probe": "fp",
                    "recording_id": "r2",
                    "offset_s": 10.0,
                    "confidence": 0.8,
                    "abstain": False,
                    "features": [],
                },
                # mert votes r2 but wrong offset (bin 999/2=499 ≠ GT bin 5)
                {
                    "probe": "mert",
                    "recording_id": "r2",
                    "offset_s": 998.0,
                    "confidence": 0.5,
                    "abstain": False,
                    "features": [],
                },
            ],
        },
    ]

    # GT rows: r1 at set_start=10, ref_start=15 → offset=5, bin=round(5/2)=2
    #          r1 at set_start=200, ref_start=205 → offset=5, bin=2
    #          r2 at set_start=100, ref_start=110 → offset=10, bin=5
    #          r2 at set_start=300, ref_start=310 → offset=10, bin=5
    gt_rows = [
        {
            "track_id": "r1",
            "slot_label": "1",
            "set_start_s": 10.0,
            "set_end_s": 50.0,
            "ref_start_s": 15.0,
            "ref_end_s": 55.0,
        },
        {
            "track_id": "r1",
            "slot_label": "3",
            "set_start_s": 200.0,
            "set_end_s": 240.0,
            "ref_start_s": 205.0,
            "ref_end_s": 245.0,
        },
        {
            "track_id": "r2",
            "slot_label": "2",
            "set_start_s": 100.0,
            "set_end_s": 140.0,
            "ref_start_s": 110.0,
            "ref_end_s": 150.0,
        },
        {
            "track_id": "r2",
            "slot_label": "4",
            "set_start_s": 300.0,
            "set_end_s": 340.0,
            "ref_start_s": 310.0,
            "ref_end_s": 350.0,
        },
    ]

    # DS-learned: fp at 0.90 (will diverge from gt if gt_measured is very different),
    # mert at 0.80 (will diverge from gt given many wrong votes below)
    # GT-measured:
    #   fp: span1 correct, span2 correct, span3 abstain (skip), span4 correct → 3/3 = 1.0
    #       |learned 0.90 − gt 1.0| = 0.10 < 0.15 → no diverge
    #   mert: span1 correct (r1@bin2), span2 wrong (r1 vs r2), span3 correct (r1@bin2), span4 wrong (r2@bin499 vs r2@bin5)
    #         → 2/4 = 0.50
    #       |learned 0.80 − gt 0.50| = 0.30 > 0.15 → DIVERGES

    votes_path = tmp_path / "test_probe_votes.json"
    accuracy_path = tmp_path / "test_pws_probe_accuracy.json"
    _write_votes_file(votes_path, spans)
    _write_accuracy_file(accuracy_path, {"fp": 0.90, "mert": 0.80})

    report = calibration_report(
        "test_set",
        votes_path=votes_path,
        accuracy_path=accuracy_path,
        bin_s=2.0,
        _gt_rows_override=gt_rows,
    )

    assert "fp" in report
    assert "mert" in report

    # fp: learned=0.90, gt_measured=1.0 → |diff|=0.10 → no diverge
    assert abs(report["fp"]["learned"] - 0.90) < 1e-6
    assert abs(report["fp"]["gt_measured"] - 1.0) < 1e-6
    assert report["fp"]["n_votes"] == 3  # span3 fp abstains
    assert report["fp"]["diverges"] is False

    # mert: learned=0.80, gt_measured=0.50 → |diff|=0.30 > 0.15 → diverges
    assert abs(report["mert"]["learned"] - 0.80) < 1e-6
    assert abs(report["mert"]["gt_measured"] - 0.50) < 1e-6
    assert report["mert"]["n_votes"] == 4
    assert report["mert"]["diverges"] is True


def test_calibration_report_missing_votes_exits(tmp_path: Path) -> None:
    """Missing votes file should sys.exit."""
    with pytest.raises(SystemExit):
        calibration_report(
            "nonexistent",
            votes_path=tmp_path / "missing_votes.json",
            accuracy_path=tmp_path / "missing_acc.json",
        )


def test_calibration_report_missing_accuracy_exits(tmp_path: Path) -> None:
    """Missing accuracy file should sys.exit."""
    votes_path = tmp_path / "votes.json"
    votes_path.write_text("[]")
    with pytest.raises(SystemExit):
        calibration_report(
            "nonexistent",
            votes_path=votes_path,
            accuracy_path=tmp_path / "missing_acc.json",
        )


# ---------------------------------------------------------------------------
# continuous_calibration_report
# ---------------------------------------------------------------------------


def _vote(probe: str, span_id: str, recording_id: str, offset_s: float) -> Vote:
    """Helper: build a non-abstained Vote for testing."""
    return Vote(
        probe=probe,
        span_id=span_id,
        recording_id=recording_id,
        offset_s=offset_s,
        confidence=0.8,
        abstained=False,
        reason=AbstainReason.NONE,
        features=(),
    )


def _spans_with_known_residuals():
    # probe "tight": residuals ~0.1s; probe "loose": residuals ~5s; 12 spans each
    spans, gt = [], {}
    for i in range(12):
        sid = f"s{i}"
        gt[sid] = ("rec", 100.0)
        spans.append(
            (
                _vote("tight", sid, "rec", 100.0 + (0.1 if i % 2 else -0.1)),
                _vote("loose", sid, "rec", 100.0 + (5.0 if i % 2 else -5.0)),
            )
        )
    return spans, gt


def test_report_measures_mad_per_probe():
    spans, gt = _spans_with_known_residuals()
    noise = {"tight": ProbeNoise(0.9, 0.2, 0.9), "loose": ProbeNoise(0.7, 6.0, 0.8)}
    report = {r.probe: r for r in continuous_calibration_report(noise, spans, gt)}
    assert abs(report["tight"].measured_mad_s - 0.1) < 1e-6
    assert abs(report["loose"].measured_mad_s - 5.0) < 1e-6
    assert report["tight"].n_scored == 12


def test_sigma_rank_inversion_tripwire():
    spans, gt = _spans_with_known_residuals()
    # Learned sigmas INVERTED vs measured: model trusts the loose probe more.
    noise = {"tight": ProbeNoise(0.9, 6.0, 0.9), "loose": ProbeNoise(0.7, 0.2, 0.8)}
    report = continuous_calibration_report(noise, spans, gt)
    inversions = sigma_rank_inversions(report)
    assert inversions == [("tight", "loose")] or inversions == [("loose", "tight")]
