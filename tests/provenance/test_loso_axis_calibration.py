from __future__ import annotations

import json

import pytest

from core.provenance import (
    Axis,
    Decision,
    ProvenanceRepository,
    connect,
)
from alignment.loso_axis_calibration import (
    fit_loso_axis_model,
    fit_source_axis_calibration,
    infer_held_set_beliefs,
    model_document,
)
from alignment.score_timeline_vs_gt import SpanScore


def _score(
    slot: str, *, place: float | None, strict: float | None
) -> SpanScore:
    return SpanScore(
        slot=slot,
        recording_id=f"rec-{slot}",
        stem="regular",
        span_class="linear",
        id_correct=True,
        place_err_s=place,
        strict=strict,
        fiber=strict,
        ref_err_s=None,
        density=1,
    )


SPANS = [
    {
        "slot_label": str(index),
        "recording_id": f"rec-{index}",
        "set_start_s": float(index * 10),
        "set_end_s": float(index * 10 + 8),
        "ref_start_s": 2.0,
        "ref_end_s": 10.0,
        "ref_segments": [
            {
                "mix_start_s": float(index * 10),
                "ref_start_s": 2.0,
                "ref_end_s": 10.0,
            }
        ],
        "confidence": 0.8,
        "start_source": "fp" if index < 5 else "rare",
        "ref_decode_status": "path",
    }
    for index in range(6)
]
SCORES = [
    _score("0", place=1.0, strict=0.9),
    _score("1", place=2.0, strict=0.95),
    _score("2", place=3.0, strict=0.85),
    _score("3", place=30.0, strict=0.1),
    _score("4", place=40.0, strict=0.2),
    _score("5", place=1.0, strict=0.9),
]


def test_beta_calibration_is_deterministic_and_sparse_falls_back():
    calibration = fit_source_axis_calibration(
        train_set_id="train",
        timeline_spans=SPANS,
        scores=SCORES,
        axis=Axis.PLACEMENT,
        minimum_family_n=3,
    )
    probability, bucket = calibration.probability("fp")
    assert probability == pytest.approx(4 / 7)  # Beta(1+3, 1+2)
    assert bucket == "fp"

    sparse_probability, sparse_bucket = calibration.probability("rare")
    assert sparse_bucket == "__global__"
    assert sparse_probability == pytest.approx(5 / 8)  # Beta(1+4, 1+2)


def test_loso_refuses_same_set_and_stamps_development_only():
    with pytest.raises(ValueError, match="LOSO violation"):
        fit_loso_axis_model(
            train_set_id="same",
            held_set_id="same",
            timeline_spans=SPANS,
            scores=SCORES,
        )

    model = fit_loso_axis_model(
        train_set_id="train",
        held_set_id="held",
        timeline_spans=SPANS,
        scores=SCORES,
    )
    doc = model_document(model)
    assert doc["intended_use"] == "development"
    assert doc["independent_set_count"] == 2
    assert doc["placement"]["calibration_status"] == "development_only"


def test_frozen_cross_set_model_emits_separate_axis_beliefs(tmp_path):
    model = fit_loso_axis_model(
        train_set_id="train",
        held_set_id="held",
        timeline_spans=SPANS,
        scores=SCORES,
        minimum_family_n=1,
    )
    held_span = {
        **SPANS[0],
        "slot_label": "h1",
        "recording_id": "rec-held",
    }
    held = {"set_id": "held", "spans": [held_span]}
    repo = ProvenanceRepository(connect(tmp_path))
    run = repo.begin_run(
        process="test",
        process_version="1",
        code_commit="abc",
        params_hash="0",
    )

    beliefs = infer_held_set_beliefs(
        model=model,
        held_timeline=held,
        repo=repo,
        run=run,
        minimum_posterior=0.5,
    )

    assert [belief.axis for belief in beliefs] == [
        Axis.PLACEMENT,
        Axis.STRUCTURE,
    ]
    assert all(belief.decision is Decision.ACCEPTED for belief in beliefs)
    placement = json.loads(beliefs[0].chosen or "")
    structure = json.loads(beliefs[1].chosen or "")
    assert placement["set_start_s"] == held_span["set_start_s"]
    assert structure["mix_time_frame"] == "set_absolute_s"
    assert structure["segments"][0]["mix_start_s"] == held_span["set_start_s"]


def test_model_cannot_apply_to_training_set(tmp_path):
    model = fit_loso_axis_model(
        train_set_id="train",
        held_set_id="held",
        timeline_spans=SPANS,
        scores=SCORES,
    )
    repo = ProvenanceRepository(connect(tmp_path))
    run = repo.begin_run(
        process="test",
        process_version="1",
        code_commit="abc",
        params_hash="0",
    )
    with pytest.raises(ValueError, match="held timeline set"):
        infer_held_set_beliefs(
            model=model,
            held_timeline={"set_id": "train", "spans": []},
            repo=repo,
            run=run,
        )
