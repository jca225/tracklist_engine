from __future__ import annotations

import json

import pytest

from core.provenance import (
    Axis,
    Decision,
    DecisionRule,
    ProvenanceRepository,
    SubjectRef,
    connect,
)
from workspaces.alignment_prototype.harness.contract import AlignmentResult, RefSegment
from workspaces.alignment_prototype.placement_structure_beliefs import (
    CalibratedAxisProbability,
    record_placement_structure_beliefs,
)


PLACEMENT_RULE = DecisionRule("placement-v1", Axis.PLACEMENT, "1", 0.6, 0.8)
STRUCTURE_RULE = DecisionRule("structure-v1", Axis.STRUCTURE, "1", 0.6, 0.8)


def _repo_run(tmp_path):
    repo = ProvenanceRepository(connect(tmp_path))
    run = repo.begin_run(
        process="path_decode",
        process_version="1",
        code_commit="abc",
        params_hash="params",
    )
    return repo, run


def test_committed_probe_records_independent_axis_beliefs(tmp_path):
    repo, run = _repo_run(tmp_path)
    result = AlignmentResult(
        recording_id="rec-1",
        offset_s=12.5,
        ref_end_s=24.0,
        segments=(
            RefSegment(0.0, 12.5, 18.0),
            RefSegment(5.5, 4.0, 9.5),
        ),
        confidence=0.7,
        source="path",
    )

    beliefs = record_placement_structure_beliefs(
        repo=repo,
        run=run,
        subject=SubjectRef("set-slot", "set::001"),
        result=result,
        placement_rule=PLACEMENT_RULE,
        structure_rule=STRUCTURE_RULE,
        set_start_s=101.25,
        segment_mix_time_origin_s=101.25,
        placement_probability=CalibratedAxisProbability(
            Axis.PLACEMENT, 0.8, "path-placement-cal-v1"
        ),
        structure_probability=CalibratedAxisProbability(
            Axis.STRUCTURE, 0.75, "path-structure-cal-v1"
        ),
    )

    assert beliefs.placement.axis is Axis.PLACEMENT
    assert beliefs.structure.axis is Axis.STRUCTURE
    assert beliefs.placement.decision is Decision.ACCEPTED
    assert beliefs.structure.decision is Decision.ACCEPTED
    placement = json.loads(beliefs.placement.chosen or "")
    structure = json.loads(beliefs.structure.chosen or "")
    assert placement == {"schema": "placement-v1", "set_start_s": 101.25}
    assert structure["schema"] == "structure-v1"
    assert structure["mix_time_frame"] == "set_absolute_s"
    assert structure["segments"][0]["mix_start_s"] == 101.25
    assert len(structure["segments"]) == 2
    assert (
        beliefs.placement.contributing_evidence_ids
        != beliefs.structure.contributing_evidence_ids
    )


def test_abstention_persists_empty_posteriors_and_null_choices(tmp_path):
    repo, run = _repo_run(tmp_path)
    result = AlignmentResult.abstained(source="path", recording_id="rec-1")

    beliefs = record_placement_structure_beliefs(
        repo=repo,
        run=run,
        subject=SubjectRef("set-slot", "set::001"),
        result=result,
        placement_rule=PLACEMENT_RULE,
        structure_rule=STRUCTURE_RULE,
    )

    assert result.offset_s is None
    assert beliefs.placement.posterior == {}
    assert beliefs.structure.posterior == {}
    assert beliefs.placement.decision is Decision.UNRESOLVED
    assert beliefs.structure.decision is Decision.UNRESOLVED
    assert beliefs.placement.chosen is None
    assert beliefs.structure.chosen is None


def test_raw_probe_confidence_is_not_accepted_as_a_posterior(tmp_path):
    repo, run = _repo_run(tmp_path)
    result = AlignmentResult(
        recording_id="rec-1",
        offset_s=3.0,
        confidence=0.99,
        source="path",
    )

    with pytest.raises(ValueError, match="require .*calibrated"):
        record_placement_structure_beliefs(
            repo=repo,
            run=run,
            subject=SubjectRef("set-slot", "set::001"),
            result=result,
            placement_rule=PLACEMENT_RULE,
            structure_rule=STRUCTURE_RULE,
            set_start_s=10.0,
        )


def test_abstention_rejects_fabricated_probability(tmp_path):
    repo, run = _repo_run(tmp_path)
    with pytest.raises(ValueError, match="cannot carry"):
        record_placement_structure_beliefs(
            repo=repo,
            run=run,
            subject=SubjectRef("set-slot", "set::001"),
            result=AlignmentResult.abstained(source="path"),
            placement_rule=PLACEMENT_RULE,
            structure_rule=STRUCTURE_RULE,
            set_start_s=None,
            placement_probability=CalibratedAxisProbability(
                Axis.PLACEMENT, 0.8, "cal-v1"
            ),
        )


def test_ref_offset_cannot_silently_become_mix_time_placement(tmp_path):
    repo, run = _repo_run(tmp_path)
    result = AlignmentResult(
        recording_id="rec-1",
        offset_s=987.0,
        confidence=0.9,
        source="path",
    )
    with pytest.raises(ValueError, match="explicit set_start_s"):
        record_placement_structure_beliefs(
            repo=repo,
            run=run,
            subject=SubjectRef("set-slot", "set::001"),
            result=result,
            placement_rule=PLACEMENT_RULE,
            structure_rule=STRUCTURE_RULE,
            placement_probability=CalibratedAxisProbability(
                Axis.PLACEMENT, 0.9, "cal-v1"
            ),
        )


def test_probe_without_segments_abstains_on_structure(tmp_path):
    repo, run = _repo_run(tmp_path)
    result = AlignmentResult(
        recording_id="rec-1",
        offset_s=4.0,
        confidence=0.8,
        source="fp",
    )
    beliefs = record_placement_structure_beliefs(
        repo=repo,
        run=run,
        subject=SubjectRef("set-slot", "set::001"),
        result=result,
        placement_rule=PLACEMENT_RULE,
        structure_rule=STRUCTURE_RULE,
        set_start_s=50.0,
        placement_probability=CalibratedAxisProbability(
            Axis.PLACEMENT, 0.8, "placement-cal-v1"
        ),
    )
    assert beliefs.placement.decision is Decision.ACCEPTED
    assert beliefs.structure.decision is Decision.UNRESOLVED
    assert beliefs.structure.chosen is None
