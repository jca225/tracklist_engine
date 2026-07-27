from __future__ import annotations

import json

import pytest

from core.provenance import (
    ArtifactStore,
    Axis,
    AxisBelief,
    Decision,
    ProvenanceRepository,
    SubjectRef,
    connect,
)
from workspaces.alignment_prototype.belief_timeline import (
    axis_belief_bundle,
    beliefs_from_bundle,
    decode_and_register_timeline,
    decode_timeline_document,
    materialize_evaluation_view,
    prepare_belief_evaluation_timeline,
    write_belief_bundle,
)


def _belief(
    slot: str,
    axis: Axis,
    *,
    chosen: dict | None,
    belief_id: str,
) -> AxisBelief:
    candidate = (
        json.dumps(chosen, sort_keys=True, separators=(",", ":"))
        if chosen is not None
        else None
    )
    return AxisBelief(
        belief_id=belief_id,
        subject=SubjectRef("set-slot", f"set-a::{slot}"),
        axis=axis,
        inference_run_id=f"run-{belief_id}",
        posterior={candidate: 0.9} if candidate is not None else {},
        entropy=0.0,
        decision=Decision.ACCEPTED if candidate is not None else Decision.UNRESOLVED,
        chosen=candidate,
        contributing_evidence_ids=(f"ev-{belief_id}",),
    )


def _placement(slot: str, value: float | None, belief_id: str) -> AxisBelief:
    return _belief(
        slot,
        Axis.PLACEMENT,
        chosen=(
            {"schema": "placement-v1", "set_start_s": value}
            if value is not None
            else None
        ),
        belief_id=belief_id,
    )


def _structure(slot: str, belief_id: str, *, accepted: bool = True) -> AxisBelief:
    return _belief(
        slot,
        Axis.STRUCTURE,
        chosen=(
            {
                "schema": "structure-v1",
                "mix_time_frame": "set_absolute_s",
                "recording_id": "rec-a",
                "segments": [
                    {
                        "mix_start_s": 40.0,
                        "ref_start_s": 12.0,
                        "ref_end_s": 22.0,
                    }
                ],
            }
            if accepted
            else None
        ),
        belief_id=belief_id,
    )


TIMELINE = {
    "set_id": "set-a",
    "spans": [
        {
            "slot_label": "1",
            "recording_id": "rec-a",
            "set_start_s": 10.0,
            "set_end_s": 30.0,
            "ref_start_s": 2.0,
            "ref_end_s": 22.0,
            "ref_segments": [],
            "name": "one",
        },
        {
            "slot_label": "2",
            "recording_id": "rec-b",
            "set_start_s": 100.0,
            "set_end_s": 120.0,
            "ref_start_s": 7.0,
            "ref_end_s": 27.0,
            "ref_segments": [],
            "name": "two",
        },
    ],
}


BELIEFS = [
    _placement("1", 40.0, "p1"),
    _structure("1", "s1"),
    _placement("2", None, "p2"),
    _structure("2", "s2", accepted=False),
]


def test_belief_bundle_json_round_trip():
    bundle = axis_belief_bundle("set-a", BELIEFS)
    encoded = json.dumps(bundle, sort_keys=True)
    restored = beliefs_from_bundle(json.loads(encoded))
    assert restored == BELIEFS


def test_decoder_consumes_beliefs_and_persists_null_abstention():
    decoded = decode_timeline_document(TIMELINE, BELIEFS)
    first, second = decoded["spans"]

    assert first["set_start_s"] == 40.0
    assert first["set_end_s"] == 60.0  # original 20-second extent preserved
    assert first["ref_start_s"] == 12.0
    assert first["ref_segments"][0]["mix_start_s"] == 40.0
    assert first["axis_beliefs"]["placement"]["belief_id"] == "p1"

    assert second["set_start_s"] is None
    assert second["set_end_s"] is None
    assert second["ref_start_s"] is None
    assert second["ref_segments"] is None
    assert second["axis_beliefs"]["placement"]["chosen"] is None


def test_decoder_rejects_old_ref_offset_as_placement_shape():
    bad = [
        _belief(
            "1",
            Axis.PLACEMENT,
            chosen={"recording_id": "rec-a", "ref_start_s": 999.0},
            belief_id="bad-p",
        ),
        _structure("1", "s1"),
    ]
    one_span = {**TIMELINE, "spans": [TIMELINE["spans"][0]]}
    with pytest.raises(ValueError, match="candidate schema"):
        decode_timeline_document(one_span, bad)


def test_evaluation_view_reports_coverage_without_mutating_canonical_nulls():
    decoded = decode_timeline_document(TIMELINE, BELIEFS)
    view = materialize_evaluation_view(decoded)
    assert len(view["spans"]) == 1
    assert view["belief_evaluation"]["placement_abstained"] == 1
    assert view["spans"][0]["ref_start_s"] == 12.0
    assert decoded["spans"][1]["set_start_s"] is None
    assert decoded["spans"][1]["ref_start_s"] is None


def test_evaluation_view_restores_real_source_fields_but_skips_structure():
    beliefs = [
        _placement("1", 40.0, "p1"),
        _structure("1", "s1"),
        _placement("2", 110.0, "p2"),
        _structure("2", "s2", accepted=False),
    ]
    decoded = decode_timeline_document(TIMELINE, beliefs)
    view = materialize_evaluation_view(decoded)
    second = view["spans"][1]
    assert second["structure_abstained"] is True
    assert second["ref_start_s"] == 7.0  # source value, not a fabricated zero
    assert view["belief_evaluation"][
        "structure_abstained_among_placement_accepted"
    ] == 1
    assert decoded["spans"][1]["ref_start_s"] is None


def test_decoded_timeline_is_content_addressed_and_round_trips(tmp_path):
    timeline_path = tmp_path / "source.json"
    timeline_path.write_text(json.dumps(TIMELINE))
    conn = connect(tmp_path / "prov")
    store = ArtifactStore(tmp_path / "prov", conn)
    repo = ProvenanceRepository(conn)

    result = decode_and_register_timeline(
        timeline_path,
        BELIEFS,
        store=store,
        repo=repo,
        code_commit="abc",
    )

    with store.open(result.artifact.content_sha256) as handle:
        restored = json.loads(handle.read())
    assert restored == result.document
    assert result.n_spans == 2
    assert result.n_placement_abstained == 1
    assert result.n_structure_abstained == 1

    rows = conn.execute(
        "SELECT parent_sha256, relation FROM derivation "
        "WHERE child_sha256=? ORDER BY parent_sha256",
        (result.artifact.content_sha256,),
    ).fetchall()
    assert len(rows) == 2
    assert {row["relation"] for row in rows} == {"decoded_from"}


def test_prepare_evaluation_timeline_checks_set_and_writes_coverage(tmp_path):
    source = tmp_path / "source.json"
    bundle = tmp_path / "beliefs.json"
    output = tmp_path / "eval.json"
    source.write_text(json.dumps(TIMELINE))
    write_belief_bundle(bundle, "set-a", BELIEFS)

    coverage = prepare_belief_evaluation_timeline(
        source, bundle, output, expected_set_id="set-a"
    )
    assert coverage == {
        "source_spans": 2,
        "placement_accepted": 1,
        "placement_abstained": 1,
        "structure_abstained_among_placement_accepted": 0,
    }
    assert len(json.loads(output.read_text())["spans"]) == 1

    with pytest.raises(ValueError, match="set mismatch"):
        prepare_belief_evaluation_timeline(
            source, bundle, output, expected_set_id="wrong-set"
        )
