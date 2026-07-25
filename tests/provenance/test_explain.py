"""Brick 4 — the read side (Law 21: every_published_value_is_explainable).

Builds a small provenanced world (identity belief + placement/structure obs on
one subject, plus a derived artifact) and asserts explain_subject / lineage
surface it faithfully: all three axes on one subject, the evidence behind a
belief, diagnostics on observations, and the derivation graph walked to its root.
"""

from __future__ import annotations

import pytest

from core.provenance import (
    ArtifactKind,
    ArtifactStore,
    Axis,
    Decision,
    DecisionRule,
    EvidenceDirection,
    ObservationStatus,
    ProvenanceRepository,
    SubjectRef,
    connect,
    explain_subject,
    lineage,
)


@pytest.fixture
def world(tmp_path):
    conn = connect(tmp_path)
    store = ArtifactStore(tmp_path, conn)
    repo = ProvenanceRepository(conn)
    return conn, store, repo


ID_RULE = DecisionRule("id-v1", Axis.IDENTITY, "1", 0.6, 0.9)
SUBJ = SubjectRef("set-slot", "1fsnxchk::7")


def _seed_identity(store, repo, *, posterior):
    run = repo.begin_run(
        process="identity.export", process_version="1", code_commit="c", params_hash="0"
    )
    src = store.put_bytes(b"<row/>", kind=ArtifactKind.HTML_ROW, media_type="text/html")
    repo.record_observation(
        subject=SUBJ,
        predicate="claimed_recording",
        value="rec_a",
        source=src,
        run=run,
        status=ObservationStatus.OBSERVED,
    )
    claim = repo.record_claim(
        axis=Axis.IDENTITY,
        subject=SUBJ,
        predicate="is_recording",
        candidate=SubjectRef("recording", "rec_a"),
        run=run,
        context={},
    )
    ev = repo.record_evidence(
        claim=claim,
        source_family="tokenizer_claim",
        direction=EvidenceDirection.SUPPORTS,
        run=run,
        native_score={"prior": 1.0},
    )
    repo.record_belief(
        subject=SUBJ,
        axis=Axis.IDENTITY,
        posterior=posterior,
        rule=ID_RULE,
        run=run,
        contributing_evidence_ids=[ev.evidence_id],
    )
    repo.succeed(run)
    return src


def test_explain_subject_surfaces_all_axes(world):
    conn, store, repo = world
    _seed_identity(store, repo, posterior={"rec_a": 0.95})
    # add a placement observation on the SAME subject via a second run
    run2 = repo.begin_run(
        process="timeline.register",
        process_version="1",
        code_commit="c",
        params_hash="0",
    )
    tl = store.put_bytes(
        b"{}", kind=ArtifactKind.PREDICTED_TIMELINE, media_type="application/json"
    )
    repo.record_observation(
        subject=SUBJ,
        predicate="set_start_s",
        value={"chosen_s": 12.0},
        source=tl,
        run=run2,
        diagnostic_code="uncalibrated_priority_decode",
    )
    repo.succeed(run2)

    exp = explain_subject(conn, "1fsnxchk::7")
    assert not exp.is_empty
    # identity axis: accepted, with the tokenizer evidence attached
    id_axis = next(a for a in exp.axes if a.axis == Axis.IDENTITY.value)
    assert id_axis.decision == Decision.ACCEPTED.value
    assert id_axis.chosen == "rec_a"
    assert id_axis.candidate == "rec_a"
    assert id_axis.evidence[0].source_family == "tokenizer_claim"
    # placement observation surfaces with its diagnostic + producing run
    place = next(o for o in exp.observations if o.predicate == "set_start_s")
    assert place.diagnostic_code == "uncalibrated_priority_decode"
    assert place.producer_process == "timeline.register"


def test_explain_shows_abstention(world):
    conn, store, repo = world
    _seed_identity(store, repo, posterior={"a": 0.5, "b": 0.5})  # ambiguous → abstain
    exp = explain_subject(conn, "1fsnxchk::7")
    id_axis = next(a for a in exp.axes if a.axis == Axis.IDENTITY.value)
    assert id_axis.decision == Decision.UNRESOLVED.value
    assert id_axis.chosen is None


def test_explain_unknown_subject_is_empty(world):
    conn, _store, _repo = world
    exp = explain_subject(conn, "nope::0")
    assert exp.is_empty
    assert exp.axes == [] and exp.observations == []


def test_lineage_walks_derivation_to_root(world):
    conn, store, repo = world
    run = repo.begin_run(
        process="timeline.register",
        process_version="1",
        code_commit="c",
        params_hash="0",
    )
    parent = store.put_bytes(
        b"cfg", kind=ArtifactKind.DIAGNOSTICS, media_type="application/json"
    )
    child = store.put_bytes(
        b"tl", kind=ArtifactKind.PREDICTED_TIMELINE, media_type="application/json"
    )
    repo.add_output(run, child, role="predicted_timeline")
    repo.derive(child, parent, run, relation="registered_from")
    repo.succeed(run)

    nodes = lineage(conn, child.content_sha256)
    assert [n.depth for n in nodes] == [0, 1]
    assert nodes[0].kind == ArtifactKind.PREDICTED_TIMELINE.value
    assert nodes[0].producer_process == "timeline.register"
    assert nodes[1].kind == ArtifactKind.DIAGNOSTICS.value
    assert nodes[1].relation == "registered_from"  # edge to its child


def test_lineage_unknown_artifact_yields_unknown_node(world):
    # An unknown sha yields a single "(unknown)" node, not a crash.
    conn, _store, _repo = world
    nodes = lineage(conn, "deadbeef" * 8)
    assert len(nodes) == 1 and nodes[0].kind == "(unknown)"
