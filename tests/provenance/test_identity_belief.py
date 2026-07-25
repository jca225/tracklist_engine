"""Brick 2 tests — identity expressed as claim → evidence → belief (§5/§9).

The load-bearing behaviors: the tokenizer claim is *evidence*, not identity; a
low/ambiguous posterior ABSTAINS rather than guessing; axes are decided
independently; and the whole chain is provenanced (belief → evidence → claim →
run).
"""

from __future__ import annotations

import math

import pytest

from core.provenance import (
    ArtifactKind,
    ArtifactStore,
    Axis,
    Decision,
    DecisionRule,
    EvidenceDirection,
    ProvenanceRepository,
    connect,
    decide,
    entropy,
)


@pytest.fixture
def repo_stack(tmp_path):
    conn = connect(tmp_path)
    return ArtifactStore(tmp_path, conn), ProvenanceRepository(conn)


ID_RULE = DecisionRule(
    decision_rule_id="id-v1",
    axis=Axis.IDENTITY,
    version="1",
    minimum_posterior=0.6,
    maximum_entropy=0.9,
)


# --- pure decide (§9) ------------------------------------------------------
def test_decide_accepts_confident_winner():
    d, chosen = decide({"recA": 0.8, "recB": 0.2}, ID_RULE)
    assert d == Decision.ACCEPTED and chosen == "recA"


def test_decide_abstains_below_posterior_floor():
    d, chosen = decide({"recA": 0.5, "recB": 0.5}, ID_RULE)
    assert d == Decision.UNRESOLVED and chosen is None


def test_decide_abstains_on_high_entropy():
    # four-way near-uniform: winner just over the floor but posterior too spread.
    post = {"a": 0.61, "b": 0.13, "c": 0.13, "d": 0.13}
    assert entropy(post) > ID_RULE.maximum_entropy
    d, chosen = decide(post, ID_RULE)
    assert d == Decision.UNRESOLVED and chosen is None


def test_entropy_zero_for_degenerate():
    assert entropy({"only": 1.0}) == 0.0
    assert entropy({}) == 0.0


# --- the tokenizer claim is EVIDENCE, not identity (Law 4/5) ---------------
def test_tokenizer_claim_is_evidence_and_can_be_overridden(repo_stack):
    store, repo = repo_stack
    run = repo.begin_run(
        process="identity.infer",
        process_version="1",
        code_commit="abc",
        params_hash="0",
    )
    src = store.put_bytes(b"<row/>", kind=ArtifactKind.HTML_ROW, media_type="text/html")

    from core.provenance import SubjectRef

    slot = SubjectRef("set-slot", "2nvzlh2k::013w1")
    claimed = SubjectRef("recording", "rec_claimed")
    audio_says = SubjectRef("recording", "rec_audio")

    # Claim carries the tokenizer's proposal — but only as a hypothesis.
    claim = repo.record_claim(
        axis=Axis.IDENTITY,
        subject=slot,
        predicate="is_recording",
        candidate=claimed,
        run=run,
        context={"source": "set_track_slots"},
    )
    # Tokenizer claim = one piece of (weak) supporting evidence.
    ev_tok = repo.record_evidence(
        claim=claim,
        source_family="tokenizer_claim",
        direction=EvidenceDirection.SUPPORTS,
        run=run,
        native_score={"prior": 1.0},
    )
    # Audio perception CONTRADICTS the claim and supports another recording.
    ev_audio = repo.record_evidence(
        claim=claim,
        source_family="mert_identity_head",
        direction=EvidenceDirection.CONTRADICTS,
        run=run,
        native_score={"margin": 0.7},
    )
    # Belief weighs them: audio wins -> chosen != claimed (override, provenanced).
    belief = repo.record_belief(
        subject=slot,
        axis=Axis.IDENTITY,
        posterior={"rec_claimed": 0.2, "rec_audio": 0.8},
        rule=ID_RULE,
        run=run,
        contributing_evidence_ids=[ev_tok.evidence_id, ev_audio.evidence_id],
    )
    assert belief.decision == Decision.ACCEPTED
    assert belief.chosen == "rec_audio"  # NOT the tokenizer's claim
    assert set(belief.contributing_evidence_ids) == {
        ev_tok.evidence_id,
        ev_audio.evidence_id,
    }


def test_belief_abstains_when_pool_ambiguous(repo_stack):
    store, repo = repo_stack
    run = repo.begin_run(
        process="identity.infer",
        process_version="1",
        code_commit="abc",
        params_hash="0",
    )
    from core.provenance import SubjectRef

    slot = SubjectRef("set-slot", "s1")
    belief = repo.record_belief(
        subject=slot,
        axis=Axis.IDENTITY,
        posterior={"a": 0.5, "b": 0.5},
        rule=ID_RULE,
        run=run,
    )
    assert belief.decision == Decision.UNRESOLVED
    assert belief.chosen is None  # abstain persisted, not a coin-flip guess
    row = store._conn.execute(
        "SELECT decision, chosen FROM belief WHERE belief_id=?", (belief.belief_id,)
    ).fetchone()
    assert row["decision"] == Decision.UNRESOLVED.value and row["chosen"] is None


def test_axes_are_decided_independently(repo_stack):
    # Same subject, two axes -> two separate beliefs (Law 10).
    store, repo = repo_stack
    run = repo.begin_run(
        process="infer", process_version="1", code_commit="abc", params_hash="0"
    )
    from core.provenance import SubjectRef

    slot = SubjectRef("set-slot", "s1")
    b_id = repo.record_belief(
        subject=slot, axis=Axis.IDENTITY, posterior={"a": 0.9}, rule=ID_RULE, run=run
    )
    place_rule = DecisionRule("pl", Axis.PLACEMENT, "1", 0.6, 0.9)
    b_pl = repo.record_belief(
        subject=slot,
        axis=Axis.PLACEMENT,
        posterior={"t=120": 0.95},
        rule=place_rule,
        run=run,
    )
    assert b_id.axis == Axis.IDENTITY and b_pl.axis == Axis.PLACEMENT
    n = store._conn.execute(
        "SELECT COUNT(DISTINCT axis) c FROM belief WHERE subject_id='s1'"
    ).fetchone()["c"]
    assert n == 2
