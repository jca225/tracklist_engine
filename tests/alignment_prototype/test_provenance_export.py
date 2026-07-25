"""Brick 2b — the aligner's identity expressed through the provenance substrate."""

from __future__ import annotations

import pytest

from core.provenance import (
    ArtifactKind,
    ArtifactStore,
    Axis,
    Decision,
    DecisionRule,
    ObservationStatus,
    ProvenanceRepository,
    connect,
)
from workspaces.alignment_prototype.provenance_export import (
    SlotClaim,
    _claims_from_rows,
    export_identity_beliefs,
)

ID_RULE = DecisionRule(
    "id-v1", Axis.IDENTITY, "1", minimum_posterior=0.6, maximum_entropy=0.9
)


def test_tlp_placeholder_row_maps_to_unresolved_claim():
    # A `tlp*` recording_id is a sided-row placeholder (no data-trackid), not a
    # canonical recording. The runner must NOT carry it forward as an identity —
    # it becomes an unresolved claim so the belief abstains. A canonical id and a
    # bare id (no recording) round-trip as themselves / None.
    rows = [
        {"slot_label": "1", "recording_id": "tlp2594024", "claimed_stem": "regular"},
        {"slot_label": "2", "recording_id": "g8gtgdx", "claimed_stem": "instrumental"},
        {"slot_label": "3", "recording_id": None, "claimed_stem": "acappella"},
    ]
    claims = _claims_from_rows(rows)
    assert claims[0].claimed_recording_id is None  # placeholder scrubbed
    assert claims[1].claimed_recording_id == "g8gtgdx"  # canonical kept
    assert claims[1].claimed_stem == "instrumental"
    assert claims[2].claimed_recording_id is None  # already unidentified


@pytest.fixture
def env(tmp_path):
    conn = connect(tmp_path)
    store = ArtifactStore(tmp_path, conn)
    repo = ProvenanceRepository(conn)
    source = store.put_bytes(
        b"<tracklist/>", kind=ArtifactKind.HTML_PAGE, media_type="text/html"
    )
    return store, repo, source


# A realistic spine: two identified slots + one unidentified ("ID") slot.
SPINE = [
    SlotClaim("013w1", "rec_alpha", "regular"),
    SlotClaim("014w1", "rec_beta", "acappella"),
    SlotClaim("015w1", None, "regular"),  # an ID — no claimed recording
]


def test_current_pipeline_accepts_claim_and_abstains_on_id(env):
    store, repo, source = env
    beliefs = export_identity_beliefs(
        "1fsnxchk", SPINE, store=store, repo=repo, source=source, rule=ID_RULE
    )
    by_slot = {b.subject.subject_id.split("::")[1]: b for b in beliefs}

    # Size-1 pool -> belief accepts the claim (decision #19, made explicit).
    assert by_slot["013w1"].decision == Decision.ACCEPTED
    assert by_slot["013w1"].chosen == "rec_alpha"
    # The ID slot abstains — never a guessed recording.
    assert by_slot["015w1"].decision == Decision.UNRESOLVED
    assert by_slot["015w1"].chosen is None


def test_id_slot_records_persisted_abstention_observation(env):
    store, repo, source = env
    export_identity_beliefs(
        "1fsnxchk", SPINE, store=store, repo=repo, source=source, rule=ID_RULE
    )
    row = store._conn.execute(
        "SELECT status, diagnostic_code FROM observation WHERE subject_id LIKE '%015w1'"
    ).fetchone()
    assert row["status"] == ObservationStatus.ABSTAINED.value
    assert row["diagnostic_code"] == "unidentified_slot"


def test_chain_is_fully_provenanced(env):
    store, repo, source = env
    export_identity_beliefs(
        "1fsnxchk", SPINE, store=store, repo=repo, source=source, rule=ID_RULE
    )
    c = store._conn
    # one claim + one evidence per slot; three beliefs; the run succeeded.
    assert c.execute("SELECT COUNT(*) n FROM claim").fetchone()["n"] == 3
    assert c.execute("SELECT COUNT(*) n FROM evidence").fetchone()["n"] == 3
    assert c.execute("SELECT COUNT(*) n FROM belief").fetchone()["n"] == 3
    assert c.execute("SELECT status FROM run").fetchone()["status"] == "succeeded"
    # every belief cites its evidence (Law 21 groundwork).
    b = c.execute(
        "SELECT contributing_evidence_ids_json FROM belief LIMIT 1"
    ).fetchone()
    assert b["contributing_evidence_ids_json"] != "[]"


def test_audio_evidence_source_can_override_claim(env):
    # The parked open_set_identity, once wired, supplies a posterior that
    # contradicts the claim; the belief then overrides — provenanced.
    store, repo, source = env
    beliefs = export_identity_beliefs(
        "1fsnxchk",
        [SlotClaim("013w1", "rec_alpha", "regular")],
        store=store,
        repo=repo,
        source=source,
        rule=ID_RULE,
        posterior_by_slot={"013w1": {"rec_alpha": 0.15, "rec_audio_says": 0.85}},
    )
    assert beliefs[0].decision == Decision.ACCEPTED
    assert beliefs[0].chosen == "rec_audio_says"  # overrode the tokenizer claim
