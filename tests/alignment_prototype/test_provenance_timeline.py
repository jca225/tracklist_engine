"""Brick 3 — the predicted timeline as a provenanced artifact (Laws 1/3/10/12).

The load-bearing behaviors: the timeline gets a content-addressed artifact + a
derivation edge (no more bare file); placement and structure are SEPARATE per-axis
records — and share the one subject the identity belief uses; and placement keeps
its competing probe proposals + an uncalibrated flag rather than faking a
posterior (Law 12 debt stays visible).
"""

from __future__ import annotations

import json

import pytest

from core.provenance import (
    ArtifactKind,
    ArtifactStore,
    Axis,
    DecisionRule,
    ObservationStatus,
    ProvenanceRepository,
    connect,
)
from workspaces.alignment_prototype.provenance_export import (
    SlotClaim,
    export_identity_beliefs,
)
from workspaces.alignment_prototype.provenance_timeline import register_timeline


def _timeline(tmp_path, spans, **top):
    doc = {"set_id": "1fsnxchk", "train_set_id": "2nvzlh2k", "spans": spans, **top}
    p = tmp_path / "tl.json"
    p.write_text(json.dumps(doc))
    return p


@pytest.fixture
def repo_stack(tmp_path):
    conn = connect(tmp_path)
    return ArtifactStore(tmp_path, conn), ProvenanceRepository(conn)


SPANS = [
    {  # probes disagree by ~300s -> uncalibrated priority decode; has structure
        "slot_label": "1",
        "set_start_s": 34.46,
        "start_source": "mert",
        "confidence": 0.7,
        "probe_proposals": {"mert_decode": 34.46, "fp": 337.8},
        "ref_segments": [
            {"mix_start_s": 34.46, "ref_start_s": 103.9, "ref_end_s": 126.9}
        ],
        "ref_decode_status": "ok",
    },
    {  # single probe (agrees trivially); NO structure decoded
        "slot_label": "2w1",
        "set_start_s": 120.0,
        "start_source": "fp",
        "confidence": 0.9,
        "probe_proposals": {"fp": 120.0},
        "ref_segments": [],
        "ref_decode_status": "abstain",
    },
]


def test_timeline_is_content_addressed_with_derivation_edge(repo_stack, tmp_path):
    store, repo = repo_stack
    reg = register_timeline(
        "1fsnxchk", _timeline(tmp_path, SPANS), store=store, repo=repo
    )

    c = store._conn
    # Law 3: the timeline is an artifact addressed by its sha (a real 64-hex).
    assert len(reg.timeline_sha256) == 64
    art = c.execute(
        "SELECT kind FROM artifact WHERE content_sha256=?", (reg.timeline_sha256,)
    ).fetchone()
    assert art["kind"] == ArtifactKind.PREDICTED_TIMELINE.value
    # Law 1: a RunOutput edge + a derivation edge to the run-config parent exist.
    assert (
        c.execute(
            "SELECT COUNT(*) n FROM run_output WHERE content_sha256=?",
            (reg.timeline_sha256,),
        ).fetchone()["n"]
        == 1
    )
    d = c.execute(
        "SELECT relation FROM derivation WHERE child_sha256=?", (reg.timeline_sha256,)
    ).fetchone()
    assert d["relation"] == "registered_from"


def test_placement_and_structure_are_separate_axes(repo_stack, tmp_path):
    store, repo = repo_stack
    register_timeline("1fsnxchk", _timeline(tmp_path, SPANS), store=store, repo=repo)
    c = store._conn
    preds = {
        r["predicate"]
        for r in c.execute("SELECT DISTINCT predicate FROM observation").fetchall()
    }
    assert {"set_start_s", "ref_trajectory"} <= preds  # two distinct axes, not one


def test_uncalibrated_placement_is_flagged_not_faked(repo_stack, tmp_path):
    # The disagreeing-probe span must keep BOTH proposals and be flagged; the
    # substrate must NOT invent a posterior (Law 12 honesty).
    store, repo = repo_stack
    reg = register_timeline(
        "1fsnxchk", _timeline(tmp_path, SPANS), store=store, repo=repo
    )
    assert reg.n_uncalibrated_placement == 1
    c = store._conn
    row = c.execute(
        "SELECT value_json, diagnostic_code FROM observation "
        "WHERE predicate='set_start_s' AND diagnostic_code IS NOT NULL"
    ).fetchone()
    assert row["diagnostic_code"] == "uncalibrated_priority_decode"
    val = json.loads(row["value_json"])
    assert set(val["proposals"]) == {"mert_decode", "fp"}  # competitors preserved
    # No belief was written for placement — it is an observation, not a fake belief.
    assert (
        c.execute(
            "SELECT COUNT(*) n FROM belief WHERE axis=?", (Axis.PLACEMENT.value,)
        ).fetchone()["n"]
        == 0
    )


def test_missing_structure_abstains(repo_stack, tmp_path):
    store, repo = repo_stack
    reg = register_timeline(
        "1fsnxchk", _timeline(tmp_path, SPANS), store=store, repo=repo
    )
    assert reg.n_no_structure == 1
    c = store._conn
    row = c.execute(
        "SELECT status, diagnostic_code FROM observation "
        "WHERE predicate='ref_trajectory' AND diagnostic_code IS NOT NULL"
    ).fetchone()
    assert row["status"] == ObservationStatus.ABSTAINED.value
    assert row["diagnostic_code"] == "no_structure_decoded"


def test_three_axes_hang_off_one_subject(repo_stack, tmp_path):
    # The crown of Law 10: identity (belief) + placement + structure (observations)
    # all key off the SAME set-slot subject, produced by different runs.
    store, repo = repo_stack
    source = store.put_bytes(
        b"<tl/>", kind=ArtifactKind.HTML_PAGE, media_type="text/html"
    )
    rule = DecisionRule("id-v1", Axis.IDENTITY, "1", 0.6, 0.9)
    export_identity_beliefs(
        "1fsnxchk",
        [SlotClaim("1", "rec_alpha", "regular")],
        store=store,
        repo=repo,
        source=source,
        rule=rule,
    )
    register_timeline("1fsnxchk", _timeline(tmp_path, SPANS), store=store, repo=repo)

    subject_id = "1fsnxchk::1"
    c = store._conn
    has_identity = c.execute(
        "SELECT COUNT(*) n FROM belief WHERE subject_id=? AND axis=?",
        (subject_id, Axis.IDENTITY.value),
    ).fetchone()["n"]
    has_place = c.execute(
        "SELECT COUNT(*) n FROM observation WHERE subject_id=? AND predicate='set_start_s'",
        (subject_id,),
    ).fetchone()["n"]
    has_struct = c.execute(
        "SELECT COUNT(*) n FROM observation WHERE subject_id=? AND predicate='ref_trajectory'",
        (subject_id,),
    ).fetchone()["n"]
    assert has_identity == 1 and has_place == 1 and has_struct == 1
