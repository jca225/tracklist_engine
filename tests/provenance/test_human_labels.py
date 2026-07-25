"""Brick 6 — §7 human-label provenance (Laws 8/9 made checkable).

Three shapes of test: the append-only writers (a revision is a NEW row, the old
one survives; a model-less assertion is refused at the writer); the laws 8/9
predicates on a clean world PASS (not N/A) once assertions exist; and each law
is violated by exactly the corruption it exists to catch — injected via direct
SQL, i.e. bypassing the writers, which is precisely the case a checker must
catch.
"""

from __future__ import annotations

import pytest

from core.provenance import (
    ArtifactKind,
    ArtifactStore,
    LawVerdict,
    ProvenanceRepository,
    SubjectRef,
    check_laws,
    connect,
    explain_subject,
)

SUBJ = SubjectRef("set-slot", "1fsnxchk::013")

CATEGORICAL = {"kind": "categorical_prior", "mean": 0.99, "effective_sample_size": 20}
STUDENT_T = {"kind": "student_t", "scale_s": 0.25, "df": 4}


@pytest.fixture
def world(tmp_path):
    conn = connect(tmp_path)
    store = ArtifactStore(tmp_path, conn)
    repo = ProvenanceRepository(conn)
    return conn, store, repo


def _seed_gt(store, repo):
    """A small honest §7 world: one bundle, two fields with DIFFERENT models."""
    run = repo.begin_run(
        process="human_gt.import", process_version="1", code_commit="c", params_hash="0"
    )
    src = store.put_bytes(
        b"set_id: 1fsnxchk\n",
        kind=ArtifactKind.LABEL_MANIFEST,
        media_type="application/x-yaml",
        source_system="labeling.ground_truth",
    )
    bundle = repo.record_human_bundle(
        set_id="1fsnxchk", source=src, annotator_id="user", run=run
    )
    a_id = repo.record_human_assertion(
        bundle=bundle,
        subject=SUBJ,
        field="recording_id",
        observed_value="281u6p4x",
        uncertainty_model=CATEGORICAL,
        run=run,
    )
    a_start = repo.record_human_assertion(
        bundle=bundle,
        subject=SUBJ,
        field="mix_start_s",
        observed_value=64.4327,
        uncertainty_model=STUDENT_T,
        run=run,
    )
    repo.succeed(run)
    return run, src, bundle, a_id, a_start


def _by_number(results):
    return {r.number: r for r in results}


# --- writers: append-only discipline -----------------------------------------
def test_revision_inserts_new_row_and_keeps_the_old(world):
    conn, store, repo = world
    run, _, _, _, a_start = _seed_gt(store, repo)
    run2 = repo.begin_run(
        process="human_gt.revise", process_version="1", code_commit="c", params_hash="0"
    )
    new = repo.revise_human_assertion(old=a_start, observed_value=64.51, run=run2)
    repo.succeed(run2)

    assert new.supersedes_assertion_id == a_start.assertion_id
    assert new.field == a_start.field and new.subject == a_start.subject
    # The OLD row still exists, unmutated — history is append-only (Law 8).
    old_row = conn.execute(
        "SELECT value_json FROM human_label_assertion WHERE assertion_id=?",
        (a_start.assertion_id,),
    ).fetchone()
    assert old_row is not None and "64.4327" in old_row["value_json"]
    assert (
        conn.execute("SELECT COUNT(*) n FROM human_label_assertion").fetchone()["n"]
        == 3
    )


def test_writer_refuses_assertion_without_uncertainty_model(world):
    _, store, repo = world
    run, src, bundle, _, _ = _seed_gt(store, repo)
    with pytest.raises(ValueError, match="Law 9"):
        repo.record_human_assertion(
            bundle=bundle,
            subject=SUBJ,
            field="tempo_ratio",
            observed_value=1.02,
            uncertainty_model={},
            run=run,
        )


# --- laws 8/9: PASS on a real world, N/A stays honest on empty ---------------
def test_laws_8_and_9_pass_not_na_once_assertions_exist(world):
    conn, store, repo = world
    _seed_gt(store, repo)
    by = _by_number(check_laws(conn, store))
    assert by[8].verdict is LawVerdict.PASS and by[8].checked > 0
    assert by[9].verdict is LawVerdict.PASS and by[9].checked > 0


def test_laws_8_and_9_stay_na_on_empty_db(world):
    conn, store, _ = world
    by = _by_number(check_laws(conn, store))
    assert by[8].verdict is LawVerdict.NOT_APPLICABLE
    assert by[9].verdict is LawVerdict.NOT_APPLICABLE


# --- one corruption per law ---------------------------------------------------
def test_deleted_superseded_row_violates_law_8(world):
    conn, store, repo = world
    run, _, _, _, a_start = _seed_gt(store, repo)
    repo.revise_human_assertion(old=a_start, observed_value=64.51, run=run)
    # Rewrite history behind the writers' back: delete the superseded row.
    conn.execute(
        "DELETE FROM human_label_assertion WHERE assertion_id=?",
        (a_start.assertion_id,),
    )
    by = _by_number(check_laws(conn, store))
    assert by[8].verdict is LawVerdict.VIOLATED
    assert any("no longer exists" in o for o in by[8].offenders)


def test_supersession_cycle_violates_law_8(world):
    conn, store, repo = world
    run, _, _, a_id, a_start = _seed_gt(store, repo)
    repo.revise_human_assertion(old=a_start, observed_value=64.51, run=run)
    # Inject a 2-cycle: the old row now claims to supersede its own reviser.
    new_id = conn.execute(
        "SELECT assertion_id FROM human_label_assertion "
        "WHERE supersedes_assertion_id=?",
        (a_start.assertion_id,),
    ).fetchone()["assertion_id"]
    conn.execute(
        "UPDATE human_label_assertion SET supersedes_assertion_id=? "
        "WHERE assertion_id=?",
        (new_id, a_start.assertion_id),
    )
    by = _by_number(check_laws(conn, store))
    assert by[8].verdict is LawVerdict.VIOLATED
    assert any("cycle" in o for o in by[8].offenders)


def test_nulled_uncertainty_violates_law_9(world):
    conn, store, repo = world
    _, _, _, a_id, _ = _seed_gt(store, repo)
    conn.execute(
        "UPDATE human_label_assertion SET uncertainty_model_json='{}' "
        "WHERE assertion_id=?",
        (a_id.assertion_id,),
    )
    by = _by_number(check_laws(conn, store))
    assert by[9].verdict is LawVerdict.VIOLATED
    assert any("no usable uncertainty_model" in o for o in by[9].offenders)


def test_one_global_model_across_fields_violates_law_9(world):
    conn, store, repo = world
    _seed_gt(store, repo)
    # Flatten the seeded per-field models into ONE global model — the exact
    # global-scalar-confidence smell the law forbids.
    conn.execute(
        "UPDATE human_label_assertion SET "
        'uncertainty_model_json=\'{"kind": "scalar", "sigma": 0.5}\''
    )
    by = _by_number(check_laws(conn, store))
    assert by[9].verdict is LawVerdict.VIOLATED
    assert any("uniformly" in o for o in by[9].offenders)


# --- read side ----------------------------------------------------------------
def test_explain_subject_surfaces_human_assertions(world):
    conn, store, repo = world
    run, _, _, _, a_start = _seed_gt(store, repo)
    repo.revise_human_assertion(old=a_start, observed_value=64.51, run=run)
    exp = explain_subject(conn, SUBJ.subject_id)
    assert not exp.is_empty
    starts = [h for h in exp.human_assertions if h.field == "mix_start_s"]
    assert len(starts) == 2
    superseded = next(h for h in starts if h.superseded)
    assert superseded.observed_value == pytest.approx(64.4327)
    live = next(h for h in starts if not h.superseded)
    assert live.observed_value == pytest.approx(64.51)
    rec = next(h for h in exp.human_assertions if h.field == "recording_id")
    assert rec.uncertainty_model["kind"] == "categorical_prior"
    assert rec.annotator_id == "user"
