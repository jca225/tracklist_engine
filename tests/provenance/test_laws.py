"""Brick 5 — the §16 laws as executable predicates over a substrate DB.

Two shapes of test: a clean provenanced world PASSES every checkable law (and
the unbuilt-entity laws honestly report NOT_APPLICABLE, never fake green); and
each checkable law is violated by exactly the corruption it exists to catch —
mostly injected via direct SQL, i.e. bypassing the writers, which is precisely
the case a checker must catch.
"""

from __future__ import annotations

import pytest

from core.provenance import (
    ArtifactKind,
    ArtifactStore,
    Axis,
    DecisionRule,
    EvidenceDirection,
    LawResult,
    LawVerdict,
    ObservationStatus,
    ProvenanceRepository,
    SubjectRef,
    check_laws,
    connect,
    format_law_results,
)

ID_RULE = DecisionRule("id-v1", Axis.IDENTITY, "1", 0.6, 0.9)
SUBJ = SubjectRef("set-slot", "1fsnxchk::1")


@pytest.fixture
def world(tmp_path):
    conn = connect(tmp_path)
    store = ArtifactStore(tmp_path, conn)
    repo = ProvenanceRepository(conn)
    return conn, store, repo


def _seed(store, repo):
    """A small honest world: source -> derived artifact, an accepted and an
    abstaining belief, an observed and an abstained observation."""
    run = repo.begin_run(
        process="test.seed", process_version="1", code_commit="c", params_hash="0"
    )
    src = store.put_bytes(
        b"<row/>",
        kind=ArtifactKind.HTML_ROW,
        media_type="text/html",
        source_system="test",
    )
    derived = store.put_bytes(
        b"{}", kind=ArtifactKind.PREDICTED_TIMELINE, media_type="application/json"
    )
    repo.add_output(run, derived, role="timeline")
    repo.derive(derived, src, run, "registered_from")

    repo.record_observation(
        subject=SUBJ, predicate="claimed_recording", value="rec_a", source=src, run=run
    )
    repo.record_observation(
        subject=SUBJ,
        predicate="source_track_key",
        value=None,
        source=src,
        run=run,
        status=ObservationStatus.ABSTAINED,
        diagnostic_code="no_trackid",
    )
    claim = repo.record_claim(
        axis=Axis.IDENTITY,
        subject=SUBJ,
        predicate="is_recording",
        candidate=SubjectRef("recording", "rec_a"),
        run=run,
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
        posterior={"rec_a": 0.95},
        rule=ID_RULE,
        run=run,
        contributing_evidence_ids=[ev.evidence_id],
    )
    repo.record_belief(
        subject=SubjectRef("set-slot", "1fsnxchk::2"),
        axis=Axis.IDENTITY,
        posterior={"a": 0.5, "b": 0.5},  # ambiguous -> persisted abstention
        rule=ID_RULE,
        run=run,
    )
    repo.succeed(run)
    return run, src, derived


def _by_number(results: list[LawResult]) -> dict[int, LawResult]:
    return {r.number: r for r in results}


# --- shape + clean world -----------------------------------------------------
def test_results_cover_all_21_laws_in_order(world):
    conn, store, _ = world
    results = check_laws(conn, store)
    assert [r.number for r in results] == list(range(1, 22))


def test_clean_world_passes_every_checkable_law(world):
    conn, store, repo = world
    _seed(store, repo)
    by = _by_number(check_laws(conn, store))
    assert not any(r.verdict is LawVerdict.VIOLATED for r in by.values())
    for n in (1, 2, 3, 4, 5, 6, 7, 10, 11, 21):
        assert by[n].verdict is LawVerdict.PASS, (n, by[n].reason)
        assert by[n].checked > 0


def test_unbuilt_entity_laws_are_not_applicable_never_faked(world):
    conn, store, repo = world
    _seed(store, repo)
    by = _by_number(check_laws(conn, store))
    # 8/9 (§7) and 13/14 (§8) are checkable but have no rows in this seed;
    # 12/15–20 have no substrate entities at all. Either way: N/A, never green.
    for n in (8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20):
        assert by[n].verdict is LawVerdict.NOT_APPLICABLE, n


def test_empty_db_is_not_applicable_not_pass(world):
    conn, store, _ = world
    by = _by_number(check_laws(conn, store))
    assert not any(r.verdict is LawVerdict.VIOLATED for r in by.values())
    # An empty DB proves nothing — data-dependent laws must not glow green.
    for n in (1, 2, 3, 4, 5, 6, 7, 10, 11, 21):
        assert by[n].verdict is LawVerdict.NOT_APPLICABLE, n


# --- one corruption per checkable law ---------------------------------------
def test_orphan_artifact_violates_law_1(world):
    conn, store, repo = world
    _seed(store, repo)
    conn.execute(
        "INSERT INTO artifact(content_sha256, kind, media_type, byte_size, "
        "object_uri, created_at) VALUES ('ff'||substr(hex(randomblob(31)),1,62), "
        "'feature_blob', 'application/octet-stream', 1, 'nowhere', 'now')"
    )
    by = _by_number(check_laws(conn, store))
    assert by[1].verdict is LawVerdict.VIOLATED
    assert "no source and no producing run" in by[1].offenders[0]


def test_unfinished_run_violates_law_1(world):
    conn, store, repo = world
    _seed(store, repo)
    repo.begin_run(
        process="crashed", process_version="1", code_commit="c", params_hash="0"
    )  # never finished
    by = _by_number(check_laws(conn, store))
    assert by[1].verdict is LawVerdict.VIOLATED
    assert any("never finished" in o for o in by[1].offenders)


def test_cycle_injected_via_sql_violates_law_2(world):
    conn, store, repo = world
    run, src, derived = _seed(store, repo)
    # The writer's guard refuses this edge; inject it directly (the exact case
    # a checker exists for: a DB the writers did not build).
    conn.execute(
        "INSERT INTO derivation(child_sha256, parent_sha256, run_id, relation) "
        "VALUES (?,?,?,?)",
        (src.content_sha256, derived.content_sha256, run.run_id, "evil"),
    )
    by = _by_number(check_laws(conn, store))
    assert by[2].verdict is LawVerdict.VIOLATED
    assert "cycle" in by[2].offenders[0]


def test_tampered_object_violates_law_3(world):
    conn, store, repo = world
    _, src, _ = _seed(store, repo)
    store._object_path(src.content_sha256).write_bytes(b"tampered")
    by = _by_number(check_laws(conn, store))
    assert by[3].verdict is LawVerdict.VIOLATED
    assert src.content_sha256[:12] in by[3].offenders[0]


def test_law_3_without_store_is_not_applicable(world):
    conn, store, repo = world
    _seed(store, repo)
    by = _by_number(check_laws(conn, store=None))
    assert by[3].verdict is LawVerdict.NOT_APPLICABLE


def test_placeholder_identity_violates_law_4(world):
    conn, store, repo = world
    run, _, _ = _seed(store, repo)
    # The writers do not forbid a tlp* posterior key — the checker must catch it.
    repo.record_belief(
        subject=SubjectRef("set-slot", "1fsnxchk::9"),
        axis=Axis.IDENTITY,
        posterior={"tlp2594024": 1.0},
        rule=ID_RULE,
        run=run,
    )
    by = _by_number(check_laws(conn, store))
    assert by[4].verdict is LawVerdict.VIOLATED
    assert "tlp2594024" in by[4].offenders[0]


def test_pathlike_identity_violates_law_5(world):
    conn, store, repo = world
    run, _, _ = _seed(store, repo)
    repo.record_claim(
        axis=Axis.IDENTITY,
        subject=SubjectRef("set-slot", "1fsnxchk::9"),
        predicate="is_recording",
        candidate=SubjectRef("recording", "/mnt/storage/objects/x/y.mp3"),
        run=run,
    )
    by = _by_number(check_laws(conn, store))
    assert by[5].verdict is LawVerdict.VIOLATED
    assert "path-like" in by[5].offenders[0]


def test_abstained_observation_without_diagnostic_violates_law_6(world):
    conn, store, repo = world
    run, src, _ = _seed(store, repo)
    repo.record_observation(
        subject=SUBJ,
        predicate="claimed_stem",
        value=None,
        source=src,
        run=run,
        status=ObservationStatus.ABSTAINED,
        diagnostic_code=None,  # writer allows it; the law does not
    )
    by = _by_number(check_laws(conn, store))
    assert by[6].verdict is LawVerdict.VIOLATED
    assert "no diagnostic_code" in by[6].offenders[0]


def test_smuggled_choice_on_abstention_violates_law_7(world):
    conn, store, repo = world
    _seed(store, repo)
    conn.execute("UPDATE belief SET chosen='sneaky' WHERE decision='unresolved'")
    by = _by_number(check_laws(conn, store))
    assert by[7].verdict is LawVerdict.VIOLATED
    assert "sneaky" in by[7].offenders[0]


def test_cross_axis_evidence_violates_law_10(world):
    conn, store, repo = world
    _seed(store, repo)
    conn.execute("UPDATE evidence SET axis='placement'")  # claim stays identity
    by = _by_number(check_laws(conn, store))
    assert by[10].verdict is LawVerdict.VIOLATED
    assert any("fused onto" in o for o in by[10].offenders)


def test_numeric_coordinate_on_abstained_observation_violates_law_11(world):
    conn, store, repo = world
    run, src, _ = _seed(store, repo)
    repo.record_observation(
        subject=SUBJ,
        predicate="set_start_s",
        value={"chosen_s": 0.0},  # unknown encoded as 0.0 — the exact lie
        source=src,
        run=run,
        status=ObservationStatus.ABSTAINED,
        diagnostic_code="gated",
    )
    by = _by_number(check_laws(conn, store))
    assert by[11].verdict is LawVerdict.VIOLATED
    assert "chosen_s" in by[11].offenders[0]


def test_dangling_evidence_reference_violates_law_21(world):
    conn, store, repo = world
    _seed(store, repo)
    conn.execute(
        "UPDATE belief SET contributing_evidence_ids_json='[\"no-such-ev\"]' "
        "WHERE chosen='rec_a'"
    )
    by = _by_number(check_laws(conn, store))
    assert by[21].verdict is LawVerdict.VIOLATED
    assert any("missing evidence" in o for o in by[21].offenders)


# --- §8 model provenance (laws 13/14, Brick 7) -------------------------------
def _seed_model(store, repo, tmp_path):
    """A fully-versioned fitted model: spec'd training run, frozen snapshot,
    content-addressed state — the world laws 13/14 must PASS on."""
    from core.provenance import (
        Axis,
        capture_environment,
        make_fitted_model,
        make_parameter_set,
        make_process_spec,
        make_training_snapshot,
    )

    req = tmp_path / "req.txt"
    req.write_text("torch==0\n")
    gt = store.put_bytes(
        b"gt-yaml",
        kind=ArtifactKind.LABEL_MANIFEST,
        media_type="text/yaml",
        source_system="test",
    )
    state = store.put_bytes(
        b"weights",
        kind=ArtifactKind.MODEL_CHECKPOINT,
        media_type="application/octet-stream",
        source_system="test",
    )
    ps = repo.record_parameter_set(make_parameter_set("train.head", {"epochs": 1}))
    env = repo.record_environment_spec(capture_environment(req))
    spec = repo.record_process_spec(
        make_process_spec(
            name="train.head",
            version="1",
            stage="training",
            code_commit="abc1234",
            parameter_set=ps,
            environment=env,
        )
    )
    run = repo.begin_run_from_spec(spec)
    repo.add_input(run, gt, role="training_member")
    repo.add_output(run, state, role="model_state")
    repo.derive(state, gt, run, "trained_on")
    repo.succeed(run)
    snap = repo.record_training_snapshot(make_training_snapshot([gt.content_sha256]))
    model = repo.record_fitted_model(
        make_fitted_model(
            axis=Axis.IDENTITY,
            process_spec=spec,
            serialized_state_sha256=state.content_sha256,
            training_snapshot=snap,
        )
    )
    return model, spec, snap


def test_versioned_model_passes_laws_13_and_14(world, tmp_path):
    conn, store, repo = world
    _seed(store, repo)
    _seed_model(store, repo, tmp_path)
    by = _by_number(check_laws(conn, store))
    assert by[13].verdict is LawVerdict.PASS, by[13].reason
    assert by[14].verdict is LawVerdict.PASS, by[14].reason
    assert by[13].checked > 0 and by[14].checked > 0
    # And the versioned world stays clean on every other law.
    assert not any(r.verdict is LawVerdict.VIOLATED for r in by.values())


def test_dangling_training_snapshot_violates_law_13(world, tmp_path):
    conn, store, repo = world
    _seed(store, repo)
    _seed_model(store, repo, tmp_path)
    # History rewritten out from under the model — the exact §8 violation.
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("DELETE FROM training_snapshot")
    by = _by_number(check_laws(conn, store))
    assert by[13].verdict is LawVerdict.VIOLATED
    assert "missing" in by[13].offenders[0]


def test_null_training_snapshot_violates_law_13(world, tmp_path):
    conn, store, repo = world
    _seed(store, repo)
    _seed_model(store, repo, tmp_path)
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("UPDATE fitted_model SET training_snapshot_id=''")
    by = _by_number(check_laws(conn, store))
    assert by[13].verdict is LawVerdict.VIOLATED
    assert "no training snapshot" in by[13].offenders[0]


def test_process_spec_missing_environment_violates_law_14(world, tmp_path):
    conn, store, repo = world
    _seed(store, repo)
    _seed_model(store, repo, tmp_path)
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("DELETE FROM environment_spec")
    by = _by_number(check_laws(conn, store))
    assert by[14].verdict is LawVerdict.VIOLATED
    assert any("missing environment_spec" in o for o in by[14].offenders)


def test_model_with_empty_code_commit_violates_law_14(world, tmp_path):
    conn, store, repo = world
    _seed(store, repo)
    _seed_model(store, repo, tmp_path)
    conn.execute("UPDATE process_spec SET code_commit=''")
    by = _by_number(check_laws(conn, store))
    assert by[14].verdict is LawVerdict.VIOLATED
    assert any("no code_commit" in o for o in by[14].offenders)


def test_registered_kind_artifact_without_spec_run_violates_law_14(world):
    conn, store, repo = world
    run, src, _ = _seed(store, repo)
    # A registered (non-enum) feature kind written through the LEGACY inline
    # run path — provenanced but not ProcessSpec-versioned. Law 14 must flag it.
    feat = store.put_bytes(
        b"feature-bytes", kind="rogue_registered_kind", media_type="application/json"
    )
    repo.add_output(run, feat, role="features")
    repo.derive(feat, src, run, "analyzed_from")
    by = _by_number(check_laws(conn, store))
    assert by[14].verdict is LawVerdict.VIOLATED
    assert any("rogue_registered_kind" in o for o in by[14].offenders)


# --- rendering ---------------------------------------------------------------
def test_format_reports_scope_and_tally(world):
    conn, store, repo = world
    _seed(store, repo)
    text = format_law_results(check_laws(conn, store))
    assert "substrate DB scope only" in text  # the honesty banner
    assert "law_audit.md" in text
    assert "tally:" in text
    assert "provenance_is_acyclic" in text
