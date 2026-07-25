"""Brick 7 — §2/§8 versioning backbone: deterministic builders + writers.

The load-bearing property: identity is CONTENT. The same parameters /
environment / process / training set always build to the same id, so the
writers' INSERT OR IGNORE makes recording idempotent — one row per distinct
entity, never duplicates, never overwrites.
"""

from __future__ import annotations

import pytest

from core.provenance import (
    ArtifactKind,
    ArtifactStore,
    Axis,
    ProvenanceRepository,
    canonical_values_hash,
    capture_environment,
    connect,
    make_fitted_model,
    make_parameter_set,
    make_process_spec,
    make_training_snapshot,
)


@pytest.fixture
def world(tmp_path):
    conn = connect(tmp_path)
    store = ArtifactStore(tmp_path, conn)
    repo = ProvenanceRepository(conn)
    return conn, store, repo


def test_canonical_hash_ignores_key_order():
    assert canonical_values_hash({"a": 1, "b": 2}) == canonical_values_hash(
        {"b": 2, "a": 1}
    )
    assert canonical_values_hash({"a": 1}) != canonical_values_hash({"a": 2})


def test_parameter_set_is_deterministic():
    a = make_parameter_set("analyze.x", {"sr": 22050, "fan": 8})
    b = make_parameter_set("analyze.x", {"fan": 8, "sr": 22050})
    assert a.parameter_set_id == b.parameter_set_id
    assert a.canonical_hash == b.canonical_hash
    # A different namespace over the same values is a DIFFERENT parameter set.
    c = make_parameter_set("analyze.y", {"sr": 22050, "fan": 8})
    assert c.parameter_set_id != a.parameter_set_id


def test_capture_environment_from_requirements_file(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("numpy==1.0\n")
    env = capture_environment(req)
    assert env.os and env.architecture and env.runtime.startswith("python-")
    assert env.dependency_lock_hash
    # Same lock -> same id; different lock -> different id.
    assert capture_environment(req).environment_spec_id == env.environment_spec_id
    req.write_text("numpy==2.0\n")
    assert capture_environment(req).environment_spec_id != env.environment_spec_id


def test_capture_environment_without_file_uses_installed_lock():
    env = capture_environment()
    assert len(env.dependency_lock_hash) == 64


def test_process_spec_id_is_deterministic_and_idempotent(world, tmp_path):
    conn, _, repo = world
    req = tmp_path / "r.txt"
    req.write_text("x==1\n")
    ps = repo.record_parameter_set(make_parameter_set("p", {"k": 1}))
    env = repo.record_environment_spec(capture_environment(req))
    build = lambda: make_process_spec(  # noqa: E731
        name="analyze.thing",
        version="1",
        stage="analysis",
        code_commit="abc1234",
        parameter_set=ps,
        environment=env,
        model_refs=("m-a-p/MERT-v1-330M",),
    )
    s1, s2 = build(), build()
    assert s1.process_spec_id == s2.process_spec_id  # identical process = same id
    repo.record_process_spec(s1)
    repo.record_process_spec(s2)  # INSERT OR IGNORE — no duplicate
    n = conn.execute("SELECT COUNT(*) FROM process_spec").fetchone()[0]
    assert n == 1
    # Any defining field changes the id.
    s3 = make_process_spec(
        name="analyze.thing",
        version="2",
        stage="analysis",
        code_commit="abc1234",
        parameter_set=ps,
        environment=env,
        model_refs=("m-a-p/MERT-v1-330M",),
    )
    assert s3.process_spec_id != s1.process_spec_id


def test_training_snapshot_hash_is_order_free():
    a = make_training_snapshot(["sha_b", "sha_a"])
    b = make_training_snapshot(["sha_a", "sha_b", "sha_a"])
    assert a.snapshot_hash == b.snapshot_hash
    assert a.training_snapshot_id == b.training_snapshot_id
    assert a.member_artifact_shas == ("sha_a", "sha_b")


def test_begin_run_from_spec_stamps_versioning(world, tmp_path):
    conn, _, repo = world
    req = tmp_path / "r.txt"
    req.write_text("x==1\n")
    ps = repo.record_parameter_set(make_parameter_set("p", {"k": 1}))
    env = repo.record_environment_spec(capture_environment(req))
    spec = repo.record_process_spec(
        make_process_spec(
            name="analyze.thing",
            version="1",
            stage="analysis",
            code_commit="abc1234",
            parameter_set=ps,
            environment=env,
        )
    )
    run = repo.begin_run_from_spec(spec, seed=7)
    assert run.process == "analyze.thing"
    assert run.process_version == "1"
    assert run.code_commit == "abc1234"
    assert run.params_hash == ps.canonical_hash
    assert run.process_spec_id == spec.process_spec_id
    row = conn.execute(
        "SELECT process_spec_id, params_hash, random_seed FROM run WHERE run_id=?",
        (run.run_id,),
    ).fetchone()
    assert row["process_spec_id"] == spec.process_spec_id
    assert row["params_hash"] == ps.canonical_hash
    assert row["random_seed"] == 7
    repo.succeed(run)


def test_begin_run_from_spec_requires_recorded_parameter_set(world, tmp_path):
    _, _, repo = world
    req = tmp_path / "r.txt"
    req.write_text("x==1\n")
    ps = make_parameter_set("p", {"k": 1})  # NOT recorded
    env = repo.record_environment_spec(capture_environment(req))
    spec = make_process_spec(
        name="analyze.thing",
        version="1",
        stage="analysis",
        code_commit="abc1234",
        parameter_set=ps,
        environment=env,
    )
    with pytest.raises(ValueError, match="unrecorded parameter_set"):
        repo.begin_run_from_spec(spec)


def test_fitted_model_roundtrip_and_idempotency(world, tmp_path):
    conn, store, repo = world
    req = tmp_path / "r.txt"
    req.write_text("x==1\n")
    state = store.put_bytes(
        b"weights",
        kind=ArtifactKind.MODEL_CHECKPOINT,
        media_type="application/octet-stream",
        source_system="test",
    )
    gt = store.put_bytes(
        b"gt",
        kind=ArtifactKind.LABEL_MANIFEST,
        media_type="text/yaml",
        source_system="test",
    )
    ps = repo.record_parameter_set(make_parameter_set("train", {"epochs": 2}))
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
    snap = repo.record_training_snapshot(make_training_snapshot([gt.content_sha256]))
    model = make_fitted_model(
        axis=Axis.IDENTITY,
        process_spec=spec,
        serialized_state_sha256=state.content_sha256,
        training_snapshot=snap,
    )
    repo.record_fitted_model(model)
    repo.record_fitted_model(model)  # idempotent
    rows = conn.execute("SELECT * FROM fitted_model").fetchall()
    assert len(rows) == 1
    assert rows[0]["axis"] == "identity"
    assert rows[0]["model_state_hash"] == state.content_sha256
    assert rows[0]["status"] == "candidate"
