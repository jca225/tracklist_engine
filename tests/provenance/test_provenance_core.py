"""Brick 1 substrate tests — the content-addressing + lineage laws, executable."""

from __future__ import annotations

import pytest

from core.provenance import (
    ArtifactKind,
    ArtifactStore,
    ObservationStatus,
    ProvenanceCycle,
    ProvenanceRepository,
    RunStatus,
    SubjectRef,
    connect,
)


@pytest.fixture
def stack(tmp_path):
    conn = connect(tmp_path)
    return ArtifactStore(tmp_path, conn), ProvenanceRepository(conn)


def _put(store, content: bytes):
    return store.put_bytes(
        content, kind=ArtifactKind.FEATURE_BLOB, media_type="application/octet-stream"
    )


# --- content addressing (Law 3) -------------------------------------------
def test_put_is_content_addressed_and_readable(stack):
    store, _ = stack
    art = _put(store, b"hello world")
    assert len(art.content_sha256) == 64
    with store.open(art.content_sha256) as fh:
        assert fh.read() == b"hello world"


def test_identical_bytes_dedupe_to_one_artifact(stack):
    store, _ = stack
    a = _put(store, b"same")
    b = _put(store, b"same")
    assert a.content_sha256 == b.content_sha256
    rows = store._conn.execute("SELECT COUNT(*) c FROM artifact").fetchone()
    assert rows["c"] == 1


def test_verify_true_then_false_on_tamper(stack):
    store, _ = stack
    art = _put(store, b"trustme")
    assert store.verify(art.content_sha256) is True
    # Path-swap the object bytes -> verify must catch it (bytes are identity).
    store._object_path(art.content_sha256).write_bytes(b"tampered")
    assert store.verify(art.content_sha256) is False


# --- run lifecycle ---------------------------------------------------------
def _run(repo):
    return repo.begin_run(
        process="test.proc",
        process_version="1",
        code_commit="deadbeef",
        params_hash="0",
    )


def test_run_begins_running_then_succeeds(stack):
    store, repo = stack
    run = _run(repo)
    assert run.status == RunStatus.RUNNING
    repo.succeed(run)
    row = store._conn.execute(
        "SELECT status FROM run WHERE run_id=?", (run.run_id,)
    ).fetchone()
    assert row["status"] == RunStatus.SUCCEEDED.value


# --- lineage + closure (Law 1) --------------------------------------------
def test_closure_returns_transitive_ancestors(stack):
    store, repo = stack
    run = _run(repo)
    a, b, c = _put(store, b"A"), _put(store, b"B"), _put(store, b"C")
    repo.derive(b, a, run, "derived_from")
    repo.derive(c, b, run, "derived_from")
    assert repo.closure(c.content_sha256) == {a.content_sha256, b.content_sha256}
    assert repo.closure(a.content_sha256) == set()


# --- acyclicity (Law 2) ----------------------------------------------------
def test_derive_rejects_cycle(stack):
    store, repo = stack
    run = _run(repo)
    a, b = _put(store, b"A"), _put(store, b"B")
    repo.derive(b, a, run, "derived_from")  # b <- a
    with pytest.raises(ProvenanceCycle):
        repo.derive(a, b, run, "derived_from")  # a <- b closes the loop


def test_derive_rejects_self_edge(stack):
    store, repo = stack
    run = _run(repo)
    a = _put(store, b"A")
    with pytest.raises(ProvenanceCycle):
        repo.derive(a, a, run, "derived_from")


# --- observations, incl. persisted abstention (Law 6/7) --------------------
def test_observation_records_value_and_abstention(stack):
    store, repo = stack
    run = _run(repo)
    src = store.put_bytes(
        b"<html/>", kind=ArtifactKind.HTML_ROW, media_type="text/html"
    )
    subj = SubjectRef("set-slot", "2nvzlh2k::013w1")

    repo.record_observation(
        subject=subj, predicate="claimed_title", value="Some Song", source=src, run=run
    )
    repo.record_observation(
        subject=subj,
        predicate="source_track_key",
        value=None,
        source=src,
        run=run,
        status=ObservationStatus.ABSTAINED,
        diagnostic_code="no_trackid",
    )
    rows = store._conn.execute(
        "SELECT predicate, status, diagnostic_code FROM observation ORDER BY predicate"
    ).fetchall()
    assert len(rows) == 2
    abstained = [r for r in rows if r["status"] == ObservationStatus.ABSTAINED.value]
    assert abstained and abstained[0]["diagnostic_code"] == "no_trackid"
