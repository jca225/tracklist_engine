"""Reconcile crash residue (enable-blocker #1) — stale runs + orphan artifacts."""

from __future__ import annotations

from core.provenance import (
    ArtifactKind,
    ArtifactStore,
    ProvenanceRepository,
    check_laws,
    connect,
    reconcile_store,
    register_computed_feature,
)
from core.provenance.laws import LawVerdict


def _stack(tmp_path):
    conn = connect(tmp_path)
    return conn, ArtifactStore(tmp_path, conn), ProvenanceRepository(conn)


def test_stale_running_run_is_failed_and_law1_recovers(tmp_path):
    conn, store, repo = _stack(tmp_path)
    # A run left 'running' (process died before succeed/fail) — a Law-1 offender.
    spec_args = dict(process="x", process_version="1", code_commit="c", params_hash="0")
    run = repo.begin_run(**spec_args)
    assert (
        conn.execute("SELECT status FROM run WHERE run_id=?", (run.run_id,)).fetchone()[
            "status"
        ]
        == "running"
    )
    # law 1 flags the unfinished run
    before = {r.number: r.verdict for r in check_laws(conn)}
    assert before[1] == LawVerdict.VIOLATED

    rep = reconcile_store(conn)
    assert rep.stale_runs_failed == 1
    assert (
        conn.execute("SELECT status FROM run WHERE run_id=?", (run.run_id,)).fetchone()[
            "status"
        ]
        == "failed"
    )
    after = {r.number: r.verdict for r in check_laws(conn)}
    assert after[1] != LawVerdict.VIOLATED  # recovered


def test_orphan_artifact_reported_and_pruned(tmp_path):
    conn, store, repo = _stack(tmp_path)
    # Simulate the crash window: an artifact committed with no producing run
    # (put_bytes landed, add_output never did).
    art = store.put_bytes(
        b"orphan", kind="mert_features", media_type="application/octet-stream"
    )
    orphans = reconcile_store(conn).orphan_artifacts
    assert art.content_sha256 in orphans
    # report-only by default (row still present)
    assert (
        conn.execute(
            "SELECT COUNT(*) n FROM artifact WHERE content_sha256=?",
            (art.content_sha256,),
        ).fetchone()["n"]
        == 1
    )
    # --prune removes it → law 1 clean
    rep = reconcile_store(conn, prune_orphans=True)
    assert rep.pruned_artifacts == 1
    assert (
        conn.execute(
            "SELECT COUNT(*) n FROM artifact WHERE content_sha256=?",
            (art.content_sha256,),
        ).fetchone()["n"]
        == 0
    )
    assert {r.number: r.verdict for r in check_laws(conn)}[1] != LawVerdict.VIOLATED


def test_completed_feature_is_not_touched(tmp_path):
    # A properly-registered feature (run succeeded, output edge present) is NOT an
    # orphan and its run is NOT stale — reconcile leaves a clean store alone.
    conn, store, repo = _stack(tmp_path)
    register_computed_feature(
        "mert_features",
        b"good",
        media_type="application/octet-stream",
        analyzer_name="analysis.mert",
        analyzer_version="1",
        store=store,
        repo=repo,
    )
    rep = reconcile_store(conn)
    assert rep.clean
    # idempotent
    assert reconcile_store(conn).clean


def test_audio_source_root_is_not_an_orphan(tmp_path):
    # An audio artifact marked as a source (source_system) is a legitimate root,
    # never pruned even with no producing run.
    conn, store, repo = _stack(tmp_path)
    store.put_bytes(
        b"AUDIO",
        kind=ArtifactKind.AUDIO,
        media_type="audio/mp4",
        source_system="tracklist_engine.audio_rip",
    )
    rep = reconcile_store(conn, prune_orphans=True)
    assert rep.orphan_artifacts == () and rep.pruned_artifacts == 0
