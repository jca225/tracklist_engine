"""register_computed_feature — the substrate primitive behind the dual-write."""

from __future__ import annotations

from core.provenance import (
    ArtifactKind,
    ArtifactStore,
    ProvenanceRepository,
    connect,
    register_computed_feature,
)
from core.provenance.laws import LawVerdict, check_laws


def _stack(tmp_path):
    conn = connect(tmp_path)
    return ArtifactStore(tmp_path, conn), ProvenanceRepository(conn)


def test_feature_is_content_addressed_and_versioned(tmp_path):
    store, repo = _stack(tmp_path)
    art = register_computed_feature(
        "mert_features",
        b"FEATUREBYTES",
        media_type="application/octet-stream",
        analyzer_name="analysis.mert",
        analyzer_version="1",
        store=store,
        repo=repo,
        model_refs=("m-a-p/MERT-v1-95M", "layer=6"),
    )
    assert store.verify(art.content_sha256)
    row = store._conn.execute(
        "SELECT r.process_spec_id FROM run_output ro JOIN run r ON r.run_id=ro.run_id "
        "WHERE ro.content_sha256=?",
        (art.content_sha256,),
    ).fetchone()
    assert row["process_spec_id"] is not None  # a real versioned run produced it


def test_derives_from_audio_when_given(tmp_path):
    store, repo = _stack(tmp_path)
    audio = store.put_bytes(
        b"AUDIO",
        kind=ArtifactKind.AUDIO,
        media_type="audio/mp4",
        source_system="tracklist_engine.audio_rip",
    )
    feat = register_computed_feature(
        "mert_features",
        b"F",
        media_type="application/octet-stream",
        analyzer_name="analysis.mert",
        analyzer_version="1",
        store=store,
        repo=repo,
        audio=audio,
    )
    n = store._conn.execute(
        "SELECT COUNT(*) n FROM derivation WHERE child_sha256=? AND parent_sha256=?",
        (feat.content_sha256, audio.content_sha256),
    ).fetchone()["n"]
    assert n == 1


def test_idempotent_process_spec(tmp_path):
    store, repo = _stack(tmp_path)
    for _ in range(2):
        register_computed_feature(
            "mert_features",
            b"SAME",
            media_type="application/octet-stream",
            analyzer_name="analysis.mert",
            analyzer_version="1",
            store=store,
            repo=repo,
            code_commit="c",
        )
    assert (
        store._conn.execute("SELECT COUNT(*) n FROM process_spec").fetchone()["n"] == 1
    )
    assert store._conn.execute("SELECT COUNT(*) n FROM artifact").fetchone()["n"] == 1
    assert store._conn.execute("SELECT COUNT(*) n FROM run").fetchone()["n"] == 2


def test_store_passes_law_checker(tmp_path):
    store, repo = _stack(tmp_path)
    register_computed_feature(
        "mert_features",
        b"F",
        media_type="application/octet-stream",
        analyzer_name="analysis.mert",
        analyzer_version="1",
        store=store,
        repo=repo,
    )
    verdicts = {r.number: r.verdict for r in check_laws(store._conn)}
    assert verdicts[1] != LawVerdict.VIOLATED  # feature traces to its producing run
    assert verdicts[14] != LawVerdict.VIOLATED  # versioned code/env/params
