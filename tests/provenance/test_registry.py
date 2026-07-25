"""Brick 7 — the metaprogramming registry, proved on a BRAND-NEW data type.

The acceptance property: registering a never-seen-before artifact type + a
transformation over it (two decorators, zero extra plumbing) and running it
through ``Registry.run`` yields an output that is content-addressed AND fully
provenanced — input/output/derivation edges plus a deterministic, persisted
ProcessSpec (ParameterSet + EnvironmentSpec). If that holds for a dummy type
it holds for any future feature kind.
"""

from __future__ import annotations

import json

import pytest

from core.provenance import (
    ArtifactKind,
    ArtifactStore,
    ProvenanceRepository,
    Registry,
    artifact_type,
    connect,
    transformation,
)
from core.provenance.registry import REGISTRY


@pytest.fixture
def world(tmp_path):
    conn = connect(tmp_path)
    store = ArtifactStore(tmp_path, conn)
    repo = ProvenanceRepository(conn)
    return conn, store, repo


@pytest.fixture
def registry():
    """An isolated Registry so tests never leak declarations."""
    return Registry()


def _declare_dummy(registry: Registry) -> None:
    # Same declarations the decorators would make, pointed at an isolated
    # registry (the decorator API itself is covered against the singleton).
    from core.provenance.registry import ArtifactTypeSpec, TransformationSpec

    registry.register_artifact_type(
        ArtifactTypeSpec(kind="dummy_spectral_moment", media_type="application/json")
    )

    def compute(inputs):
        payload = inputs["audio"]
        return {
            "features": json.dumps(
                {"n_bytes": len(payload), "first": payload[:1].hex()}
            ).encode()
        }

    registry.register_transformation(
        TransformationSpec(
            name="analyze.dummy_spectral_moment",
            version="1",
            inputs={"audio": "audio"},
            outputs={"features": "dummy_spectral_moment"},
            stage="analysis",
            params={"order": 2},
            fn=compute,
        )
    )


# --- the proof test ----------------------------------------------------------
def test_new_dummy_type_is_content_addressed_and_fully_provenanced(world, registry):
    conn, store, repo = world
    _declare_dummy(registry)

    audio = store.put_bytes(
        b"\x00\x01fake-audio-bytes",
        kind=ArtifactKind.AUDIO,
        media_type="audio/mp4",
        source_system="test",
    )
    outputs = registry.run(
        "analyze.dummy_spectral_moment",
        {"audio": audio},
        store=store,
        repo=repo,
        code_commit="abc1234",
    )
    feat = outputs["features"]

    # Content-addressed: stored bytes re-hash to the id, and the payload is
    # exactly what the pure fn returned.
    assert store.verify(feat.content_sha256)
    assert feat.kind == "dummy_spectral_moment"
    with store.open(feat.content_sha256) as f:
        assert json.loads(f.read())["n_bytes"] == len(b"\x00\x01fake-audio-bytes")

    # Fully provenanced: one succeeded run with the input, output, and a
    # derivation edge feature <- audio, relation = transformation name.
    run_row = conn.execute("SELECT * FROM run").fetchone()
    assert run_row["status"] == "succeeded"
    assert run_row["process"] == "analyze.dummy_spectral_moment"
    inp = conn.execute("SELECT * FROM run_input").fetchone()
    assert inp["content_sha256"] == audio.content_sha256 and inp["role"] == "audio"
    out = conn.execute("SELECT * FROM run_output").fetchone()
    assert out["content_sha256"] == feat.content_sha256 and out["role"] == "features"
    der = conn.execute("SELECT * FROM derivation").fetchone()
    assert der["child_sha256"] == feat.content_sha256
    assert der["parent_sha256"] == audio.content_sha256
    assert der["relation"] == "analyze.dummy_spectral_moment"

    # Versioned: the run points at a persisted ProcessSpec whose ParameterSet
    # and EnvironmentSpec exist (this is what law 14 checks).
    spec = conn.execute(
        "SELECT * FROM process_spec WHERE process_spec_id=?",
        (run_row["process_spec_id"],),
    ).fetchone()
    assert spec is not None
    assert spec["code_commit"] == "abc1234"
    assert json.loads(
        conn.execute(
            "SELECT values_json FROM parameter_set WHERE parameter_set_id=?",
            (spec["parameter_set_id"],),
        ).fetchone()["values_json"]
    ) == {"order": 2}
    assert (
        conn.execute(
            "SELECT 1 FROM environment_spec WHERE environment_spec_id=?",
            (spec["environment_spec_id"],),
        ).fetchone()
        is not None
    )
    assert spec["implementation_hash"]


def test_rerun_is_idempotent_on_specs_and_dedupes_output(world, registry):
    conn, store, repo = world
    _declare_dummy(registry)
    audio = store.put_bytes(
        b"same-bytes",
        kind=ArtifactKind.AUDIO,
        media_type="audio/mp4",
        source_system="test",
    )
    a = registry.run(
        "analyze.dummy_spectral_moment",
        {"audio": audio},
        store=store,
        repo=repo,
        code_commit="abc1234",
    )
    b = registry.run(
        "analyze.dummy_spectral_moment",
        {"audio": audio},
        store=store,
        repo=repo,
        code_commit="abc1234",
    )
    # Same bytes in, pure fn -> same content sha out; artifact deduped.
    assert a["features"].content_sha256 == b["features"].content_sha256
    assert conn.execute("SELECT COUNT(*) FROM artifact").fetchone()[0] == 2
    # Identical process -> ONE process_spec row; two runs reference it.
    assert conn.execute("SELECT COUNT(*) FROM process_spec").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM run").fetchone()[0] == 2


def test_input_kind_mismatch_is_rejected_before_any_provenance(world, registry):
    conn, store, repo = world
    _declare_dummy(registry)
    not_audio = store.put_bytes(
        b"{}",
        kind=ArtifactKind.PREDICTED_TIMELINE,
        media_type="application/json",
        source_system="test",
    )
    with pytest.raises(ValueError, match="declared 'audio'"):
        registry.run(
            "analyze.dummy_spectral_moment",
            {"audio": not_audio},
            store=store,
            repo=repo,
            code_commit="abc1234",
        )
    assert conn.execute("SELECT COUNT(*) FROM run").fetchone()[0] == 0


def test_missing_input_role_is_rejected(world, registry):
    _, store, repo = world
    _declare_dummy(registry)
    with pytest.raises(ValueError, match="input roles"):
        registry.run(
            "analyze.dummy_spectral_moment",
            {},
            store=store,
            repo=repo,
            code_commit="abc1234",
        )


def test_fn_failure_records_failed_run_and_reraises(world, registry):
    from core.provenance.registry import ArtifactTypeSpec, TransformationSpec

    conn, store, repo = world
    registry.register_artifact_type(
        ArtifactTypeSpec(kind="doomed_feature", media_type="application/json")
    )

    def explode(inputs):
        raise RuntimeError("decode blew up")

    registry.register_transformation(
        TransformationSpec(
            name="analyze.doomed",
            version="1",
            inputs={"audio": "audio"},
            outputs={"features": "doomed_feature"},
            stage="analysis",
            fn=explode,
        )
    )
    audio = store.put_bytes(
        b"x", kind=ArtifactKind.AUDIO, media_type="audio/mp4", source_system="test"
    )
    with pytest.raises(RuntimeError, match="decode blew up"):
        registry.run(
            "analyze.doomed",
            {"audio": audio},
            store=store,
            repo=repo,
            code_commit="abc1234",
        )
    row = conn.execute("SELECT status, error_json FROM run").fetchone()
    assert row["status"] == "failed"
    assert "decode blew up" in row["error_json"]


def test_undeclared_output_role_fails_the_run(world, registry):
    from core.provenance.registry import ArtifactTypeSpec, TransformationSpec

    conn, store, repo = world
    registry.register_artifact_type(
        ArtifactTypeSpec(kind="rogue_feature", media_type="application/json")
    )

    def rogue(inputs):
        return {"features": b"{}", "extra": b"{}"}  # 'extra' undeclared

    registry.register_transformation(
        TransformationSpec(
            name="analyze.rogue",
            version="1",
            inputs={"audio": "audio"},
            outputs={"features": "rogue_feature"},
            stage="analysis",
            fn=rogue,
        )
    )
    audio = store.put_bytes(
        b"x", kind=ArtifactKind.AUDIO, media_type="audio/mp4", source_system="test"
    )
    with pytest.raises(ValueError, match="returned roles"):
        registry.run(
            "analyze.rogue",
            {"audio": audio},
            store=store,
            repo=repo,
            code_commit="abc1234",
        )
    assert conn.execute("SELECT status FROM run").fetchone()["status"] == "failed"


# --- decorators + graph ------------------------------------------------------
def test_decorators_register_into_singleton_and_return_target_unchanged():
    # Snapshot the singleton so other modules' import-time declarations
    # (e.g. workspaces producers) survive this test regardless of ordering.
    saved_types = dict(REGISTRY._artifact_types)
    saved_transforms = dict(REGISTRY._transformations)
    REGISTRY.reset()
    try:

        @artifact_type(kind="deco_feature", media_type="application/json")
        class DecoFeature:
            pass

        @transformation(
            "analyze.deco",
            "1",
            inputs={"audio": "audio"},
            outputs={"features": "deco_feature"},
            stage="analysis",
        )
        def produce(inputs):
            return {"features": b"{}"}

        assert DecoFeature.__name__ == "DecoFeature"  # returned unchanged
        assert produce({"audio": b""}) == {"features": b"{}"}  # still callable
        assert REGISTRY.artifact_type("deco_feature").media_type == ("application/json")
        assert REGISTRY.producers_of("deco_feature") == ["analyze.deco"]
        assert REGISTRY.graph()["analyze.deco"] == {
            "inputs": {"audio": "audio"},
            "outputs": {"features": "deco_feature"},
        }
    finally:
        REGISTRY.reset()
        REGISTRY._artifact_types.update(saved_types)
        REGISTRY._transformations.update(saved_transforms)


def test_unknown_transformation_and_kind_raise(registry):
    with pytest.raises(ValueError, match="not registered"):
        registry.transformation("analyze.nope")
    with pytest.raises(ValueError, match="not registered"):
        registry.artifact_type("nope_kind")
    assert registry.producers_of("nope_kind") == []
