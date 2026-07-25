"""Forward-routing seam (cutover Decision 2 step 2): new analysis routed through
the registry lands as a canonical, provenanced feature in the central store."""

from __future__ import annotations

import pytest

from core.provenance.laws import LawVerdict, check_laws
from core.provenance.registry import REGISTRY, artifact_type, transformation
from workspaces.alignment_prototype import canonical_ingest as ci


@pytest.fixture(autouse=True)
def _preserve_registry():
    # canonical_ingest imports producers (registers the real analyze.* at import).
    # Save + restore around each test so a throwaway dummy producer never wipes
    # the real ones (reset-to-empty would, since import-time registration is gone).
    saved_types = dict(REGISTRY._artifact_types)
    saved_transforms = dict(REGISTRY._transformations)
    yield
    REGISTRY._artifact_types.clear()
    REGISTRY._artifact_types.update(saved_types)
    REGISTRY._transformations.clear()
    REGISTRY._transformations.update(saved_transforms)


def _register_dummy() -> str:
    @artifact_type(kind="unit_probe_feat", media_type="application/octet-stream")
    @transformation(
        name="analyze.unit_probe",
        version="1",
        inputs={"audio": "audio"},
        outputs={"features": "unit_probe_feat"},
        stage="analysis",
        params={"k": 3},
    )
    def _p(inputs):
        return {"features": inputs["audio"][:4] + b"::probe"}

    return "analyze.unit_probe"


def test_forward_routed_feature_is_content_addressed_and_provenanced(tmp_path):
    name = _register_dummy()
    store, repo = ci.open_central_store(tmp_path)
    audio = ci.register_audio(store, b"REALAUDIOBYTES", recording_id="rec_x")

    out = ci.ingest_feature(name, audio, store=store, repo=repo, code_commit="deadbee")
    feat = out["features"]

    assert store.verify(feat.content_sha256)  # content-addressed
    c = store._conn
    assert (
        c.execute(
            "SELECT COUNT(*) n FROM derivation WHERE child_sha256=? AND parent_sha256=?",
            (feat.content_sha256, audio.content_sha256),
        ).fetchone()["n"]
        == 1
    )  # derived from the audio
    run = c.execute(
        "SELECT process_spec_id FROM run_output ro JOIN run r ON r.run_id=ro.run_id "
        "WHERE ro.content_sha256=?",
        (feat.content_sha256,),
    ).fetchone()
    assert run["process_spec_id"] is not None  # fully versioned


def test_ingest_is_idempotent_on_content(tmp_path):
    name = _register_dummy()
    store, repo = ci.open_central_store(tmp_path)
    audio = ci.register_audio(store, b"SAME", recording_id="r")
    a = ci.ingest_feature(name, audio, store=store, repo=repo, code_commit="c")
    b = ci.ingest_feature(name, audio, store=store, repo=repo, code_commit="c")
    assert a["features"].content_sha256 == b["features"].content_sha256
    # same ProcessSpec reused (deterministic id), only a second run row
    assert (
        store._conn.execute("SELECT COUNT(*) n FROM process_spec").fetchone()["n"] == 1
    )
    assert store._conn.execute("SELECT COUNT(*) n FROM run").fetchone()["n"] == 2


def test_store_passes_law_checker(tmp_path):
    name = _register_dummy()
    store, repo = ci.open_central_store(tmp_path)
    audio = ci.register_audio(store, b"AUDIO", recording_id="r")
    ci.ingest_feature(name, audio, store=store, repo=repo, code_commit="c")
    results = {r.number: r.verdict for r in check_laws(store._conn)}
    assert results[1] != LawVerdict.VIOLATED  # provenance complete
    assert results[3] != LawVerdict.VIOLATED  # artifacts verify
    assert results[14] != LawVerdict.VIOLATED  # versioned code/env/params


def test_audio_producers_lists_the_real_analyzers():
    # producers.py registered analyze.landmark_fp / whisper / hubert at import,
    # and the preserve fixture keeps them alive.
    names = ci.audio_producers()
    assert "analyze.landmark_fp" in names
    assert "analyze.whisper" in names and "analyze.hubert" in names
