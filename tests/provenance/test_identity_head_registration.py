"""Brick 9 — a saved MERT identity head registers as FittedModel(axis=IDENTITY)
and the store passes laws 13/14.

Uses a tiny REAL head checkpoint (the exact `external/checkpoint.py` payload
`train.py --save-head-checkpoint` writes) against a tmp store; the production
run registers the actually-trained BB12 head into the Brick-7 producers store.
"""

from __future__ import annotations

import pytest

from core.provenance import ArtifactStore, Axis, ProvenanceRepository, connect
from core.provenance.laws import LawVerdict, check_laws

torch = pytest.importorskip("torch")

from alignment.register_identity_head import (  # noqa: E402
    register_mert_identity_head,
)


@pytest.fixture
def world(tmp_path):
    root = tmp_path / "store"
    conn = connect(root)
    store = ArtifactStore(root, conn)
    repo = ProvenanceRepository(conn)
    return conn, store, repo


@pytest.fixture
def training_files(tmp_path):
    """A tiny real head checkpoint + stand-in GT/MERT training-input files."""
    from alignment.external.checkpoint import (
        PretrainMeta,
        save_head,
    )
    from alignment.mert_model import MertAlignHead, TrainConfig

    ckpt = tmp_path / "head.pt"
    save_head(
        MertAlignHead(4),
        ckpt,
        meta=PretrainMeta(
            feature_kind="mert",
            dim=4,
            n_heads=1,
            n_examples=2,
            n_mixes=1,
            source="bb_gt:testset",
        ),
        cfg=TrainConfig(epochs=1, n_heads=1),
    )
    gt = tmp_path / "testset_ground_truth.yaml"
    gt.write_text("set_id: testset\nspans: []\n")
    mert = tmp_path / "testset_mert.npz"
    mert.write_bytes(b"PK\x03\x04 fake-npz training features")
    return ckpt, gt, mert


def _register(store, repo, training_files, **kw):
    ckpt, gt, mert = training_files
    return register_mert_identity_head(
        ckpt,
        set_id="testset",
        store=store,
        repo=repo,
        code_commit="deadbeef",
        gt_path=gt,
        mert_cache=mert,
        **kw,
    )


def test_registers_identity_fitted_model_row(world, training_files):
    conn, store, repo = world
    model = _register(store, repo, training_files)
    assert model is not None
    assert model.axis is Axis.IDENTITY

    row = conn.execute(
        "SELECT axis, serialized_state_sha256, training_snapshot_id, "
        "process_spec_id FROM fitted_model"
    ).fetchone()
    assert row["axis"] == "identity"
    # the snapshot holds exactly the two REAL training inputs (GT + features)
    import json

    snap = conn.execute(
        "SELECT member_artifact_shas_json FROM training_snapshot "
        "WHERE training_snapshot_id=?",
        (row["training_snapshot_id"],),
    ).fetchone()
    assert snap is not None
    assert len(json.loads(snap["member_artifact_shas_json"])) == 2
    # process spec carries the training params + code commit
    spec = conn.execute(
        "SELECT name, stage, code_commit FROM process_spec WHERE process_spec_id=?",
        (row["process_spec_id"],),
    ).fetchone()
    assert spec["name"] == "train.mert_identity_head"
    assert spec["stage"] == "training"
    assert spec["code_commit"] == "deadbeef"


def test_laws_13_and_14_pass_with_identity_model(world, training_files):
    conn, store, repo = world
    assert _register(store, repo, training_files) is not None
    results = {r.number: r for r in check_laws(conn, store)}
    assert results[13].verdict is LawVerdict.PASS, results[13]
    assert results[14].verdict is LawVerdict.PASS, results[14]
    # and the whole store stays violation-free
    assert not [r for r in results.values() if r.verdict is LawVerdict.VIOLATED]


def test_registration_is_idempotent(world, training_files):
    conn, store, repo = world
    m1 = _register(store, repo, training_files)
    m2 = _register(store, repo, training_files)
    assert m1 is not None and m2 is not None
    assert m1.fitted_model_id == m2.fitted_model_id
    n = conn.execute("SELECT COUNT(*) FROM fitted_model").fetchone()[0]
    assert n == 1


def test_missing_inputs_refuse_to_register(world, training_files, tmp_path):
    _conn, store, repo = world
    ckpt, gt, _mert = training_files
    missing = tmp_path / "nope.npz"
    model = register_mert_identity_head(
        ckpt,
        set_id="testset",
        store=store,
        repo=repo,
        code_commit="deadbeef",
        gt_path=gt,
        mert_cache=missing,
    )
    assert model is None
    n = _conn.execute("SELECT COUNT(*) FROM fitted_model").fetchone()[0]
    assert n == 0
