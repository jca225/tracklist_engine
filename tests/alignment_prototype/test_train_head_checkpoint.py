"""Brick 9 — `train.py --save-head-checkpoint` is opt-in and default-off.

GPU/network-free: exercises the CLI wiring (the flag defaults to None and is
threaded verbatim into `_run_mert_eval`) and the exact save/load payload the
flag uses (`external/checkpoint.py`) on a tiny real head. The full training
path is exercised by the CLI run, not here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from workspaces.alignment_prototype import train as train_mod


@pytest.fixture
def captured_mert_eval(monkeypatch):
    """Stub the heavy MERT path; record the kwargs main() threads into it."""
    calls: list[dict] = []

    def _fake(*args, **kwargs):
        calls.append(kwargs)
        return 0

    monkeypatch.setattr(train_mod, "_run_mert_eval", _fake)
    return calls


def test_save_flag_defaults_to_none(captured_mert_eval):
    rc = train_mod.main(["--eval", "--train-mert"])
    assert rc == 0
    assert len(captured_mert_eval) == 1
    assert captured_mert_eval[0]["save_checkpoint"] is None


def test_save_flag_threads_path_through(captured_mert_eval, tmp_path):
    target = tmp_path / "head.pt"
    rc = train_mod.main(
        ["--eval", "--train-mert", "--save-head-checkpoint", str(target)]
    )
    assert rc == 0
    assert captured_mert_eval[0]["save_checkpoint"] == target


def test_dry_run_never_touches_the_flag_path(monkeypatch, tmp_path):
    def _boom(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("_run_mert_eval must not run under --dry-run")

    monkeypatch.setattr(train_mod, "_run_mert_eval", _boom)
    target = tmp_path / "head.pt"
    rc = train_mod.main(["--dry-run", "--save-head-checkpoint", str(target)])
    assert rc == 0
    assert not target.exists()


def test_saved_head_round_trips_through_checkpoint_module(tmp_path):
    torch = pytest.importorskip("torch")
    from workspaces.alignment_prototype.external.checkpoint import (
        PretrainMeta,
        load_head,
        save_head,
    )
    from workspaces.alignment_prototype.mert_model import MertAlignHead, TrainConfig

    dim = 8
    head = MertAlignHead(dim)
    path = tmp_path / "tiny_head.pt"
    save_head(
        head,
        path,
        meta=PretrainMeta(
            feature_kind="mert",
            dim=dim,
            n_heads=1,
            n_examples=3,
            n_mixes=1,
            source="bb_gt:testset",
        ),
        cfg=TrainConfig(epochs=1, n_heads=1),
    )
    assert path.is_file()
    loaded, meta = load_head(path, device="cpu", expected_dim=dim)
    assert isinstance(loaded, MertAlignHead)
    assert meta.source == "bb_gt:testset"
    # identical weights back
    for a, b in zip(head.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)
