"""Tests for co-training dataset loading."""

from pathlib import Path
from workspaces.alignment_prototype.dataset import load_set
from core.result import Ok

_REPO = Path(__file__).resolve().parents[3]
BB12 = _REPO / "labeling/fixtures/bb12_ground_truth.yaml"


def test_load_set_stamps_set_id_on_every_target():
    match load_set(BB12):
        case Ok((gt, targets)):
            assert len(targets) > 0
            assert all(t.set_id == gt.set_id for t in targets)
            assert gt.set_id  # non-empty
        case _:
            raise AssertionError("bb12 fixture failed to load")


import workspaces.alignment_prototype.cotrain as ct
from workspaces.alignment_prototype.cotrain import SetStores, cotrain


def test_cotrain_concatenates_examples_across_sets(monkeypatch):
    # build_examples returns a per-set stub list; train_ensemble captures the
    # concatenated length. No GPU / no real stores.
    calls = {}

    def fake_build_examples(spans, mix, refs, pools, **kw):
        return ["ex"] * len(spans)  # one example per span

    def fake_train_ensemble(examples, **kw):
        calls["n"] = len(examples)
        return "HEAD"

    monkeypatch.setattr(ct, "build_examples", fake_build_examples)
    monkeypatch.setattr(ct, "train_ensemble", fake_train_ensemble)

    s1 = SetStores("a", ("x", "y"), None, {}, {})  # 2 spans
    s2 = SetStores("b", ("p", "q", "r"), None, {}, {})  # 3 spans
    head = cotrain([s1, s2], device="cpu")
    assert head == "HEAD"
    assert calls["n"] == 5  # 2 + 3 concatenated
