"""Tests for co-training dataset loading."""

from pathlib import Path
from alignment.dataset import load_set
from core.result import Ok

_REPO = Path(__file__).resolve().parents[2]
BB12 = _REPO / "labeling/fixtures/bb12_ground_truth.yaml"


def test_load_set_stamps_set_id_on_every_target():
    match load_set(BB12):
        case Ok((gt, targets)):
            assert len(targets) > 0
            assert all(t.set_id == gt.set_id for t in targets)
            assert gt.set_id  # non-empty
        case _:
            raise AssertionError("bb12 fixture failed to load")


import alignment.cotrain as ct
from alignment.cotrain import SetStores, cotrain


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


def test_run_loso_holds_each_set_out(monkeypatch):
    import alignment.cotrain as ctmod

    # two fake sets; capture which set is held out on each cotrain call
    held_train_ids = []

    def fake_load_stores(set_id):  # returns a SetStores-like per set
        return SetStores(set_id, (f"{set_id}span",), object(), {}, {"s": ()})

    def fake_cotrain(train_sets, **kw):
        held_train_ids.append(tuple(s.set_id for s in train_sets))
        return "HEAD"

    class FakeAligner:
        def __init__(self, **kw):
            pass

        def predict_sequence(self, spans):
            return ("pred",)

    def fake_eval(preds, spans):
        class R:  # minimal report stub
            def lines(self):
                return ["median=1.0s"]

        return R()

    monkeypatch.setattr(ctmod, "_load_set_stores", fake_load_stores, raising=False)
    monkeypatch.setattr(ctmod, "cotrain", fake_cotrain)
    monkeypatch.setattr(ctmod, "MertLearnedAligner", FakeAligner, raising=False)
    monkeypatch.setattr(ctmod, "evaluate", fake_eval, raising=False)

    rep = ctmod.run_loso({"a": "a.yaml", "b": "b.yaml"}, device="cpu")
    # each set held out once; the OTHER set is the training set
    assert set(rep.keys()) == {"a", "b"}
    assert ("b",) in held_train_ids and ("a",) in held_train_ids
