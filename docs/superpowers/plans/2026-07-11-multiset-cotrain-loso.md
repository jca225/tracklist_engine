# Multi-Set Co-Train + LOSO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the learned aligner train on multiple GT sets at once and report a leave-one-set-out (LOSO) generalization number — fixing the flywheel's broken retrain gear.

**Architecture:** The head trains on set-agnostic materialized examples (`build_examples` already exists per set), so co-train = concatenate per-set examples and train one head; LOSO = wrap that head around the held-out set's stores and predict. Thread `set_id` through `SpanTarget` for bookkeeping.

**Tech Stack:** Python 3, PyTorch (MPS/CPU), `venvs/audio/bin/python`. MERT stores load from pi-storage (cached locally).

## Global Constraints

- Run from repo root `/Users/johnnycabrahams/Desktop/tracklist_engine` with `venvs/audio/bin/python`.
- Test form: `venvs/audio/bin/python -m pytest <path> -v`
- Branch first: `git checkout -b flywheel-cotrain-loso` before Task 1.
- Unit/golden tests MUST NOT depend on pi-storage or GPU — monkeypatch `build_examples`, `train_ensemble`, and store loaders.
- The existing single-set `train.py --eval [--train-mert]` path must keep working unchanged (additive only).
- Cross-set eval MUST anchor placement on **scraped tracklist cues** (aligner input), never the held-out set's GT — see Task 3. Anchoring on GT is leakage; `anchor_sigma_s=None` collapses to front-of-mix (floor artifact). Cues are fair.
- Existing tests live in `workspaces/alignment_prototype/tests/` and `.../external/tests/`. New tests: `workspaces/alignment_prototype/tests/test_cotrain.py`.
- GT fixtures: `labeling/fixtures/bb11_ground_truth.yaml`, `labeling/fixtures/bb12_ground_truth.yaml`.

---

### Task 1: Thread `set_id` through `SpanTarget`

**Files:**
- Modify: `workspaces/alignment_prototype/records.py` (SpanTarget dataclass)
- Modify: `workspaces/alignment_prototype/dataset.py` (`track_to_target`, `load_set`)
- Test: `workspaces/alignment_prototype/tests/test_cotrain.py` (create)

**Interfaces:**
- Produces: `SpanTarget.set_id: str` (default `""`, additive/non-breaking). `dataset.track_to_target(t, set_id: str = "")` populates it; `load_set` passes `gt.set_id`.

- [ ] **Step 1: Write the failing test**

```python
# workspaces/alignment_prototype/tests/test_cotrain.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest workspaces/alignment_prototype/tests/test_cotrain.py -v`
Expected: FAIL — `SpanTarget` has no `set_id` (AttributeError) or the values are empty.

- [ ] **Step 3: Add the field and populate it**

In `records.py`, add to the `SpanTarget` frozen dataclass, as the LAST field (so existing positional construction elsewhere is unaffected):

```python
    set_id: str = ""
```

In `dataset.py`, change `track_to_target` to accept and set it:

```python
def track_to_target(t: GroundTruthTrack, set_id: str = "") -> SpanTarget:
    return SpanTarget(
        # ... existing fields unchanged ...
        set_id=set_id,
    )
```

Then in `load_set`, where targets are built from the parsed `gt`, pass `gt.set_id`. The current build is a comprehension over `gt.tracks`; change it to:

```python
        targets = tuple(track_to_target(t, set_id=gt.set_id) for t in gt.tracks)
```

(Read `load_set`'s body to place this exactly — the parse yields `gt`; keep the surrounding `Result` handling identical.)

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest workspaces/alignment_prototype/tests/test_cotrain.py -v`
Expected: PASS.

Also confirm no regression in the loader-dependent suite:
Run: `venvs/audio/bin/python -m pytest workspaces/alignment_prototype/tests/ -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add workspaces/alignment_prototype/records.py workspaces/alignment_prototype/dataset.py \
        workspaces/alignment_prototype/tests/test_cotrain.py
git commit -m "feat(records): SpanTarget.set_id, stamped by load_set"
```

---

### Task 2: `cotrain.py` — multi-set head training

**Files:**
- Create: `workspaces/alignment_prototype/cotrain.py`
- Test: `workspaces/alignment_prototype/tests/test_cotrain.py` (append)

**Interfaces:**
- Consumes: `build_examples`, `train_ensemble`, `TrainConfig` (all in `mert_model.py`); `MertSeries` (from `mert_store`).
- Produces:
  - `SetStores` frozen dataclass: `set_id: str`, `train_spans: tuple[SpanTarget, ...]`, `mix: MertSeries`, `refs: dict[str, MertSeries]`, `slot_pools: dict[str, tuple]`.
  - `cotrain(train_sets: list[SetStores], *, cfg: TrainConfig | None = None, device: str = "cpu", init=None) -> MertAlignHead | MertAlignEnsemble` — calls `build_examples` per set, concatenates, `train_ensemble(all_examples)`. Returns the head only.

- [ ] **Step 1: Write the failing test**

```python
# append to test_cotrain.py
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

    s1 = SetStores("a", ("x", "y"), None, {}, {})          # 2 spans
    s2 = SetStores("b", ("p", "q", "r"), None, {}, {})     # 3 spans
    head = cotrain([s1, s2], device="cpu")
    assert head == "HEAD"
    assert calls["n"] == 5  # 2 + 3 concatenated
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest workspaces/alignment_prototype/tests/test_cotrain.py::test_cotrain_concatenates_examples_across_sets -v`
Expected: FAIL — `cotrain` module does not exist.

- [ ] **Step 3: Implement `cotrain.py`**

```python
# workspaces/alignment_prototype/cotrain.py
"""Multi-set co-training: concatenate per-set MERT examples, train one head.

The head trains on set-agnostic materialized examples (`build_examples`), so
co-train is a concat + single train_ensemble. LOSO wraps the head per held-out
set (see train.py --loso)."""
from __future__ import annotations

from dataclasses import dataclass

from workspaces.alignment_prototype.mert_model import (
    TrainConfig,
    build_examples,
    train_ensemble,
)
from workspaces.alignment_prototype.records import SpanTarget


@dataclass(frozen=True)
class SetStores:
    set_id: str
    train_spans: tuple[SpanTarget, ...]
    mix: object  # MertSeries
    refs: dict
    slot_pools: dict


def cotrain(train_sets, *, cfg: TrainConfig | None = None, device: str = "cpu", init=None):
    cfg = cfg or TrainConfig()
    all_examples: list = []
    for s in train_sets:
        all_examples.extend(
            build_examples(
                s.train_spans, s.mix, s.refs, s.slot_pools,
                search_margin_s=cfg.search_margin_s,
            )
        )
    return train_ensemble(tuple(all_examples), cfg=cfg, device=device, init=init)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest workspaces/alignment_prototype/tests/test_cotrain.py -v`
Expected: PASS (all cotrain tests).

- [ ] **Step 5: Commit**

```bash
git add workspaces/alignment_prototype/cotrain.py workspaces/alignment_prototype/tests/test_cotrain.py
git commit -m "feat(cotrain): SetStores + cotrain — concat per-set examples, one head"
```

---

### Task 3: LOSO driver + `train.py --loso`

**Files:**
- Modify: `workspaces/alignment_prototype/cotrain.py` (add `run_loso`)
- Modify: `workspaces/alignment_prototype/train.py` (CLI `--loso`)
- Test: `workspaces/alignment_prototype/tests/test_cotrain.py` (append)

**Interfaces:**
- Consumes: `cotrain`, `SetStores`, `load_set` (dataset), `load_bb12_mert` (mert_store), `slot_candidates_from_targets` (dataset), `MertLearnedAligner` + `median_start_by_label` (slot_priors) + `median_duration_by_slot` (mert_model), `evaluate` (eval).
- Produces: `run_loso(set_yamls: dict[str, Path], *, cfg=None, device="cpu", anchor_from_cues=True) -> dict[str, object]` returning a per-held-out-set report dict. For each held-out set: cotrain on the others, wrap the head around the held-out stores with the **scraped-cue anchor** (`anchor_sigma_s` set), `predict_sequence` on the held-out spans, `evaluate`.

**Anchor sourcing (the load-bearing correctness point):** the held-out set's
placement anchor must come from scraped cues, not its GT. `infer.py.fetch_slot_rows(set_id)`
already fetches `cue_seconds` per slot via SSH. Reuse it to build a
`{slot_label: cue_seconds}` map and pass that as the aligner's `train_medians`
(the anchor centre). If cues are unavailable for a set, fall back to
`anchor_sigma_s=None` and mark that set's result `"anchor": "none (floor)"` in the
report — an honest floor, never GT leakage.

- [ ] **Step 1: Write the failing test (loop wiring, fully monkeypatched — no pi/GPU)**

```python
# append to test_cotrain.py
def test_run_loso_holds_each_set_out(monkeypatch):
    import workspaces.alignment_prototype.cotrain as ctmod

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest workspaces/alignment_prototype/tests/test_cotrain.py::test_run_loso_holds_each_set_out -v`
Expected: FAIL — `run_loso` / `_load_set_stores` do not exist.

- [ ] **Step 3: Implement `_load_set_stores` + `run_loso` in `cotrain.py`**

```python
# add imports at top of cotrain.py
from pathlib import Path

# add below cotrain():
def _load_set_stores(set_id_or_yaml):
    """Load a set's (targets, mix, refs, slot_pools) into SetStores. Accepts a
    yaml Path; resolves set_id from the parsed GT. SSHes pi for MERT (cached)."""
    from core.result import Err, Ok
    from workspaces.alignment_prototype.dataset import (
        load_set, slot_candidates_from_targets,
    )
    from workspaces.alignment_prototype.mert_store import load_bb12_mert

    match load_set(Path(set_id_or_yaml)):
        case Err(msg):
            raise RuntimeError(f"load_set failed: {msg}")
        case Ok((gt, targets)):
            pass
    match load_bb12_mert(gt.set_id):
        case Err(msg):
            raise RuntimeError(f"MERT load failed for {gt.set_id}: {msg}")
        case Ok((_sid, mix, refs)):
            pass
    pools = slot_candidates_from_targets(targets)
    return SetStores(gt.set_id, targets, mix, refs, pools)


def _cue_anchor(set_id):
    """{slot_label: cue_seconds} from scraped tracklist cues (aligner INPUT, not
    GT). Returns {} if unavailable."""
    try:
        from workspaces.alignment_prototype.infer import fetch_slot_rows
        rows = fetch_slot_rows(set_id)
        out = {}
        for r in rows:
            cue = r.get("cue_seconds") if isinstance(r, dict) else None
            if cue not in (None, ""):
                out[r["slot_label"]] = float(cue)
        return out
    except Exception:  # noqa: BLE001
        return {}


def run_loso(set_yamls, *, cfg=None, device="cpu", anchor_from_cues=True):
    from workspaces.alignment_prototype.eval import evaluate
    from workspaces.alignment_prototype.mert_model import (
        MertLearnedAligner, median_duration_by_slot,
    )

    stores = {sid: _load_set_stores(y) for sid, y in set_yamls.items()}
    report = {}
    for held in stores:
        train_sets = [s for sid, s in stores.items() if sid != held]
        head = cotrain(train_sets, cfg=cfg, device=device)
        h = stores[held]
        cue_anchor = _cue_anchor(held) if anchor_from_cues else {}
        aligner = MertLearnedAligner(
            head=head, mix=h.mix, refs=h.refs,
            slot_medians=median_duration_by_slot(h.train_spans),
            slot_pools=h.slot_pools,
            train_medians=(cue_anchor or {}),
            anchor_sigma_s=(30.0 if cue_anchor else None),
            device=device,
        )
        preds = aligner.predict_sequence(h.train_spans)
        rep = evaluate(preds, h.train_spans)
        report[held] = {
            "anchor": "cues" if cue_anchor else "none (floor)",
            "lines": list(rep.lines()),
        }
    return report
```

(If `MertLearnedAligner`/`evaluate` must be monkeypatchable by the test, import
them at module top too — the test patches `ctmod.MertLearnedAligner` /
`ctmod.evaluate` with `raising=False`, so a module-level name is preferred; add
`from ...mert_model import MertLearnedAligner` and `from ...eval import evaluate`
at the top and drop the local imports.)

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest workspaces/alignment_prototype/tests/test_cotrain.py -v`
Expected: PASS (all cotrain tests).

- [ ] **Step 5: Wire `train.py --loso`**

In `train.py` `main`, add args and a branch (do NOT disturb the existing `--eval` path):

```python
    p.add_argument("--loso", action="store_true",
                   help="Leave-one-set-out co-train eval over --sets")
    p.add_argument("--sets", default="bb11,bb12",
                   help="comma-separated fixture stems in labeling/fixtures/")
```

Near the top of `main` (after parsing), before the single-set `load_set` path:

```python
    if args.loso:
        from workspaces.alignment_prototype.cotrain import run_loso
        from workspaces.alignment_prototype.mert_model import TrainConfig
        fixtures = _REPO / "labeling/fixtures"
        yamls = {s: fixtures / f"{s}_ground_truth.yaml" for s in args.sets.split(",")}
        rep = run_loso(yamls, cfg=TrainConfig(epochs=40, search_margin_s=90.0),
                       device=_torch_device())
        for sid, r in rep.items():
            print(f"\n=== LOSO held-out {sid} (anchor={r['anchor']}) ===")
            for line in r["lines"]:
                print(f"  {line}")
        return 0
```

- [ ] **Step 6: Run the deliverable (offline, real MERT cache — needs bb11+bb12 stores or pi)**

Run:

```bash
venvs/audio/bin/python -m workspaces.alignment_prototype.train --loso --sets bb11,bb12
```

Expected: two blocks — held-out bb11 and held-out bb12 — each printing set_start median / `<Xs` / identity from `evaluate`, with the anchor source noted. This is the first honest cross-set generalization number. Record the printed numbers in the commit body. If MERT stores are missing and pi is unreachable, report BLOCKED with the error (do not fake numbers).

- [ ] **Step 7: Commit**

```bash
git add workspaces/alignment_prototype/cotrain.py workspaces/alignment_prototype/train.py \
        workspaces/alignment_prototype/tests/test_cotrain.py
git commit -m "feat(cotrain): run_loso + train.py --loso — first cross-set generalization number

<paste the held-out bb11 / bb12 numbers here>"
```

---

## Self-Review

- **Spec coverage:** change 1 (set_id) = Task 1; change 2 (cotrain) = Task 2; change 3 (LOSO driver + --loso) = Task 3. The cross-set anchor-from-cues requirement = Task 3 anchor sourcing + fallback. Unit tests avoid pi/GPU (monkeypatched). The offline LOSO run = Task 3 Step 6 deliverable. ✓
- **Placeholder scan:** every code step has complete code; Task 1 Step 3 and Task 3 Step 3's parenthetical "read the body / prefer module-level import" are precise refactor guidance with the exact lines to add, not placeholders. The commit-body number paste is a data step (forbids faking), not a code gap. ✓
- **Type consistency:** `SetStores(set_id, train_spans, mix, refs, slot_pools)` constructed identically in Task 2 test, Task 3 `_load_set_stores`, and Task 3 test. `cotrain(train_sets, *, cfg, device, init)` and `run_loso(set_yamls, *, cfg, device, anchor_from_cues)` signatures match across tasks. `SpanTarget.set_id` default `""` consistent. ✓
