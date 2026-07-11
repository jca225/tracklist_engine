# Alignment Ablation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible ablation harness that runs every method/toggle through ONE scorer, stores per-span results in one long-format table, and emits paper tables with span-level bootstrap CIs.

**Architecture:** A new `workspaces/alignment_prototype/experiments/` package generalizes the existing `drivers/race.py`: a declarative `Cell` matrix `{driver-config × set}` → each driver's existing `align_set(ctx)` produces a timeline (cached) → a single extracted `score_spans()` yields per-span rows → a sqlite long store → a report with paired span-bootstrap CIs. Reuses all drivers and the scorer; the only refactor to existing code is extracting the scorer's inline per-span loop into a reusable `score_spans()` (DRY — a second copy would be the metric drift we're eliminating).

**Tech Stack:** Python 3 (repo style: `from __future__ import annotations`, frozen dataclasses, full type hints), numpy, sqlite3 (stdlib — no pyarrow/pandas in `venvs/audio`), pytest.

## Global Constraints

- Interpreter: `venvs/audio/bin/python`; run/import from repo root.
- **Set ids (verified — do NOT swap): BB11 = `2nvzlh2k` (Episode 11), BB12 = `1fsnxchk` (Volume 12).**
- **Single scorer:** only `score_spans()` in `score_timeline_vs_gt.py` supplies headline metrics. No other trajectory/accuracy function (e.g. a duplicated loop) may feed the store or report.
- **Results store: sqlite** (`experiments/results/scores.db`). No parquet/pandas dependency.
- **CIs: span-level bootstrap only** (resample the ~300 spans, fixed seed `np.random.default_rng(0)`). No set-level CI anywhere (n=2 GT sets).
- **Cross-set generalization = a LOSO row**, sourced from `cotrain.run_loso`, never a CI.
- **fiber-aware − strict is a standing column** in every trajectory table.
- **No driver/probe rewrites.** Drivers are invoked via their existing config kwargs only. Decoder toggle values are exactly `"looptrace"` / `"legacy"` (`joint_ref_decode.py:100-101`).
- Style: frozen dataclasses for records; keep I/O at edges; small focused files.
- Commit after each task with `git add <files> && git commit`. `make check` must stay green.

---

## File Structure

```
workspaces/alignment_prototype/
  score_timeline_vs_gt.py        # MODIFY: add SpanScore + score_spans(); main() consumes it
  experiments/
    __init__.py                  # CREATE (empty)
    matrix.py                    # CREATE: Cell, cell_hash, PAPER matrix
    store.py                     # CREATE: sqlite long-format results store
    run.py                       # CREATE: cell → timeline (cached) → score_spans → store
    report.py                    # CREATE: headline + ablation tables, bootstrap CIs, markdown
    cli.py                       # CREATE: `python -m ...experiments.cli` entrypoint
tests/alignment_prototype/
  test_experiments_matrix.py     # CREATE
  test_experiments_store.py      # CREATE
  test_experiments_report.py     # CREATE
  test_experiments_run.py        # CREATE (stub driver; no heavy inference)
  test_score_spans.py            # CREATE (golden-output guard for the refactor)
Makefile                         # MODIFY: add `align-ablate` target
```

---

## Task 1: Extract `score_spans()` from the scorer (the one refactor)

**Files:**
- Modify: `workspaces/alignment_prototype/score_timeline_vs_gt.py`
- Test: `tests/alignment_prototype/test_score_spans.py`

**Interfaces:**
- Produces: `SpanScore` (frozen dataclass) and `score_spans(set_id: str, timeline_path: Path, *, fibers: bool = False, hubert_layer: int = 9, gt_path: Path | None = None) -> list[SpanScore]`.

`SpanScore` fields (one row per timeline span; `None` where not computed):

```python
@dataclass(frozen=True)
class SpanScore:
    slot: str
    recording_id: str
    stem: str | None          # matched-GT claimed_stem (axis), None if no same-rec GT
    span_class: str | None    # linear/multiseg/loop/oddratio, None if no same-rec GT
    id_correct: bool | None   # None if no overlapping GT row
    place_err_s: float | None
    strict: float | None
    fiber: float | None
    ref_err_s: float | None   # straight clips only
    density: int | None
```

- [ ] **Step 1: Capture the current output as a golden file**

The refactor must not change `main()`'s output. Pick a set whose `_lt` timeline exists locally (check `workspaces/alignment_prototype/out/`). Run WITHOUT `--fibers` (no audio needed):

Run:
```bash
cd /Users/johnnycabrahams/Desktop/tracklist_engine
ls workspaces/alignment_prototype/out/*_predicted_timeline*.json
venvs/audio/bin/python -m workspaces.alignment_prototype.score_timeline_vs_gt \
  --set-id 1fsnxchk \
  --timeline workspaces/alignment_prototype/out/1fsnxchk_predicted_timeline_lt.json \
  --decompose > /tmp/score_golden_1fsnxchk.txt 2>&1
cat /tmp/score_golden_1fsnxchk.txt
```
Expected: the scorecard prints (identity, placement, trajectory, decomposition). If the `_lt` timeline is absent, use whatever `out/1fsnxchk_predicted_timeline*.json` exists and adjust the path. Save the exact path used.

- [ ] **Step 2: Write the failing test**

```python
# tests/alignment_prototype/test_score_spans.py
from __future__ import annotations
from pathlib import Path
import pytest
from workspaces.alignment_prototype.score_timeline_vs_gt import score_spans, SpanScore

REPO = Path(__file__).resolve().parents[2]
TL = REPO / "workspaces/alignment_prototype/out/1fsnxchk_predicted_timeline_lt.json"

@pytest.mark.skipif(not TL.exists(), reason="needs local _lt timeline")
def test_score_spans_returns_row_per_span():
    rows = score_spans("1fsnxchk", TL)  # fibers=False → no audio needed
    assert rows and all(isinstance(r, SpanScore) for r in rows)
    import json
    n_spans = len(json.loads(TL.read_text())["spans"])
    assert len(rows) == n_spans          # one row per timeline span
    # strict is populated for spans with a same-recording GT row
    assert any(r.strict is not None for r in rows)
    # fiber == strict when fibers=False (trajectory_acc contract)
    assert all(r.fiber == r.strict for r in rows if r.strict is not None)
```

Run: `venvs/audio/bin/python -m pytest tests/alignment_prototype/test_score_spans.py -v`
Expected: FAIL with `ImportError: cannot import name 'score_spans'`.

- [ ] **Step 3: Add `SpanScore` and `score_spans()`, refactor `main()` to consume it**

In `score_timeline_vs_gt.py`:
1. Add the `SpanScore` dataclass (above) near the top (after imports; add `from dataclasses import dataclass`).
2. Extract the per-span loop (currently `score_timeline_vs_gt.py:245-348`) into `score_spans()`. Move the GT-resolution/join-guard/id-map/`gt_by_tid` setup (lines 168-243) into `score_spans()` too, so it is self-contained. The function returns one `SpanScore` per `timeline["spans"]`:
   - identity: set `id_correct` from the overlapping-GT block (lines 253-271) — `True`/`False` when `overlapping` non-empty, else `None`.
   - `rows = gt_by_tid.get(recording_id)`: if absent, emit `SpanScore(slot, recording_id, stem=None, span_class=None, id_correct, place_err_s=None, strict=None, fiber=None, ref_err_s=None, density=None)` and continue.
   - else compute `gstem`, `place_err_s`, `(strict,_,facc)` via `trajectory_acc`, `density`, and `ref_err_s` (the straight-clip branch, lines 331-348; `None` when loop/segment/tempo-excluded).
3. Rewrite `main()` to call `score_spans(args.set_id, tl_path, fibers=args.fibers, hubert_layer=args.hubert_layer, gt_path=args.gt)` and reconstruct the existing aggregate prints from the returned list (identity %, placement percentiles, ref-offset stats, the trajectory tables, `--decompose`, worst-lists). The printed numbers MUST be identical to the golden file. Keep helper fns (`_span_class`, `_pred_segs_from_span`, `_decompose_span`, `trajectory_acc`) as-is.

- [ ] **Step 4: Verify golden output unchanged**

Run:
```bash
venvs/audio/bin/python -m workspaces.alignment_prototype.score_timeline_vs_gt \
  --set-id 1fsnxchk \
  --timeline workspaces/alignment_prototype/out/1fsnxchk_predicted_timeline_lt.json \
  --decompose > /tmp/score_after_1fsnxchk.txt 2>&1
diff /tmp/score_golden_1fsnxchk.txt /tmp/score_after_1fsnxchk.txt && echo "IDENTICAL"
```
Expected: `IDENTICAL` (empty diff). Then run the pytest from Step 2 — expected PASS.

- [ ] **Step 5: Commit**

```bash
git add workspaces/alignment_prototype/score_timeline_vs_gt.py tests/alignment_prototype/test_score_spans.py
git commit -m "refactor(scorer): extract score_spans() per-span rows; main() consumes it"
```

---

## Task 2: `matrix.py` — cells, hashing, PAPER matrix

**Files:**
- Create: `workspaces/alignment_prototype/experiments/__init__.py` (empty), `workspaces/alignment_prototype/experiments/matrix.py`
- Test: `tests/alignment_prototype/test_experiments_matrix.py`

**Interfaces:**
- Produces: `Cell` (frozen dataclass), `cell_hash(cell) -> str`, `PAPER: tuple[Cell, ...]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignment_prototype/test_experiments_matrix.py
from __future__ import annotations
from workspaces.alignment_prototype.experiments.matrix import Cell, cell_hash, PAPER

def test_cell_hash_stable_and_order_independent():
    a = Cell(driver="classical", set_id="1fsnxchk", decoder="looptrace")
    b = Cell(driver="classical", set_id="1fsnxchk", decoder="looptrace")
    assert cell_hash(a) == cell_hash(b)
    assert cell_hash(a) != cell_hash(Cell(driver="classical", set_id="1fsnxchk", decoder="legacy"))

def test_paper_matrix_covers_c4_and_both_sets():
    sets = {c.set_id for c in PAPER}
    assert sets == {"2nvzlh2k", "1fsnxchk"}          # BB11, BB12
    # C4 ablation present: classical looptrace vs legacy on each set
    for sid in sets:
        decoders = {c.decoder for c in PAPER if c.driver == "classical" and c.set_id == sid}
        assert {"looptrace", "legacy"} <= decoders
    # all three drivers represented
    assert {c.driver for c in PAPER} == {"classical", "agentic", "ml"}
```

Run: `venvs/audio/bin/python -m pytest tests/alignment_prototype/test_experiments_matrix.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 2: Implement `matrix.py`**

```python
# workspaces/alignment_prototype/experiments/matrix.py
"""Declarative ablation matrix: one Cell = one (driver-config, set) to run+score.

An ablation is a pair of cells differing by exactly one field. Sets:
BB11=2nvzlh2k (Episode 11), BB12=1fsnxchk (Volume 12).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict

BB11 = "2nvzlh2k"
BB12 = "1fsnxchk"


@dataclass(frozen=True)
class Cell:
    driver: str                       # "classical" | "agentic" | "ml"
    set_id: str
    decoder: str = "looptrace"        # classical only: "looptrace" | "legacy"
    ml_gate: bool = True              # ml only: gated vs ungated decode
    live: bool = True                 # agentic only

    @property
    def label(self) -> str:
        extra = {
            "classical": self.decoder,
            "ml": "gated" if self.ml_gate else "ungated",
            "agentic": "live" if self.live else "replay",
        }[self.driver]
        return f"{self.driver}:{extra}"


def cell_hash(cell: Cell) -> str:
    payload = repr(sorted(asdict(cell).items()))
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def _per_set(sid: str) -> tuple[Cell, ...]:
    return (
        Cell("classical", sid, decoder="looptrace"),   # baseline
        Cell("classical", sid, decoder="legacy"),      # C4 ablation
        Cell("agentic", sid),
        Cell("ml", sid, ml_gate=True),
        Cell("ml", sid, ml_gate=False),
    )


PAPER: tuple[Cell, ...] = _per_set(BB11) + _per_set(BB12)
```

- [ ] **Step 3: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/alignment_prototype/test_experiments_matrix.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add workspaces/alignment_prototype/experiments/__init__.py workspaces/alignment_prototype/experiments/matrix.py tests/alignment_prototype/test_experiments_matrix.py
git commit -m "feat(experiments): ablation matrix (Cell, cell_hash, PAPER)"
```

---

## Task 3: `store.py` — sqlite long-format results store

**Files:**
- Create: `workspaces/alignment_prototype/experiments/store.py`
- Test: `tests/alignment_prototype/test_experiments_store.py`

**Interfaces:**
- Consumes: `SpanScore` (Task 1), `Cell`/`cell_hash` (Task 2).
- Produces: `Store(db_path: Path)` with `.upsert(cell: Cell, rows: list[SpanScore]) -> None` and `.fetch(*, driver=None, set_id=None) -> list[dict]` (long rows: one dict per span×cell with cell fields + SpanScore fields flattened).

- [ ] **Step 1: Write the failing test**

```python
# tests/alignment_prototype/test_experiments_store.py
from __future__ import annotations
from pathlib import Path
from workspaces.alignment_prototype.experiments.store import Store
from workspaces.alignment_prototype.experiments.matrix import Cell
from workspaces.alignment_prototype.score_timeline_vs_gt import SpanScore

def _row(strict):
    return SpanScore("6", "recX", "acappella", "multiseg", True, 3.0, strict, strict, None, 2)

def test_upsert_is_idempotent_and_fetch_filters(tmp_path: Path):
    s = Store(tmp_path / "scores.db")
    c = Cell("classical", "1fsnxchk", decoder="looptrace")
    s.upsert(c, [_row(0.4), _row(0.5)])
    s.upsert(c, [_row(0.4), _row(0.5)])          # re-run: no duplication
    got = s.fetch(set_id="1fsnxchk")
    assert len(got) == 2
    assert got[0]["driver"] == "classical" and got[0]["decoder"] == "looptrace"
    assert {r["strict"] for r in got} == {0.4, 0.5}
    assert s.fetch(driver="agentic") == []
```

Run: `venvs/audio/bin/python -m pytest tests/alignment_prototype/test_experiments_store.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 2: Implement `store.py`**

```python
# workspaces/alignment_prototype/experiments/store.py
"""Long-format sqlite results store: one row per (cell × span). Tidy — every
paper table is a GROUP BY. Idempotent on (cell_hash, slot, recording_id)."""
from __future__ import annotations

import sqlite3
from dataclasses import asdict
from pathlib import Path

from workspaces.alignment_prototype.experiments.matrix import Cell, cell_hash
from workspaces.alignment_prototype.score_timeline_vs_gt import SpanScore

_COLS = [
    "cell_hash", "driver", "set_id", "decoder", "ml_gate", "live",
    "slot", "recording_id", "stem", "span_class",
    "id_correct", "place_err_s", "strict", "fiber", "ref_err_s", "density",
]


class Store:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as cx:
            cols = ", ".join(f"{c}" for c in _COLS)
            cx.execute(f"CREATE TABLE IF NOT EXISTS scores ({cols})")
            cx.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_span "
                "ON scores (cell_hash, slot, recording_id)"
            )

    def _conn(self) -> sqlite3.Connection:
        cx = sqlite3.connect(self.db_path)
        cx.row_factory = sqlite3.Row
        return cx

    def upsert(self, cell: Cell, rows: list[SpanScore]) -> None:
        h = cell_hash(cell)
        cd = asdict(cell)
        placeholders = ", ".join("?" for _ in _COLS)
        with self._conn() as cx:
            for r in rows:
                rd = asdict(r)
                vals = [
                    h, cd["driver"], cd["set_id"], cd["decoder"],
                    int(cd["ml_gate"]), int(cd["live"]),
                    rd["slot"], rd["recording_id"], rd["stem"], rd["span_class"],
                    (None if rd["id_correct"] is None else int(rd["id_correct"])),
                    rd["place_err_s"], rd["strict"], rd["fiber"],
                    rd["ref_err_s"], rd["density"],
                ]
                cx.execute(
                    f"INSERT OR REPLACE INTO scores ({', '.join(_COLS)}) "
                    f"VALUES ({placeholders})",
                    vals,
                )

    def fetch(self, *, driver: str | None = None, set_id: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM scores", []
        clauses = []
        if driver is not None:
            clauses.append("driver = ?"); args.append(driver)
        if set_id is not None:
            clauses.append("set_id = ?"); args.append(set_id)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        with self._conn() as cx:
            return [dict(r) for r in cx.execute(q, args).fetchall()]
```

- [ ] **Step 3: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/alignment_prototype/test_experiments_store.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add workspaces/alignment_prototype/experiments/store.py tests/alignment_prototype/test_experiments_store.py
git commit -m "feat(experiments): sqlite long-format results store"
```

---

## Task 4: `run.py` — cell → timeline (cached) → score → store

**Files:**
- Create: `workspaces/alignment_prototype/experiments/run.py`
- Test: `tests/alignment_prototype/test_experiments_run.py`

**Interfaces:**
- Consumes: `Cell` (Task 2), `Store` (Task 3), `score_spans` (Task 1), the existing drivers (`ClassicalDriver`, `AgenticDriver`, `HybridMlDriver`) and `SetContext.for_set` (`drivers/base.py:56`).
- Produces: `build_driver(cell, base_timeline: Path | None)` (returns an object with `.align_set(ctx) -> Path`), and `run_cell(cell, store, *, fibers, driver_factory=build_driver, base_timeline=None) -> int` (returns rows written). The `driver_factory` seam lets tests inject a stub — no heavy inference in unit tests.

- [ ] **Step 1: Write the failing test (stub driver, no inference)**

```python
# tests/alignment_prototype/test_experiments_run.py
from __future__ import annotations
import json
from pathlib import Path
from workspaces.alignment_prototype.experiments.run import run_cell
from workspaces.alignment_prototype.experiments.matrix import Cell
from workspaces.alignment_prototype.experiments.store import Store

class _StubDriver:
    calls = 0
    def __init__(self, tl: Path): self._tl = tl
    def align_set(self, ctx):
        _StubDriver.calls += 1
        return self._tl

def test_run_cell_scores_and_caches(tmp_path, monkeypatch):
    # a minimal timeline JSON with one span that has no same-rec GT → strict None,
    # but score_spans still returns one SpanScore row.
    tl = tmp_path / "tl.json"
    tl.write_text(json.dumps({"sid": "1fsnxchk", "spans": [
        {"slot_label": "6", "recording_id": "nope", "set_start_s": 1.0,
         "set_end_s": 9.0, "name": "x", "ref_start_s": 0.0, "claimed_stem": "regular"}
    ]}))
    store = Store(tmp_path / "s.db")
    cell = Cell("classical", "1fsnxchk", decoder="looptrace")
    factory = lambda c, base: _StubDriver(tl)
    n1 = run_cell(cell, store, fibers=False, driver_factory=factory)
    assert n1 == 1 and len(store.fetch(set_id="1fsnxchk")) == 1
    # second run hits the timeline cache: driver not called again
    before = _StubDriver.calls
    run_cell(cell, store, fibers=False, driver_factory=factory)
    assert _StubDriver.calls == before        # cached, no re-inference
```

Run: `venvs/audio/bin/python -m pytest tests/alignment_prototype/test_experiments_run.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 2: Implement `run.py`**

```python
# workspaces/alignment_prototype/experiments/run.py
"""Run a Cell: build its driver (with toggles), produce a timeline (cached by
cell_hash), score via the single scorer, write per-span rows to the store.

Reuse only — drivers are invoked through their existing config kwargs. Agentic
and ml cells need a classical base timeline (their refinement source), same as
drivers/race.py."""
from __future__ import annotations

import shutil
from pathlib import Path

from workspaces.alignment_prototype.drivers.base import SetContext
from workspaces.alignment_prototype.drivers.classical import ClassicalDriver
from workspaces.alignment_prototype.drivers.agentic import AgenticDriver
from workspaces.alignment_prototype.drivers.ml import HybridMlDriver
from workspaces.alignment_prototype.experiments.matrix import Cell, cell_hash
from workspaces.alignment_prototype.experiments.store import Store
from workspaces.alignment_prototype.score_timeline_vs_gt import score_spans

_CACHE = Path(__file__).resolve().parent / "cache"


def build_driver(cell: Cell, base_timeline: Path | None):
    if cell.driver == "classical":
        return ClassicalDriver(decoder=cell.decoder)
    if cell.driver == "agentic":
        assert base_timeline is not None, "agentic needs a classical base"
        return AgenticDriver(base_timeline, live=cell.live)
    if cell.driver == "ml":
        assert base_timeline is not None, "ml needs a classical base"
        return HybridMlDriver(base_timeline, gate_margin=(0.0 if cell.ml_gate else None))
    raise ValueError(f"unknown driver {cell.driver!r}")


def _cached_timeline(cell: Cell) -> Path:
    return _CACHE / f"{cell_hash(cell)}.json"


def run_cell(
    cell: Cell,
    store: Store,
    *,
    fibers: bool,
    driver_factory=build_driver,
    base_timeline: Path | None = None,
) -> int:
    _CACHE.mkdir(parents=True, exist_ok=True)
    cached = _cached_timeline(cell)
    if not cached.exists():
        ctx = SetContext.for_set(cell.set_id)
        driver = driver_factory(cell, base_timeline)
        produced = Path(driver.align_set(ctx))
        shutil.copyfile(produced, cached)
    rows = score_spans(cell.set_id, cached, fibers=fibers)
    store.upsert(cell, rows)
    return len(rows)
```

Note: `gate_margin=0.0` = gated (a real threshold), `None` = ungated — matches `HybridMlDriver.__init__` (`ml.py:42`). If the driver treats `0.0` as falsy/ungated, the implementer must use a small positive margin (confirm by reading `ml.py`'s use of `gate_margin`); the C4/ml-gate distinction only needs the two cells to differ, so pick two values that the driver treats as gated vs ungated and record them.

- [ ] **Step 3: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/alignment_prototype/test_experiments_run.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add workspaces/alignment_prototype/experiments/run.py tests/alignment_prototype/test_experiments_run.py
git commit -m "feat(experiments): run_cell — cached timeline + single-scorer rows to store"
```

---

## Task 5: `report.py` — tables + span-bootstrap CIs

**Files:**
- Create: `workspaces/alignment_prototype/experiments/report.py`
- Test: `tests/alignment_prototype/test_experiments_report.py`

**Interfaces:**
- Consumes: long rows from `Store.fetch()` (list of dicts with `strict`/`fiber`/`stem`/`driver`/`decoder`/`set_id`).
- Produces: `mean_ci(values, *, seed=0, n=1000) -> tuple[float, float, float]` (mean, lo, hi); `paired_delta_ci(a, b, *, seed=0, n=1000) -> tuple[float, float, float]`; `headline_table(rows) -> str` (markdown, strict + fiber + gap columns); `ablation_table(rows, field, left, right) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/alignment_prototype/test_experiments_report.py
from __future__ import annotations
from workspaces.alignment_prototype.experiments.report import (
    mean_ci, paired_delta_ci, headline_table,
)

def test_mean_ci_brackets_the_mean():
    m, lo, hi = mean_ci([0.0, 0.5, 1.0, 0.5], seed=0)
    assert lo <= m <= hi
    assert abs(m - 0.5) < 1e-9

def test_paired_delta_ci_sign_and_seed_stability():
    a = [0.9, 0.8, 0.85, 0.95]      # "with"
    b = [0.4, 0.3, 0.35, 0.45]      # "without"
    d, lo, hi = paired_delta_ci(a, b, seed=0)
    assert d > 0 and lo > 0          # clearly positive delta, CI excludes 0
    assert (d, lo, hi) == paired_delta_ci(a, b, seed=0)   # deterministic

def test_headline_has_strict_fiber_and_gap():
    rows = [
        {"set_id": "2nvzlh2k", "driver": "classical", "decoder": "looptrace",
         "stem": "acappella", "strict": 0.12, "fiber": 0.31},
        {"set_id": "2nvzlh2k", "driver": "classical", "decoder": "looptrace",
         "stem": "acappella", "strict": 0.10, "fiber": 0.29},
    ]
    md = headline_table(rows)
    assert "strict" in md and "fiber" in md and "gap" in md
    assert "acappella" in md
```

Run: `venvs/audio/bin/python -m pytest tests/alignment_prototype/test_experiments_report.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 2: Implement `report.py`**

```python
# workspaces/alignment_prototype/experiments/report.py
"""Paper tables from the long store. CIs are span-level bootstrap ONLY (n=2 sets
forbids a set-level CI). Every trajectory table carries the fiber − strict gap."""
from __future__ import annotations

import numpy as np


def _clean(values) -> np.ndarray:
    return np.array([v for v in values if v is not None], dtype=float)


def mean_ci(values, *, seed: int = 0, n: int = 1000) -> tuple[float, float, float]:
    v = _clean(values)
    if v.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = [v[rng.integers(0, v.size, v.size)].mean() for _ in range(n)]
    return float(v.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def paired_delta_ci(a, b, *, seed: int = 0, n: int = 1000) -> tuple[float, float, float]:
    """Paired bootstrap of mean(a) − mean(b) over a shared span index."""
    av, bv = _clean(a), _clean(b)
    m = min(av.size, bv.size)
    av, bv = av[:m], bv[:m]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        boots.append(av[idx].mean() - bv[idx].mean())
    return float(av.mean() - bv.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _pct(x: float) -> str:
    return "—" if x != x else f"{100 * x:.0f}%"


def headline_table(rows: list[dict]) -> str:
    """Per set × stem: strict, fiber-aware, and the gap (the which-instance
    residual). Uses only baseline classical/looptrace rows."""
    base = [r for r in rows if r["driver"] == "classical" and r.get("decoder") == "looptrace"]
    sets = sorted({r["set_id"] for r in base})
    stems = ("acappella", "regular", "instrumental")
    out = ["| set | stem | strict | fiber-aware | gap (fiber−strict) |",
           "|---|---|---|---|---|"]
    for sid in sets:
        for stem in stems:
            sub = [r for r in base if r["set_id"] == sid and r["stem"] == stem]
            if not sub:
                continue
            sm, _, _ = mean_ci([r["strict"] for r in sub])
            fm, _, _ = mean_ci([r["fiber"] for r in sub])
            gap = "—" if (sm != sm or fm != fm) else f"+{100 * (fm - sm):.0f}pp"
            out.append(f"| {sid} | {stem} | {_pct(sm)} | {_pct(fm)} | {gap} |")
    return "\n".join(out)


def ablation_table(rows: list[dict], field: str, left, right, *, metric: str = "strict") -> str:
    """One-toggle ablation: mean(left) vs mean(right) on `field`, with a paired
    span-bootstrap CI on the delta."""
    a = [r[metric] for r in rows if r.get(field) == left]
    b = [r[metric] for r in rows if r.get(field) == right]
    am, _, _ = mean_ci(a)
    bm, _, _ = mean_ci(b)
    d, lo, hi = paired_delta_ci(a, b)
    return (f"| {field}={left} vs {right} ({metric}) | {_pct(am)} | {_pct(bm)} "
            f"| {100 * d:+.1f}pp [{100 * lo:+.1f}, {100 * hi:+.1f}] |")
```

- [ ] **Step 3: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/alignment_prototype/test_experiments_report.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add workspaces/alignment_prototype/experiments/report.py tests/alignment_prototype/test_experiments_report.py
git commit -m "feat(experiments): report tables + span-bootstrap CIs"
```

---

## Task 6: `cli.py` + `make align-ablate` + guard/smoke

**Files:**
- Create: `workspaces/alignment_prototype/experiments/cli.py`
- Modify: `Makefile`
- Test: extend `tests/alignment_prototype/test_experiments_report.py` with a guard test (or a new `test_experiments_guard.py`).

**Interfaces:**
- Consumes: everything above.
- Produces: `python -m workspaces.alignment_prototype.experiments.cli [--fibers] [--matrix paper]` → runs every PAPER cell through `run_cell`, then prints the headline + the C4 and driver ablations. Classical base timelines are computed once per set and reused for agentic/ml cells.

- [ ] **Step 1: Write the guard test (single scorer)**

```python
# tests/alignment_prototype/test_experiments_guard.py
from __future__ import annotations
import ast
from pathlib import Path

EX

P = Path(__file__).resolve().parents[2] / "workspaces/alignment_prototype/experiments"

def test_only_score_spans_supplies_metrics():
    """No experiments module may import the raw trajectory_acc or a duplicate
    scorer — headline metrics come solely through score_spans()."""
    banned = {"trajectory_acc"}
    for py in P.glob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = {a.name for a in node.names}
                assert not (names & banned), f"{py.name} imports {names & banned}"
```

(Delete the stray `EX` placeholder line — it must not appear; the real file has only the imports shown.)

Run: `venvs/audio/bin/python -m pytest tests/alignment_prototype/test_experiments_guard.py -v`
Expected: FAIL to parse until you remove the `EX` line, then PASS (no experiments module imports `trajectory_acc`).

- [ ] **Step 2: Implement `cli.py`**

```python
# workspaces/alignment_prototype/experiments/cli.py
"""Run the PAPER ablation matrix → store → print headline + ablation tables.

Base classical timelines are computed once per set and reused for agentic/ml
(their refinement source), mirroring drivers/race.py."""
from __future__ import annotations

import argparse
from pathlib import Path

from workspaces.alignment_prototype.experiments.matrix import PAPER, Cell
from workspaces.alignment_prototype.experiments.run import run_cell, build_driver, _cached_timeline
from workspaces.alignment_prototype.experiments.store import Store
from workspaces.alignment_prototype.experiments import report

_RESULTS = Path(__file__).resolve().parent / "results" / "scores.db"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fibers", action="store_true", help="fiber-aware scoring (needs audio)")
    p.add_argument("--matrix", default="paper")
    args = p.parse_args(argv)

    store = Store(_RESULTS)
    # ensure a classical/looptrace base per set exists first (reused downstream)
    bases: dict[str, Path] = {}
    for sid in sorted({c.set_id for c in PAPER}):
        base_cell = Cell("classical", sid, decoder="looptrace")
        run_cell(base_cell, store, fibers=args.fibers)
        bases[sid] = _cached_timeline(base_cell)

    for cell in PAPER:
        base = bases[cell.set_id] if cell.driver in ("agentic", "ml") else None
        run_cell(cell, store, fibers=args.fibers, base_timeline=base)

    rows = store.fetch()
    print("\n## Headline (baseline classical/looptrace)\n")
    print(report.headline_table(rows))
    print("\n## Ablations (paired span-bootstrap CI on the delta)\n")
    print("| ablation | left | right | Δ [95% CI] |")
    print("|---|---|---|---|")
    print(report.ablation_table(rows, "decoder", "looptrace", "legacy"))     # C4
    print(report.ablation_table(rows, "driver", "agentic", "classical"))
    print(report.ablation_table(rows, "driver", "ml", "classical"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Add the Makefile target**

Add to `Makefile` (near the `race:` target):

```makefile
align-ablate:
	venvs/audio/bin/python -m workspaces.alignment_prototype.experiments.cli $(EXTRA)
```

- [ ] **Step 4: End-to-end smoke (opt-in — needs real audio/DB; may run on pi)**

This runs real drivers; it is NOT a unit test. Run manually when data is available:
```bash
make align-ablate EXTRA=""      # strict only; add EXTRA=--fibers when audio present
```
Expected: a headline table and three ablation rows, including the C4 (looptrace vs legacy) delta with a bootstrap CI. If a driver errors on missing pi-storage inputs, that is the known data-availability risk — run on pi or with a set whose audio is local. Do NOT fabricate rows.

- [ ] **Step 5: Run the guard + full fast suite, then commit**

```bash
venvs/audio/bin/python -m pytest tests/alignment_prototype/ -v
make check
git add workspaces/alignment_prototype/experiments/cli.py Makefile tests/alignment_prototype/test_experiments_guard.py
git commit -m "feat(experiments): cli + make align-ablate + single-scorer guard"
```

---

## Self-Review

**Spec coverage:** matrix (Task 2) ✓; single scorer via `score_spans` (Task 1) + guard (Task 6) ✓; long store (Task 3) ✓; run+cache (Task 4) ✓; report + span-bootstrap CIs + fiber−strict gap (Task 5) ✓; `align-ablate` target (Task 6) ✓; C4 as the first ablation (Task 6 cli) ✓. **Deferred, flagged:** oracle-condition rows and the LOSO/UnmixDB external rows are NOT in this first cut — the trajectory metric is unified (`trajectory_acc` underlies both `score_spans` and `path_decode --eval`), so they can be appended as annotated rows in a follow-up without touching the store schema. Recorded so it isn't mistaken for full coverage.

**Placeholder scan:** one intentional `EX` line is called out and instructed-to-remove in Task 6 Step 1; no other placeholders.

**Type consistency:** `SpanScore` fields flow unchanged through `store._COLS` and into `report` dict keys (`strict`, `fiber`, `stem`, `driver`, `decoder`, `set_id`). `Cell` fields (`driver`, `set_id`, `decoder`, `ml_gate`, `live`) match `store` columns and `build_driver` usage. `gate_margin` semantics flagged in Task 4 Step 2 for the implementer to confirm against `ml.py`.
