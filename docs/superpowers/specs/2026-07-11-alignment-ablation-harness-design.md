# Alignment Ablation Harness — Design Spec

**Date:** 2026-07-11
**Status:** approved design, pre-implementation
**Approach:** B (pragmatic) — unify on the harness contract; one scorer, probes as
toggles; reuse existing drivers rather than rewrite.

---

## 1. Purpose

Produce the reproducible experiment infrastructure that backs a publishable
alignment paper. The paper's spine is a **novel methodology** (audio + tracklist →
Ableton-round-trippable structure) plus a **systematic ablation** of every method
tried (classical / agentic / ml drivers; fp / HuBERT / MERT / chroma / lyrics /
looptrace / fibers probes; cotrain-head vs per-set).

The infrastructure must be *defensible under review*: one definition of "correct"
across every table, ablation semantics that are literally "turn the component off
and re-measure," and confidence intervals computed honestly for an n=2-GT-set
regime.

**"99% accuracy" is an aspirational north star, not an empirical claim.** The
paper reports real numbers + the methodology and positions 99% (via the learned
trajectory decoder + the labeling flywheel) as the design target / future work.

### Non-goals

- No new alignment methods or probes (sensor phase is frozen, 2026-07-09). This
  lives in the sanctioned "canonical driver path" lane.
- No driver rewrites. The harness *drives existing entrypoints with config*.
- No general experiment framework (Hydra/MLflow) — YAGNI for n=2 sets, solo repo.
- Not the documentation overhaul (separate handoff:
  `docs/agent_handoff_docs_overhaul_20260711.md`). But see §7 — this harness
  becomes the number-truth that `docs/alignment_status.md` regenerates from.

---

## 2. Why this shape (the review-killer it avoids)

Today the numbers come from three different scorers: `path_decode`'s
`trajectory_acc`, the scorecard's failure attribution, and
`score_timeline_vs_gt`'s fiber-aware metric. They do **not** define "correct"
identically. A paper whose Table 2 and Table 4 use different accuracy definitions
gets caught in review. The harness enforces a single scorer as the only headline
metric path; the others are demoted to debug.

---

## 3. Architecture

New package: **`workspaces/alignment_prototype/experiments/`**.

Data flow:

```
matrix.py  →  run.py        →  score.py         →  results.py     →  report.py
(cells)       (cell→Timeline,   (Timeline→per-span   (long-format      (tables +
              cached)           rows, ONE scorer)    results store)    bootstrap CIs)
```

### 3.1 `matrix.py` — the experiment matrix

A `Cell` is a frozen dataclass:

```
Cell(
    driver: "classical" | "agentic" | "ml",
    toggles: FrozenSet[str],      # subset of {fp, hubert, lyrics, looptrace,
                                  #   fiber_gate, cotrain_head}
    set_id: str,                  # "2nvzlh2k" (BB11 = Episode 11) | "1fsnxchk" (BB12 = Volume 12)
    condition: "real" | "oracle", # oracle = GT set_start (isolates decode)
)
```

- Absence of a toggle = that component OFF (the ablated baseline).
- `looptrace` present = looptrace decoder; absent = legacy path_decode.
- `cotrain_head` present = multi-set co-trained MERT head; absent = per-set head.
- A named matrix (e.g. `PAPER`) is just a curated `list[Cell]`. Ablations are
  expressed as pairs of cells differing by exactly one toggle.

`cell_hash(cell)` gives a stable content hash for caching.

### 3.2 `run.py` — cell → Timeline

For each cell: construct the driver through the existing `harness/`
`AlignmentResult` contract with `toggles` applied, run on `set_id` under
`condition`, return a `Timeline`. Results cached at
`experiments/cache/<cell_hash>.json`; a cache hit skips inference.

**Reuse boundary:** `run.py` calls existing driver entrypoints
(`drivers/{classical,agentic,ml}.py`) with configuration. It does not
reimplement decode, placement, or identity. The only code that may change *in the
drivers* is exposing a toggle as config where it is currently hardcoded (§5).

### 3.3 `score.py` — THE single scorer

Wraps the existing `score_timeline_vs_gt` logic (already computes strict +
fiber-aware + oracle-decompose) as the **only** metric path. For a Timeline it
emits one row per GT span:

```
SpanScore(
    span_id, set_id, axis,           # axis ∈ {acappella, regular, instrumental}
    cell_hash, condition,
    strict_correct: bool,            # span-exact, ±2s, sec-weighted convention
    fiber_correct: bool,             # within-repeat-class credit
    ref_offset_err_s: float,
    placement_err_s: float,          # |pred set_start − GT set_start|
    loss_bucket: str,                # placement | decode-residual | identity | ...
)
```

`path_decode.trajectory_acc` and the scorecard's attribution are **debug-only** —
importing them for a headline number is a test failure (§6).

### 3.4 `results.py` — long-format store

One row per (span × cell × metric). Persisted to
`experiments/results/scores.parquet` (or sqlite if parquet deps are unavailable
in `venvs/audio`). Long/tidy format: every paper table is a groupby. Appends are
idempotent on `(cell_hash, span_id)`.

### 3.5 `report.py` — tables + CIs

Consumes the long store, emits markdown + LaTeX:

- **Headline table** — per set × axis: identity %, placement (median / <15s /
  p90), trajectory **strict and fiber-aware**, multiseg+loop.
- **Ablation tables** — each row = a toggle's on/off delta with a **paired
  span-level bootstrap CI** (resample shared spans with replacement, fixed seed,
  ~1000 draws).
- **LOSO row** — sourced from the existing cotrain `--loso`; surfaced first-class
  (identity transfers ~100% / placement does not). Reported as LOSO, **not** a CI.
- **External row** — the UnmixDB André-baseline comparison (warp/tempo error +
  abstention), carried in as a fixed external result.

### 3.6 The three baked-in constraints

1. **CIs are span-level bootstrap only.** `report.py` has no code path that
   computes a set-level CI — n=2 makes it meaningless. Per-set values are point
   estimates; the CI comes from the pooled/paired span bootstrap.
2. **Cross-set generalization = the LOSO row**, never a CI.
3. **fiber-aware − strict is a standing column** in every trajectory table (the
   "which-instance" residual, a named contribution).

---

## 4. Ablation semantics

An ablation is a pair of cells `(with_toggle, without_toggle)` identical in all
other fields. The result is the metric delta plus its paired bootstrap CI over the
spans both cells scored. Reported as: "removing `<component>` moves `<metric>` by
`Δ` (95% CI [lo, hi])." This makes every claim of the form "component X
contributes Y" mechanically reproducible.

---

## 5. Scope of real work vs reuse

Reused unchanged: the three drivers, the scorer core, cotrain/LOSO, the UnmixDB
result, the fiber-aware + oracle-decompose logic.

New (small, pure): `matrix.py`, `run.py` (orchestration + cache), `score.py`
(thin wrapper enforcing the single metric path), `results.py`, `report.py`.

**The one place scope can creep — capped here:** exposing probe toggles as config
on drivers that currently hardcode them. The implementation plan MUST enumerate
exactly which of `{fp, hubert, lyrics, looptrace, fiber_gate, cotrain_head}` need
config plumbing and which are already configurable, and change only those. No
behavior changes to the probes themselves — only whether they can be switched off
from config.

---

## 6. Testing

- **`report.py` unit tests** on a synthetic long table with known groupby answers
  and a fixed-seed bootstrap whose CI is asserted exactly.
- **Guard test:** assert that headline metrics come only from `score.py` — no
  `trajectory_acc` / scorecard-attribution import in the report path.
- **Single end-to-end smoke cell** (one driver, one set, `real` condition) that
  runs `matrix → run → score → results → report` and asserts a non-empty table.
- **Cache test:** second `run.py` on the same cell hits cache (no re-inference).

---

## 7. Convergence with the docs overhaul

The harness's single scorer + long-format store IS the durable source of truth for
alignment numbers. Once built, `docs/alignment_status.md` (from the docs-overhaul
handoff) regenerates its headline block from `report.py`, and the approved
`make status` target wraps that. Expect the harness's numbers to diverge slightly
from legacy prose figures — that is correct (legacy used different scorers); the
overhaul treats the harness as authoritative and corrects the prose.

---

## 8. Definition of done

- [ ] `make ablate` (or `python -m workspaces.alignment_prototype.experiments.run
      --matrix paper`) produces the full paper table set from cached cells.
- [ ] Every headline number traces to `score.py`; guard test enforces it.
- [ ] Ablation rows carry paired span-level bootstrap CIs; no set-level CI exists
      anywhere in the report path.
- [ ] LOSO row and UnmixDB external row present.
- [ ] fiber-aware − strict column present in every trajectory table.
- [ ] Toggle-plumbing limited to the enumerated set; no driver/probe rewrites.
- [ ] Tests (report unit, guard, smoke, cache) green.
- [ ] Committed in reviewable units (package skeleton; scorer wrapper + store;
      toggle plumbing; report + CIs; tests).

---

## 9. Risks & gotchas

- **Numbers will shift vs legacy prose** when everything routes through one
  scorer. Expected and correct; reconcile via the docs overhaul, not by patching
  the scorer to match old figures.
- **Compute.** The full matrix (drivers × sets × conditions × toggle combos) can
  be heavy; per-cell caching is mandatory, and the `PAPER` matrix should be a
  curated subset (ablate one toggle at a time from a strong baseline), not the
  full cartesian product.
- **Data availability.** Some drivers need audio/features/stems that live on
  pi-storage, not the stale dev DB. `run.py` must fail loudly on missing inputs
  (not silently cache an empty Timeline).
- **Parquet deps.** If `venvs/audio` lacks pyarrow, fall back to sqlite for the
  store — do not add heavy deps for this.
- **n=2 honesty is load-bearing.** Any reviewer hammers the sample size; the
  design's answer is span-level bootstrap for within-corpus CIs + LOSO for
  cross-set generalization + the flywheel as the data-scaling path. `report.py`
  must never overstate this.
