# Baseline-vs-Agent Placement Comparison on Real BB Mixes

**Date:** 2026-07-17
**Status:** design (approved) — implementation plan to follow

> **Amendment 2026-07-17 (post-feasibility check).** NMF (André) and DTW build
> dense matrices sized to the mix length; they were built for UnmixDB *excerpts*
> (short, ~3 candidate tracks) and **do not scale to full-length BB mixes**
> (~3,581 s, ~150 spans) without windowing adaptations that change the method
> (DTW ≈ 12 GB/span cost matrix; NMF ≈ 2.2 GB/track and loses its joint-
> superposition purpose per-track). Decision: **do not run NMF/DTW on BB.** The
> writeup instead states that André's line optimizes a different regime (short
> synthetic excerpts) and does not mesh with full-length real mixes. The BB table
> compares the **tractable full-mix matched-filter baselines** (`no_warp`,
> `grid_mf`, `fused` — `fftconvolve`, O(N log N), run unchanged) against
> `classical` + `agentic`. The claim narrows from "beats André on real mixes" to
> "naive full-mix placement → the agent on real mixes" (a legitimate ablation).
> §3/§4/§6 below read through this amendment.
**Author:** alignment session
**Topic owner note:** distinct from `2026-07-17-synthetic-transfer-spike-design.md` (parallel agent); no file overlap.

---

## 0. One-sentence goal

Produce the head-to-head placement table that does not currently exist:
**NMF (André, reproduced) and DTW vs classical vs agentic**, on the two real
ground-truth mixes (BB11 `2nvzlh2k`, BB12 `1fsnxchk`), **placement axis only,
identity given** — the missing evidence for "the agentic aligner beats prior-art
baselines on *real* mixes."

## 1. Why this experiment (and not "agent on UnmixDB")

The intuitive move — run the agent on UnmixDB — is the wrong lever. UnmixDB has no
stems/vocals, no repeated-instance structure, and single linear warp per span, so
the agent's differentiating probes (HuBERT-vocals, lyrics, surprise, structure)
**abstain or go inert**, and it collapses to fingerprint+chroma placement — i.e.
it would merely tie the existing `fused_resample` method. See
[alignment_recharacterization.md](../../alignment_recharacterization.md) §4b.

The comparison that *is* missing and *is* informative is the inverse: the
published baselines currently exist **only on UnmixDB**, and the agent exists
**only on BB**. There is no single table where they are compared. This experiment
builds that table on the data (real mixes) and axis (placement heavy-tail) where
the agent's advantage is real.

## 2. Core principle — one scorer, one span set, all methods

Every method — `nmf`, `dtw`, `no_warp`, `classical`, `agentic` — is scored through
the **same** placement scorer over the **same** GT spans:

    set_start_err = |pred_set_start_s − GT_set_start_s|   (per span, seconds)

reported as **MAE / median / <15 s / p90** — the exact placement metric already
used for the agent in [alignment_status.md](../../alignment_status.md) §1.

- Baselines (`nmf`, `dtw`, `no_warp`) produce `set_start_s` natively.
- `classical` / `agentic` get their per-span `set_start_s` **extracted from their
  predicted timeline JSON** and scored through the identical `|pred − gt|`.

This eliminates the "different scorer" objection: the agent is not cross-referenced
from a separate scorecard; it is re-scored on the same spans as the baselines.

## 3. Architecture (smallest-code path)

New module: `workspaces/alignment_prototype/experiments/bb_baselines.py`
(final name TBD in the plan). It:

1. **Builds `eval_bench.Sample` objects from BB data** and reuses
   `method_nmf` / `method_dtw` / `method_no_warp` from
   `workspaces/alignment_prototype/external/eval_bench.py` **unchanged**. Reusing
   the identical baseline code that ran on UnmixDB is an honesty win — same
   baseline, two datasets.
2. Per GT span constructs one `Sample` with:
   - `mix_path` = `~/aligning/{set}/mix.*` (the mix audio)
   - `track_paths` = `{idx: <that span's reference recording audio>}` — a **single
     true reference** (identity given; no distractor pool)
   - `gt` = `[GTSpan(idx, GT_set_start_s, GT_tempo_ratio)]` from the GT fixture
3. Runs each baseline → `Pred(set_start_s, tempo_ratio, score)` per span.
4. Extracts `classical` / `agentic` per-span `set_start_s` from
   `out/{set}_predicted_timeline.json` and `out/{set}_agentic_timeline.json`.
5. Scores all five methods with the shared `|pred − gt|` metric and emits one
   comparison table.

### Inputs / sources (same the agent already uses)
- **Audio:** `~/aligning/{set}/` — `mix.*` + `tracks/` + `manifest.json`.
- **GT set_start per span:** `labeling/fixtures/{bb11,bb12}_ground_truth.yaml`.
- **Method code:** `external/eval_bench.py` (`method_nmf`, `method_dtw`,
  `method_no_warp`, the `Sample`/`Pred`/`GTSpan` dataclasses).
- **Ours timelines:** `out/{set}_predicted_timeline.json` (classical),
  `out/{set}_agentic_timeline.json` (agentic).

### Identity given (approved decision)
Each span's `track_paths` holds exactly one reference — the correct recording.
Baselines measure **placement in isolation**; identity is not re-litigated (it is
already ~84% and called solved). This matches how the agent's 1.9 s / 3.3 s is
measured (it refines known-identity spans) → apples-to-apples, and is the
**most favorable-to-baseline** framing.

## 4. Settled decisions

| Fork | Decision | Rationale |
|---|---|---|
| Baseline identity | **Given (placement-only)** | apples-to-apples with agent; isolates placement axis |
| Metric | **set_start MAE / median / <15 s / p90** | matches status-doc placement metric; baselines can't produce trajectory |
| Methods | **nmf, dtw, no_warp, classical, agentic** | prior-art + floor + ours; keep table readable |
| Span scope | **Every GT span** (incl. loops/multiseg, using span-start) | excluding hard spans flatters baselines and hides the real-mix tail — the whole point |
| Table shape | **Flat 5×2** (methods × {BB11, BB12}) × {MAE, median, <15 s, p90} | readable; per-stem breakdown deferred (YAGNI) |
| Compute | **Local Mac `venvs/audio`** (CPU) | NMF/DTW are librosa/mel; no MERT (identity given), no pi, no GPU |

## 5. Explicit non-goals (YAGNI)

- **NOT** running the agent on UnmixDB (wrong lever — §1).
- **NOT** reproducing André's exact published protocol (ours is a reproduction;
  claim is "beats reproduced baseline," not SOTA).
- **NOT** adding new baselines or a per-stem breakdown in v1.
- **NOT** touching the DB, pi-storage, or any parallel agent's files.
- **NOT** a fiber-aware / trajectory column — baselines can't produce trajectory;
  that stays a BB-only, agent-only strength reported elsewhere.

## 6. Caveats baked into the output (surfaced, not buried)

The emitted table/report must state, inline:

1. **NMF is our reproduction**, not André's published protocol → "beats reproduced
   baseline," never a SOTA banner.
2. **n = 2 real sets** → no cross-set confidence interval; directional only.
3. Baselines assume `ref_start = 0`, single-instance spans. BB mid-song entries
   and loops will inflate their error — **that heavy tail is the finding, not a
   bug** (real mixes stress the axis synthetic benchmarks do not; recharacterization §4b).

## 7. Preflight (first implementation step, not a fork)

Confirm `~/aligning/2nvzlh2k*` and `~/aligning/1fsnxchk*` exist locally with
`mix.*` + `tracks/` + `manifest.json`. If a set's aligning dir is missing, pull it
via the `alignment-pull` skill before running. Also confirm the `out/*_timeline.json`
files exist for classical + agentic (produced by `make race`); if stale, note it —
a fresh `make race` re-runs inference and mutates timelines (out of scope here,
flag if needed).

## 8. Validation / testing

- **Adapter sanity:** every constructed `Sample` has a readable `mix_path`, a
  readable single `track_path`, and a finite `GT_set_start_s`. Fail loud on any
  missing.
- **Metric sanity:** re-scoring the agent timeline through the shared scorer
  reproduces the status-doc agent placement (median ≈ 1.9 s BB11 / 3.3 s BB12,
  base-classical timeline) within tolerance — proves the shared scorer is
  consistent with the canonical one.
- **Baseline sanity:** on a hand-checked easy straight span, NMF/DTW `set_start`
  is within a few seconds of GT (guards against a broken adapter).

## 9. Output

- A comparison table (5 methods × 2 sets × 4 stats) written to
  `experiments/results/` (or printed + persisted alongside existing eval outputs;
  final location in the plan).
- A short findings note interpreting the table against the three-axis frame,
  with the §6 caveats attached. Numbers do **not** get hand-copied into other
  docs — [alignment_status.md](../../alignment_status.md) remains the SSOT; this
  experiment's headline is cited from wherever the harness persists it.

## 10. How this feeds the paper

This table is the **Paper 2 (system-forward)** placement head-to-head. It does not
by itself make Paper 2 writable — that still needs (a) a third GT set for a
generalization CI and (b) ideally André's published-protocol comparison. It does
close the "there is no table comparing the agent to prior art" gap, on the axis
and data where the agent's win is real.
