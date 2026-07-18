# Design: Oracle instance-selection separability + cross-set transfer (Task A)

**Date:** 2026-07-18 · **Branch:** `instance-separability` (off PR #16 `worktree-acap-oracle-ladder`)
**Origin:** `docs/agent_handoff_north_star_n2_20260718.md` → Task A (flagship)
**Status:** design, pending implementation plan

## One-line

A **parameter-free** measurement gate: within the recoverable acappella population
(spans the decoder placed in the **right fiber but wrong instance**), can
`{HuBERT diagonal evidence, fiber μ/ambiguity, fp sharpness}` rank the **GT-correct**
fiber member above its same-fiber rivals — and does that ranking **transfer BB11↔BB12**?
The answer decides whether labeling BB10 → *fitting* a learned instance selector pays off.
This builds no model and moves no scorecard; it measures separability and, as a free
by-product, emits the selector's future training dataset.

## Why this task, why now (context)

The oracle ladder (`evals/oracle_ladder.py`, PR #16) established that **instance
selection is the binding constraint at the oracle ceiling**: with placement + identity +
routing all oracle (R3), acappella *strict* caps at 0.373 (BB12) / 0.300 (BB11) while
*fiber* reaches 0.591 / 0.536. The ~+22–24 pp strict→fiber headroom survives placement
being fixed and is positive on both sets → a selector that picks the correct repeat
instance is *necessary* to break ~35%. See `evals/ORACLE_LADDER_FINDINGS.md`.

What the ladder did **not** test: whether that headroom is *realizable* — i.e. whether the
correct instance is **distinguishable** by available features from the wrong-but-same-fiber
instances. That distinguishability is exactly what this task measures.

**The n=2 boundary (honored):** *fitting* a selector and claiming a LOSO win over the
classical decoder needs a 3rd GT set (BB10, unlabeled) — six decode-layer instance threads
already died to n=2 overfitting (attic). This task stays on the *doable-at-n=2* side: it
*measures whether the selector can work* and *builds its input pipeline*. No fitted model is
claimed as a win.

## Ground truth is already sufficient (key enabler)

GT acappella rows in `labeling/fixtures/bb1{1,2}_ground_truth.yaml` carry the true ref-time
offset the DJ played: `ref_start_s` / `ref_end_s` (plus `tempo_ratio`, `pitch_shift_semi`).
The **correct instance** is therefore unambiguous and needs no new labeling: it is the fiber
member interval containing GT `ref_start_s`. BB12 has 97 acappella GT rows; the recoverable
subpopulation (fiber-correct ∧ strict-wrong) is ~20 spans/set — small n, which is precisely
why this is a *parameter-free separability gate*, not a fit.

## Scope

**In:** Task A separability + transfer measurement, acappella axis, BB11 + BB12; plus the
"bonus" persisted per-candidate feature dataset (the selector's input pipeline, ~free).

**Out (separate efforts, not this pass):** Task B (learned trajectory decoder), Task C
(regular/instrumental ladder + legacy-oracle cross-check), any *fitted* selector claimed as
a scorecard win, any new probe/channel/prior (sensor phase is frozen — this task only
*reads* existing HuBERT/fiber/fp outputs).

## Architecture

New module `workspaces/alignment_prototype/evals/instance_separability.py`, reusing
`oracle_ladder.py`'s GT-row/fiber/score plumbing. Four units, each independently testable:

### Unit 1 — recoverable population (`build_population`)
For a set: load GT acap rows (`oracle_ladder._load_gt_acap`), and the R0/looptrace decode
timeline. Using the existing `real_score_fn` machinery (`trajectory_acc` → strict, fiber),
select GT rows where **fiber == correct ∧ strict == wrong** (the +22 pp headroom). Output:
list of population rows, each `{gt_row, ref_audio}`.
- *Interface:* `build_population(set_id, gt_path, r0_timeline, by_tid) -> list[PopRow]`
- *Depends on:* `oracle_ladder`, `score_timeline_vs_gt`, `path_decode.trajectory_acc`.

### Unit 2 — candidate enumeration (`enumerate_candidates`)
For a population row: compute the ref track's HuBERT-L9 fibers
(`fibers.detect.compute_fibers` — reuse `oracle_ladder`'s cache), find the **GT fiber**
(`fiber_at` on GT `ref_start_s`), enumerate that fiber's member intervals
(`fiber_intervals`) as candidate instances. Label the interval containing GT `ref_start_s`
as `is_gt_instance=True`. Skip rows whose GT fiber has <2 members (no instance choice → not
part of the selection problem; log the count).
- *Interface:* `enumerate_candidates(pop_row, fibers, label_hz) -> list[Candidate]`
- *Candidate:* `{ref_start_s, ref_end_s, is_gt_instance}`.

### Unit 3 — feature extraction (`candidate_features`)
Per candidate, compute the three named features (all pre-existing outputs; no new sensor):
1. **HuBERT diagonal evidence** — `path_decode._scores_at_stretch` (L9) evaluated at the
   candidate's ref offset (the windowed matched-filter / path score at that instance).
2. **fiber μ / ambiguity** — `fibers.detect.compute_fibers_soft` membership μ at the
   candidate + `fiber_ambiguity` of that placement.
3. **fp sharpness** — `landmark_fp.vote_sharpness` on `landmark_fp.fp_offset` votes for that
   instance's offset window.
- *Interface:* `candidate_features(candidate, ctx) -> {hubert, mu, ambiguity, fp_sharpness}`
- *ctx* bundles the cached ref features / fiber soft-membership / fp votes so extraction is
  pure given the cache. Exact per-feature offset windowing is fixed during TDD against the
  real signatures; the design commits to *which symbol* each feature comes from.

### Unit 4 — separability + transfer metrics (`score_separability`, `transfer`)
Pure ranking/metric logic (unit-tested with synthetic candidates, no audio):
- **Oracle ceiling:** for each feature alone + one **fixed** linear combo (equal-weight
  z-scored; not fitted), rank each row's candidates; is top-1 the GT instance? Report
  **fraction of the strict→fiber gap recovered** = (top-1-correct rows / **selectable rows**),
  per set, per feature. *Selectable rows* = recoverable rows whose GT fiber has ≥2 members
  (a genuine instance choice); single-member recoverable rows (recoverable by tolerance, not
  by selection) are reported as a separate bucket and excluded from the ranking denominator,
  so the fraction is not inflated by rows with nothing to choose.
- **Transfer:** fit the simplest ranker (3-param logistic on the 3 z-scored features) on one
  set, score the other; both directions. Report **per set, both directions** — never a
  cross-set CI (n=2).

### Bonus — dataset persistence
Persist per-candidate `{set_id, gt_row_key, features, is_gt_instance}` to
`evals/out/instance_separability/<set_id>_candidates.jsonl`. This *is* the selector's
training-input pipeline: when BB10 is labeled, it drops in as a third file and fitting +
LOSO is turnkey. No fitting happens here.

## Data flow

```
GT yaml + R0 timeline ──> build_population ──> [PopRow]
                                                  │  (per row: ref fibers, cached)
                                                  ▼
                                          enumerate_candidates ──> [Candidate w/ is_gt_instance]
                                                  │
                                                  ▼
                                          candidate_features ──> feature dicts ──┬─> jsonl dataset (bonus)
                                                                                  │
                                                                                  ▼
                                          score_separability / transfer ──> per-set tables
```

## Error handling

- Rows whose GT fiber has <2 members: excluded from the ranking population (no instance
  choice), counted and logged — not an error.
- Missing ref audio / feature cache for a row: skip with a logged warning, count skips;
  never silently drop (mirrors the ladder's fixed-denominator honesty). A high skip count
  is itself a reportable caveat.
- Feature extraction that cannot evaluate at an offset (out-of-range): that candidate gets a
  `None` feature and is ranked last for that feature; logged.

## Testing (TDD)

- **Unit (no audio):** `score_separability` / `transfer` with hand-built synthetic candidate
  sets (known GT instance, known feature orderings) — verify top-1 accounting, gap-recovered
  fraction, and the LOSO both-directions bookkeeping. `enumerate_candidates` with synthetic
  fibers — verify GT-instance labelling and the <2-member exclusion.
- **Integration (local audio, both sets present):** run the full pipeline on BB11 + BB12;
  assert it produces per-set tables + the jsonl dataset, and that the recoverable population
  size matches the ladder's strict→fiber headroom (sanity cross-check against
  `ORACLE_LADDER_FINDINGS.md`).

## Decision rule (arbiter)

The arbiter for this task is the **per-set separability + transfer tables** (measurement),
not `make scorecard`. From the handoff:
- **GO** (build the selector once BB10 lands): oracle top-1 recovers **≥ ~½** the strict→fiber
  gap **AND** transfer is **non-negative in both directions**.
- **STOP / real finding**: features don't separate, or the correct pick is genuinely
  instance-ambiguous (both channels agree on the wrong instance) → the selector is bounded and
  BB10 won't rescue it. Reported as a finding, written to a new
  `evals/INSTANCE_SEPARABILITY_FINDINGS.md`.

## Guardrails honored

- Worktree off the PR #16 tip (oracle_ladder + `fibers/` plumbing not yet on `main`).
- **Axis rule:** `claimed_stem` from the matched GT row, never the timeline span.
- **n=2 → LOSO both directions, per-set, no cross-set CI.**
- **Sensor phase frozen:** no new probes/channels/priors; only reads existing outputs.
- **Off-limits:** `pws_aligner/**`, `acquisition-data-engine` / `cotrain-*` branches.

## Non-goals / YAGNI

No fitted selector as a win, no scorecard move, no new feature sensors, no regular/instrumental
axis, no hyperparameter search on the linear combo (equal-weight z-score is deliberate — a
*tuned* combo at n=2 would be the overfit trap this task exists to avoid).
