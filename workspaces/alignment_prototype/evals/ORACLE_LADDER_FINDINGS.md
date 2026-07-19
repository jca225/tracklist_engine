# Acappella oracle→e2e gap decomposition — findings

**Date:** 2026-07-18 · **Branch:** `worktree-acap-oracle-ladder` (off `origin/main` f9c9fe4)
**Harness:** `evals/oracle_ladder.py` · **Spec:** `docs/superpowers/specs/2026-07-18-acappella-oracle-ladder-design.md`

> **Verdict (one sentence):** the learned acappella instance selector is worth
> building — it is the *binding constraint at the oracle ceiling* (acappella
> strict cannot exceed ~0.30–0.37 without it, in both sets) and its headroom
> survives placement being fixed — but it is **co-equal with placement** (which
> is set-dependent), remains **gated on a 3rd GT set for LOSO**, and this ladder
> is the quantified case for labeling BB10 to unlock it. **Routing is a large,
> free hand-off to the data-engine session** (BB12 +12.6 pp).

## Method

For each set, four timelines are built by oracle-substituting one nuisance at a
time, **decoder held fixed at looptrace** (the `_lt` scorecard source of truth),
each scored by `trajectory_acc` over a **fixed GT-acappella-row denominator**
(a row with no decodable span scores 0 — never dropped). Join is by
`recording_id` + time (GT uses flat slots `002`, the timeline uses w-layers
`1w1` — a slot join returns ~empty; see the fix below).

- **R0** e2e (predicted everything) → **R1** +routing (GT stem → HuBERT) →
  **R2** +identity (GT recording; mis-identified rows recovered via a
  time-overlap coverage span) → **R3** +placement (GT set_start; all rows).

## Results

| rung | BB12 `1fsnxchk` strict → fiber | BB11 `2nvzlh2k` strict → fiber |
|---|---|---|
| R0 e2e            | 0.111 → 0.338 | 0.143 → 0.289 |
| R1 +routing       | 0.209 → 0.464 | 0.168 → 0.335 |
| R2 +identity      | 0.224 → 0.483 | 0.168 → 0.335 |
| R3 +placement     | 0.373 → 0.591 | 0.300 → 0.536 |

Denominator: BB12 n=97, BB11 n=91 GT-acappella rows (all matched/scored; BB11
has no never-covered rows, BB12 has ~9 recovered at R2/R3).

**Fiber-aware attribution of the R0→R3 gap (pp):**

| slice | BB12 | BB11 |
|---|---|---|
| routing  (R1−R0) | **+12.6** | +4.6 |
| identity (R2−R1) | +1.9 | +0.0 |
| placement (R3−R2) | +10.8 | **+20.1** |
| R0→R3 fiber gap (total) | +25.3 | +24.7 |

**Instance-selection headroom (strict→fiber gap):** BB12 **+22.7 @R0 / +21.8 @R3**;
BB11 **+14.6 @R0 / +23.5 @R3**.

**Anchors (correctness gate):** R0 reproduces the scorecard acappella e2e
(strict ~0.11–0.14 / fiber ~0.29–0.34). R3 (looptrace, full denominator) lands
strict ~0.30–0.37 / fiber ~0.54–0.59 — in the looptrace-oracle ballpark; it sits
*below* the looptrace/NOTES subset-oracle (~0.43–0.44 strict) because that
number is over the decodable straight-clip subset (n≈17–21), while R3's
denominator is *all* GT-acap rows including the hard ones. Both anchors hold.

## Reading

1. **Instance selection is the binding constraint at the ceiling.** With
   placement + identity + routing *all* oracle (R3), acappella strict still caps
   at **0.373 (BB12) / 0.300 (BB11)** while fiber is **0.591 / 0.536**. Nothing
   short of closing the strict→fiber gap moves acappella past ~35%. The headroom
   is large, positive in both sets, and **grows/holds when placement is fixed**
   (BB11 +14.6→+23.5; BB12 +22.7→+21.8) — i.e. it is *not* an artifact of bad
   placement. Unlike placement, it looks placement-independent → likelier to
   transfer cross-set. **This is the case for labeling BB10** (the 3rd GT set the
   selector needs for leave-one-set-out; see below).

2. **It is co-equal with placement, not dominant.** Instance headroom is the
   largest single slice in BB12 but **placement (+20.1) beats it in BB11**.
   Placement is **set-dependent** (BB12 +10.8 vs BB11 +20.1) — the same
   non-transfer the co-train LOSO run found (the MERT head memorizes placement
   per-set; `cotrain_loso_findings.md`). So placement is the other half of the
   prize, and its transferable lever is also the trajectory decoder / more GT,
   not a hand-tuned prior.

3. **Routing is a free hand-off.** +12.6 pp (BB12) comes from fixing the stale
   `claimed_stem` w-layer mis-route (→ HuBERT). It is an **ingest/tokenizer**
   fix (the acquisition-data-engine session's turf), not a model — quantified
   here and handed off. Minor in BB11 (+4.6).

## Decision

- **Build the learned instance selector** ({HuBERT diagonal, fiber μ/ambiguity,
  fp sharpness}) — it is the ceiling's binding constraint and the only large,
  placement-independent, transferable-looking prize. **Precondition unchanged:
  a 3rd GT set for LOSO** (still n=2; BB10 unlabeled). This ladder is the
  evidence that labeling BB10 pays for itself.
- **Hand routing to the data-engine session** (quantified: BB12 +12.6 pp).
- **Placement is the co-equal other half** and is per-set — pursue via the
  trajectory decoder + denser GT, not a hand-tuned placement prior.

## The bug that was root-caused (not patched)

First run gave R0 strict 0.002 (should be ~0.11). Cause: joining GT↔timeline by
`slot_label`, but GT uses flat numeric slots (`002`) and the timeline uses
w-layer slots (`1w1`) — a slot join matched 27/97. Fixed by joining on
`recording_id` + time and stamping a collision-free GT-row index. `build_span_table`
carries the same warning (BB12 slot join measured 0/83).

## Reproduce

```bash
venvs/audio/bin/python -m workspaces.alignment_prototype.evals.oracle_ladder \
  --set-id 1fsnxchk --gt labeling/fixtures/bb12_ground_truth.yaml \
  --r0-timeline workspaces/alignment_prototype/out/1fsnxchk_predicted_timeline_lt.json
# and 2nvzlh2k / bb11_ground_truth.yaml
```

Headline provenance is `docs/alignment_status.md`; per-rung machine output is
`out/oracle_ladder/<set_id>/ladder.json` (gitignored).
