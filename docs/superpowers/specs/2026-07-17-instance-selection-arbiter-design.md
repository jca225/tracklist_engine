# Brain-fix (structure wall) — verified handoff

**Date:** 2026-07-17 (premise re-verified + corrected same day — see "Correction")
**Status:** HANDOFF — premise re-verified against live code; ready for a fresh agent.
**Coordinate:** another agent is actively doing **co-training** work. This lever
touches `synthetic_mix/` + `trajectory/` retrain; keep clear of `cotrain*.py` /
`cotrain_seam.py`. Pull + scan the log before starting (see
[[project_parallel_aligner_agent]]).
**Area:** `workspaces/alignment_prototype/` — the learned trajectory decoder.

> This doc began as an "instance-selection arbiter" design. Verification against
> the live code **falsified its core mechanism** (a hand-tuned decode tie-breaker).
> It was then rewritten to a "synthetic emits straight plays only" lever — which a
> **second verification (2026-07-17) also falsified** (that finding cited a dead
> v1 module). It is now redirected to the one lever that survived both passes:
> **reference-internal repeat ambiguity.** Falsified ideas are retained as
> do-not-rewalk records.

## Correction (2026-07-17) — read this first

The prior draft claimed the synthetic generator "emits straight plays only" and
that the un-run lever was "make synthetic emit multiseg/loop spans with instance
labels." **That is false.** Evidence, live code:

- JOB 1 / the trajectory pipeline run `generate_v2 --curriculum bb12-lite`. That
  curriculum ([`sections.py:8-22`](../../../workspaces/alignment_prototype/synthetic_mix/sections.py))
  sets **`n_loops: 1`** and **`instr_jump_prob: 0.85` / `instr_jump_segments: (2,3)`** —
  every window emits a loop span and ~85% of instrumental blocks are multiseg.
- The structure survives into supervision: `scenario_v2._loop_acap` emits
  `is_loop=True` + repeated `slices`; `labels_v2._acap_track`/`_instr_track`
  carry `is_loop` + `ref_segments` into the GT; `targets.raster_targets`
  rasterizes per-frame labels with `span_class ∈ {linear, multiseg, loop, oddratio}`.
- The prior draft's "gap" cited `labels.py:scenario_to_gt` — that is the **v1**
  path with **zero callers** in the tree (`generate_v2.py:30,84,124` imports
  `window_to_gt` from **`labels_v2`**, not `labels`). The agent verified against
  dead code.

So the learned decoder **is already trained on loop + multiseg structure with
per-segment instance labels.** "Emit loops" is done. What remains un-run is the
*hardness* of those loops (below).

## TL;DR for the next agent

The empirical fact still holds: the trajectory decoder **moves placement but not
structure** (multiseg+loop ~26–27% BB12). The cause is NOT "never sees repeats" —
it does. The cause is that synthetic repeats are **not hard in the way real ones
are**: `_loop_acap` points every repeat at the *same* random ref window (`ref_lo`)
and picks payloads without regard to their internal structure. The genuinely hard
decision on real sets — *"the reference track contains several near-identical
regions; which one did the DJ play?"* — appears in synthetic only **incidentally**
(when a payload stem happens to have internal repeats), never engineered,
oversampled, or labeled-as-ambiguous.

**The one un-run lever:**

**Engineer reference-internal repeat ambiguity into synthetic — select/place
loop & multiseg payloads whose reference has multiple look-alike regions, label
the true instance densely — retrain the existing `trajectory/` pipeline → eval on
BB11/BB12.**

It reuses the whole built pipeline (no new mechanism, no hand-tuned prior). Either
it moves multiseg/loop trajectory on held-out real GT, or it **definitively closes**
the "can a learned decoder learn instance-selection" question. Informative either way.

## Verified findings (with evidence)

1. **Decode-geometry tie-breaks are exhausted.** Flat `lam_back` sweep hurt
   (looptrace/NOTES.md #4); magnitude-graded `--warp-jump` A/B'd 2026-07-17 on
   BB12 (chroma, multiseg+loop) is **net neutral-to-negative** (multiseg strict
   50→49; regular +1, acappella −4, instrumental −2) — banked as
   looptrace/NOTES.md "Known-failed-approaches" #7. Phase-4 mel-residual
   discriminability also regressed. `_viterbi` graded slopes exist but stay OFF.
2. **All audio features are identical across a TRUE repeat by construction** — so
   fp-sharpness / HuBERT-margin cannot disambiguate instances. Only
   position/context can, and position is what the (dead) tie-breaks used.
3. **Learned decoder on synthetic is BUILT** (`trajectory/train.py --synthetic-root`,
   `trajectory/synthetic_adapter.py:build_synthetic_sets`, synthetic is train-only)
   and **moves placement not trajectory** (`trajectory/__init__.py:4`: multiseg+loop
   ~26–27% BB12).
4. **CORRECTED (2026-07-17):** synthetic **already emits loop + multiseg with
   `ref_segments` and `is_loop`** via the live `generate_v2 → labels_v2.window_to_gt`
   path (`sections.py` bb12-lite `n_loops:1`, `instr_jump_prob:0.85`;
   `scenario_v2._loop_acap`/`_instr_slices`; `targets.raster_targets` `span_class`).
   The prior "no `ref_segments`, straight plays only" cited `labels.py:scenario_to_gt`,
   a **dead v1 function with no callers**. The real gap is repeat *hardness*, not
   repeat *presence* (see TL;DR).

## The lever — staged

### Stage 1 — reference-internal repeat ambiguity (the actual change)
In `scenario_v2` loop/multiseg construction, bias payload/ref-window selection
toward references that contain **multiple near-identical regions** (candidate
signal: self-similarity peaks in the payload stem — reuse `ref_fibers` /
Foote-novelty machinery already in the tree, read-only, no new sensor), and place
the labeled instance among competing look-alikes. Keep the existing
`ref_segments`/`is_loop` emission — it already round-trips through
`synthetic_adapter` and `targets.raster_targets`. The change is *which* ref
window a repeat points at and *how ambiguous* its neighborhood is, not adding
segment structure (that exists).

### Stage 2 — retrain + eval (measure)
Retrain the existing pipeline with ambiguity-enriched synthetic folded in:
`train.py --split set --train-set 2nvzlh2k --eval-set 1fsnxchk --synthetic-root <dir>`
(mirror the JOB-1 volume-curve harness) and `make race --sets 1fsnxchk,2nvzlh2k
--drivers classical,ml`. Score `make scorecard`. **Decision gate:** if
multiseg/loop strict rises on the held-out real set → the decoder *can* learn
instance-selection when the synthetic decision is genuinely hard; iterate on
ambiguity realism. If flat → the learned-decoder route is closed too; record and stop.

### Stage 3 — conditional
Only on Stage-2 signal: richer cross-span context in the decoder / harder
curricula. Do NOT pre-build.

## Interaction with JOB 1 (the synthetic volume curve)
JOB 1 scales `bb12-lite` volume (N=100→1000+), and that data **already contains
loops+multiseg**. So JOB 1's structure axis is a clean control for *this* lever:
if multiseg/loop strict stays flat as N grows (more of the same structure), that
is positive evidence the bottleneck is repeat **hardness**, not quantity —
exactly what Stage 1 attacks. Read JOB 1's structure curve before starting Stage 1.

## Success metric & guards
- **Metric:** strict `trajectory_acc` on multiseg+loop, `make scorecard` (BB11
  `2nvzlh2k` + BB12 `1fsnxchk`).
- **Guard:** no regression on `linear` spans, either set.
- **Ceiling (honest):** ~half of repeats are indistinguishable takes
  (`looptrace/AUDIT.md`) — bounded upside even on success.

## Caveats / dependencies
- **n=2 real GT is the deep constraint.** A **3rd hand-labeled GT set** (John, in
  parallel) is the only thing that (a) enables LOSO n≥3 and (b) captures the real
  **placement tail** (mid-song entries over medley beds) that synthetic *cannot*
  fake (per `alignment_recharacterization.md`). Synthetic enrichment attacks the
  **structure** half only; placement generalization waits on real GT.
- Sensor phase stays **frozen** (re-validated 2026-07-17, see
  [[project_sensor_freeze_validated]]); this is pure actor/data work.

## Do-not-rewalk (falsified ideas)
1. A hand-calibrated tie-breaker over {fiber μ, fp sharpness, HuBERT margin,
   position}. **Falsified in principle** (finding #2) and **in practice**
   (finding #1). Do not build it.
2. "Synthetic emits straight plays only → add loop/multiseg segments."
   **Falsified 2026-07-17** (finding #4 correction): loop/multiseg segments +
   instance labels already emit through the live `generate_v2 → labels_v2` path.
   Adding segment *structure* is a no-op; the lever is segment *ambiguity*.
