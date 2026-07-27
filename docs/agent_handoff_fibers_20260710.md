# Handoff — fiber program + honest bench session (2026-07-09/10)

One-session arc: "make fibers complete and battle-tested" → fibers validated,
consumed, and the alignment bench re-based on the honest ruler.

## What shipped (all on `synthetic-warp-wiring`, pushed)

- **`fibers/` package** (`alignment/fibers/`): detect /
  harmony / gt_als, shims at old paths (`ref_fibers`, `harmony_fibers`).
- **Two real detector bugs found by the first-ever unit fixtures**
  (`tests/test_fibers.py`, 12 tests): silence gate needed an absolute floor;
  the per-frame gate fragmented breathy vocals below `min_repeat_s` — THE
  acappella recall hole. v4 per-run voiced-fraction gate: vocal coverage
  0.06–0.28 → 0.33–0.73, zero-fiber refs rescued, **ear-validated by John**.
- **Ableton GT loop** (`fibers/gt_als.py`): render fiber overlay .als →
  hand-correct → import → `fiber_gt.yaml` + per-arm P/R. 9 refs determined
  (`~/aligning/_fibers/*.fiber_gt.yaml`). Protocol: REJ track = explicit
  negative; CAND leftovers = unreviewed; clone-certified CANDs auto-promote
  (overlap-aware).
- **Phase-cancel verdict** (John's idea; `fibers/NOTES.md` + 5-arm research):
  null test = deterministic **clone certificate**, not a detector. Wired as
  the clone tier (CLONE/KEYLOCK in clip names + GT yaml). Robust to
  hand-trimmed GT (5s lag search + spectral check at refined lag).
- **B3 wiring**: `joint_ref_decode` emits per-span `fiber_instances` +
  `fiber_ambiguity` (the W2 span-posterior hypothesis set). BB12: 52/121
  spans instance-ambiguous.
- **Overlay-density stratification** in `score_timeline_vs_gt`: median
  concurrency ≥4 = sustained medley pileup (partly ill-posed to recreate);
  reported separately + pileups-excluded headline.
- **Race board v2** (`drivers/race.py`, `--fibers` flag removed): strict AND
  fiber-aware always shown + noPile + abstention.
- **Bug fixes en route**: review/ out-dir drift from the taxonomy move
  (953c4b3 — found within a minute of attempting the listening workflow),
  clone-verdict lag slack, import-semantics poisoning.
- **Worst-spans PRED-vs-GT A/B seeder** (`review/seed_worst_spans_als.py`):
  `~/aligning/_review/BB{11,12} WORST SPANS SEEDED.als`.

## The bench (full race, refreshed timelines, 2026-07-10)

| set | driver | place | ref | strict% | fiber% | acapF% |
|---|---|---|---|---|---|---|
| BB12 | classical | 4.8 | 15.6 | 21 | 45 | 44 |
| BB12 | agentic | **3.3** | 15.6 | 21 | 45 | 44 |
| BB12 | ml (ungated) | 4.8 | **9.6** | 19 | 39↓ | 44 |
| BB11 | classical | 6.8 | 6.2 | 20 | 40 | 33 |
| BB11 | agentic | **1.9** | 7.8 | 20 | 40 | 33 |
| BB11 | ml | 6.8 | **2.9** | **23** | **41** | **38** |

Readings: agentic = placement champion both sets; ml = ref-offset champion
where classical is weak (BB11), regresses BB12 without `--ml-gate`; the
strict→fiber gap (+20–25pp, all drivers) = unknowable instance choice —
**~a quarter of the "error" was the ruler, not the aligner**. μ needs no
calibration yet (saturated 1.000 over 33.9k GT pairs, floor 0.5).

## John's queue (batched, his pace — do not nag)

1. Worst-spans A/B listen (`_review/*.als`): prepend BUG/HARD/GTW to red
   PRED track names. He suspects unnoticed bugs (one already confirmed).
2. Fiber listening (gated on alignment results): 3 instrumentals (= the
   harmony-promotion dataset) + Freeze Time/ITNOL/Cascada vocals.

## Next levers (in order)

1. Import John's verdicts → fix BUGs → re-bench (`make race`).
2. Third GT set (BB10 or Murph) → unlocks the learned instance selector
   (decode-residual = the largest loss share) + LOSO for fusion.
3. Compose the winner: agentic placement + gated-ml decode as the default
   `make align` composition (P3 flip criteria per drivers plan).
4. **INTERNAL paper** (John's call 2026-07-10, not external): the
   repeat-equivalence evaluation story; gaps = GT breadth, UnmixDB re-run
   under fiber metric, corpus clone stats.
