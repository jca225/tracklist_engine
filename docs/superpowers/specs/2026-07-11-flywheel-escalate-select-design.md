# Flywheel orchestration — agentic escalate → labeling queue (gears 2-3)

**Date:** 2026-07-11
**Status:** approved design, pre-implementation
**Owner:** John
**Home:** `alignment/` (`agentic/`, `review/`)

## North star

SOTA GT-closeness on all 1001tracklists data. Critical path = GT scale. Gear 1
(multi-set co-train + LOSO) is built: labeling more sets now improves the model.
This spec builds the piece that makes each hour of your Ableton labeling maximally
useful — **batch selection: label the spans the aligner most needs, not random
ones.**

## The insight (why the agentic harness is the batch selector)

The agentic loop already sorts every span into a ladder rung — `committed` /
`review` / `suggested` / `escalated`. The **`escalated`** spans are precisely
"the aligner cannot resolve this confidently" — i.e. the highest-value spans to
hand a human. So batch selection is not new machinery; it is **the agentic
harness's `escalate` rung, wired to the labeling seeder.** This also gives the
agentic harness (parked at `validated=False`) a real, n-robust job today:
*choosing what to label*, which does not require validated auto-commit precision.

## The flywheel loop this closes

```
infer (predict timeline)
   └─> agentic loop (resolve) ──> res.escalated  = the spans to label  [NEW WIRING]
          └─> seed labeling .als of exactly those spans                [NEW SEEDER]
                 └─> [you correct in Ableton]
                        └─> export_als_to_gt  ──> new GT               [EXISTS]
                               └─> cotrain --loso retrain              [EXISTS, gear 1]
```

## Scope — this cycle builds only the NEW pieces

1. **Escalated-span extraction** — a function that runs the agentic loop on a set
   and returns the escalated slot labels (+ their uncertainty, for ordering).
2. **Explicit-slot labeling seeder** — generalize the worst-spans seeder to accept
   an explicit slot list (the escalated ones) instead of ranking by GT-seconds-lost.
   Critically: escalated spans have **no GT** (that's the point), so the seeder
   must NOT depend on GT/scorecard loss — it seeds from the predicted timeline +
   the given slot list.
3. **Orchestration entrypoint** `flywheel_seed(set_id)` + a make target — chains
   1→2 to produce a ready-to-label Ableton session.

Out of scope (already exist / later): the retrain half (`export_als_to_gt` +
`cotrain`), probe-precision validation (n=2-limited, separate), multi-set state
tracking.

## Architecture

### 1. `agentic` escalated-span extraction
Reuse `agentic.loop.resolve` (already produces `Resolution.escalated`). Add a thin
`agentic/select.py`:
`escalated_slots(set_id, timeline_path, *, live=False) -> list[EscalatedSlot]`
where `EscalatedSlot = (slot_label: str, reason: str, uncertainty: float)`. It
builds the runners + spans exactly as `agentic/__main__.py` does, calls `resolve`,
and returns the `res.escalated` keys with their belief margin as `uncertainty`.
`live=False` uses cached/offline probes (fast); `live=True` runs real probes.

### 2. Explicit-slot seeder (`review/seed_slots_als.py`, new)
Factor the rendering core out of `seed_worst_spans_als` (which currently couples
"rank by GT loss" + "render A/B .als"). New:
`seed_slots_als(set_id, slots: list[str], *, out=None, template=DEFAULT_TEMPLATE)`
— renders one Live set with the predicted-timeline clips for exactly `slots`,
stamped a labeling-queue name. NO GT / scorecard dependency. `seed_worst_spans_als`
keeps working (it calls the same rendering core with its GT-ranked slot list).

### 3. Orchestration `flywheel_seed`
`flywheel_seed(set_id, *, live=False, top=None) -> Path` — calls
`escalated_slots`, orders by `uncertainty` desc, optionally caps at `top`, calls
`seed_slots_als`. Wire as `make flywheel-seed SET=<id> [LIVE=1] [TOP=N]`.

## Testing / validation

- **Unit — extraction** (monkeypatched, no probes/pi): `escalated_slots` with a
  stubbed `resolve` returning a fixed `Resolution` yields the escalated slot labels
  with their margins as `uncertainty`, ordered stably.
- **Unit — seeder core is GT-independent**: `seed_slots_als` on a fixture timeline +
  an explicit slot list renders without touching the scorecard/GT tables (patch the
  render call; assert it's invoked with exactly the given slots). Guards the "no GT"
  invariant that separates it from `seed_worst_spans_als`.
- **Refactor guard**: `seed_worst_spans_als` still produces the same slot set for a
  set (golden on its `worst_slots` output — unchanged behavior after extracting the
  render core).
- **Integration (offline, deliverable)**: `make flywheel-seed SET=1fsnxchk` produces
  a labeling `.als` of bb11's escalated spans. Needs the bb11 predicted timeline +
  agentic offline probes. Reported, not CI. If probes need `--live`/pi and it's
  heavy, run a capped `TOP` or report BLOCKED with the error — never fabricate.

## Risks & honest limitations

1. **Placement doesn't transfer (co-train finding).** On a *new* set the aligner's
   placement is weak, so many spans will escalate — good for labeling coverage, but
   the escalated set may be large. `TOP` caps it to a labeling-session-sized batch.
2. **Agentic probes are `validated=False`** — but escalation (choosing what to
   label) does NOT need validated auto-commit precision; a provisional uncertainty
   ordering is fine for *ranking what to hand a human*. This is the honest job the
   harness can do at n=2.
3. **Escalated spans have no GT** — the seeder must not assume GT (unlike
   `seed_worst_spans_als`); enforced by the GT-independence test.
4. **Human-in-the-loop** — this builds the *pre-label* half (produce the queue);
   the *absorb* half (export→retrain) already exists and is driven after you label.

## Non-goals
- The retrain half (exists: `export_als_to_gt` + `cotrain`).
- Probe-precision validation / flipping `validated=True` (n=2-limited).
- Multi-set flywheel state tracking (a later gear).
- New alignment methods.

## Open questions for the plan
- `EscalatedSlot.uncertainty` source: belief margin vs share-of-mass — use whatever
  `Resolution.escalated` already carries; decide when reading `belief.py`.
- Whether `flywheel_seed` lives in `review/` or a new `flywheel.py` — prefer
  `review/` (it's a seeding entrypoint), next to `seed_slots_als`.
