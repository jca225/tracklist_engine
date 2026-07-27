# Hand-off: acappella routing mis-route → data-engine / ingest session

**Date:** 2026-07-18 · **From:** the aligner/trajectory session (oracle-ladder work)
**To:** the acquisition-data-engine / ingest / tokenizer session
**Turf:** ingest/tokenizer routing — **not** the aligner. Filed here because the
oracle ladder quantified its e2e impact and the fix is yours.

## The finding (quantified)

The acappella **oracle ladder** (`alignment/evals/oracle_ladder.py`,
write-up `evals/ORACLE_LADDER_FINDINGS.md`) decomposes the acappella oracle→e2e
trajectory gap. **Routing is a large, free slice** — bigger than placement in
BB12, and it needs **no modeling**, only correct stem routing at inference:

| | BB12 `1fsnxchk` | BB11 `2nvzlh2k` |
|---|---|---|
| **routing slice** (R1−R0, fiber-aware pp of the R0→R3 gap) | **+12.6** | +4.6 |

Scorecard corroboration: **40% of GT-acappella spans are mis-routed** — the
timeline's `claimed_stem` says `regular`, so the span is decoded with **chroma**
instead of **HuBERT**, scoring **traj 4% vs 20%** for correctly-routed spans
(`make scorecard`, §5 routing diagnostic).

## Mechanism

The mis-route is the **w-layer inventory gap**: BB mashup w-layer slots whose
`(Acappella)` marker lives only in the visible row text (not the schema.org
`<meta name>`), so the materialized `set_track_slots.claimed_stem` is `regular`.
`infer`'s identity/route path trusts that stale value and routes the span to the
chroma/full-mix channel. Partial upstream fix already landed for the meta-name
path (888caca); the **row-text-only w-layers remain**.

## The scoped fix (already designed, UNWIRED)

`looptrace/NOTES.md` → **"w-layer axis prior (idea, 2026-07-09 — UNWIRED)"** and
`eda/alignment/failure_analysis/FINDINGS.md §C2`:

- Measured structural prior (BB11+BB12, 283 spans): **P(acappella | w-layer slot)
  = 82%**; 100% of GT acappellas are w-layers; main slots are 0% acappella.
- `set_track_slots.layer_role` exists in the DB and is consumed **nowhere** in
  the prototype; `harness/axes.py` is the natural wiring home.
- Two candidate gates: (a) flip routing on the w-layer prior alone (costs the
  ~16% regular w-layers), or (b) gate on cheap audio evidence — ref
  instrumental-stem silence or a `candidate_vocal_gate`-style HuBERT check.

## The ask

Route acappella by **audio evidence / the w-layer prior**, not the stale
`set_track_slots.claimed_stem`. This is data-engine routing (or `harness/axes.py`),
consistent with the sensor-phase freeze (it is not a new sensor). Expected e2e
payoff on the acappella axis is bounded above by the routing slice: **~+12.6 pp
fiber on BB12**, smaller on BB11.

Validate before/after on `make scorecard` (acappella routing diagnostic + the
per-axis headline), both sets.
