# Phase 1B — identity candidate-pool + audio-perception override (spec)

> **DRAFT 2026-07-25.** The capability half of Phase 1: close Law 5 (aligner-side)
> and Law 4 by replacing the shipped **size-1 candidate pool** with a real
> multi-candidate pool + a **gated** audio-perception override. Built + measured
> against the RT1 honest baseline (Phase 1A), on branch
> `provenance-engine-phase1`. Grounded in
> [../../workspaces/alignment_prototype/open_set_acappella_identity_findings.md](../../workspaces/alignment_prototype/open_set_acappella_identity_findings.md)
> and the law audit ([law_audit.md](law_audit.md) §4/§5).

## 0. The defect, precisely

`infer.py:124` `fetch_slot_rows` → `COALESCE(recording_id, track_id)` and
`infer.py:204` `slot_pools_from_rows` build **one `(recording_id, stem)` per
slot_label** from `set_track_slots`. `mert_model.predict_sequence` → `_assign_slot`
then "selects" from a pool of size 1 — it can only confirm the tokenizer's claim.
The discriminative MERT head already exists (`head.identity_logits`); **the missing
piece is the pool, not the scorer.** Consequence (decision #19): the shipped
"identity" is the tracklist-claim accuracy, not an aligner capability.

## 1. Non-negotiable constraints

- **Sensor phase is closed** (module CLAUDE.md): no new probes/channels/priors.
  1B promotes an *existing* channel (MERT) at tuned layers + an aggregation
  (chamfer) — the findings doc explicitly classes this as layer/aggregation
  tuning, not a new channel. **Stay inside that line.** No pitch-estimation probe
  in the first cut (see §4).
- **Abstain, never lie** (system law): the override is *fail-closed* — it replaces
  the claim only when perception is both confident AND disagrees; otherwise the
  claim stands, or the slot abstains (NULL recording_id, persisted — Law 7).
- **Do no harm to identity.** The tokenizer claim is right most of the time, and
  blind MERT is only **68% on BB12 acappellas** (unstable cross-set). A naive
  override *can regress* identity by overriding a correct claim. The gate exists
  to prevent that; the acceptance test is "identity ≥ RT1 baseline on both sets,"
  not "identity went up somewhere."

## 2. Three parts

### A. Candidate-pool construction (`candidate_pool.py`, new)
Per slot, build a pool = **{the tokenizer claim} ∪ {the set's stem-appropriate
reference pool}**, so the aligner *can* pick a different recording than claimed.
- Source of the set pool: the references already loaded for the set
  (`load_bb12_mert` `refs`, keyed by recording_id) + the aligning-dir stem files.
  The open-set eval used the set's ~89 acappella refs / ~55 instrumental refs.
- Route by stem (decision #22, final): acappella slots → acappella-ref pool;
  regular/instrumental slots → instrumental-ref pool. (Full vocal songs identify
  by their instrumental backbone — there is no third regime.)
- Always include the claim in the pool (so "no override" is representable) and
  record pool provenance (size, members, source) for the AxisBelief.

### B. Blind open-set identity LF (`open_set_identity.py`, new)
Stem-routed MERT chamfer, **blind** (no tracklist prior, no oracle pitch):
- **acappella:** query = mix **vocal** stem window; refs = acappella pool;
  **MERT L3** chamfer `mean_q max_ref cos`; **conditional top-k** (mean of top-25%
  query bars) only for spans > 60 s (long-span layered-vocal contamination).
- **instrumental / regular:** query = mix **instrumental** stem; refs =
  instrumental pool; **MERT L22** chamfer + chroma + fp (all strong on
  instrumentals; oracle ceiling 94–95% both sets).
- Fusion: **borda** over the strong LFs per stem. First cut = **MERT-chamfer
  alone** (blind, no fp/pitch) → 92% BB11 / 68% BB12 acappella, 89/89
  instrumental. This is the deployable core with zero oracle and zero new probe.
- **Data dependency:** L3 + L22 per-measure MERT for the set's mix stems + ref
  pool. Available via `export_mert_from_pi.py <set> <out> <layer>` (all-layer 330M
  blobs on pi); local cache is L6-only. Extend `mert_store` to cache a requested
  layer (`{set_id}_mert_L{n}.npz`). Runs on pi/DB — **no Mac-GPU contention**, so
  the pull can overlap RT1.

### C. Gated override seam (edit `infer.py` / `mert_model`, LAST — collides with RT1)
- Replace `slot_pools_from_rows` (size-1) with `candidate_pool.build(...)`.
- After `predict_sequence`/`_assign_slot` scores the pool, apply the **margin
  gate**: override the claim iff `top1 != claim` AND `margin(top1, claim) ≥ τ`
  AND `top1_confidence ≥ floor`. Else keep the claim. Else (claim not in pool /
  all low) abstain.
- Emit per-slot identity provenance = {pool, chosen, claim, margin, LF votes,
  decision ∈ accept-claim | override | abstain} — the `AxisBelief(IDENTITY)` seed
  (Phase 1 provenance primitive; also satisfies Law 21 explainability for identity).
- τ / floor tuned on ONE set, validated on the OTHER (LOSO), never fit on both
  (Law 19). Report per-set (Law 18).

## 3. Measurement (the acceptance gate)
1. RT1 baseline (Phase 1A): identity with the size-1 pool = the claim accuracy.
2. Re-infer with pool + gated override → identity per set.
3. **Accept iff** identity ≥ baseline on **both** BB11 and BB12, override precision
   is high (few correct-claim→wrong-override flips), and abstentions are persisted.
   A capability that helps BB11 but regresses BB12 (the MERT-68% risk) is **not**
   shipped — it waits for the learned combiner (co-train) that absorbs the
   cross-set LF flip.
4. Numbers → `alignment_status.md` (SSOT) only after this gate, hand-verified.

## 4. Deferred (explicitly out of first cut)
- **Blind pitch/key estimation** to make the pitch-fp LF deployable (the oracle in
  the 96% result). Adding it is arguably a new prior → defer past the sensor-phase
  reopening or fold into the learned combiner.
- **Learned cross-set combiner** (co-train) — the findings' stated lever past
  borda; buys robustness against the BB11↔BB12 LF-dominance flip. Phase 3 (needs
  the cotraining lineage machinery).
- **Corpus-wide pool** (beyond the set's own refs) — larger open-set; later.

## 5. Law ledger deltas when 1B lands
Flip in `tests/laws/test_system_laws.py`: **Law 5** (aligner-side no longer
path/slot identity), **Law 4** (drop the `COALESCE` source-key fallback), partial
progress on **Law 21** (identity explainability) and **Law 7** (durable aligner-side
abstention store). Write the real guard bodies as each flips.
