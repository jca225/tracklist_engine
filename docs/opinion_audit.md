# Opinion audit — the pipeline's hard-coded decisions, censused

**Opened by John 2026-07-10**: "the code has fixable opinions — perhaps another
labeled DJ set is not the solution." The recurring failure signature (three
confirmed instances in one day): **a rule that silently discards strong
evidence**. This doc is the census of every such rule in the live decode path,
the audit's work-list. Fixes land ONE AT A TIME, race board after each
(`make race`); the third GT set decision re-opens only when this audit runs dry.

Census method: full sweep of infer / joint_ref_decode / path_decode / fp stack /
stem placement / harness / agentic / fibers / looptrace / drivers, classifying
every gate as (A) guillotine — binary discard of potentially-strong evidence,
(B) soft/graded, (C) benign bound. **Totals: 11 A / 8 B / 14 C; 7 of the 11
guillotines are SILENT (no log when they fire).**

## Class A — guillotines (the audit targets)

| # | gate | where | discards |
|---|---|---|---|
| 1 | `fp_placement_gate_s=90` | infer.py:459 | fp diagonal regardless of votes/sharpness when \|fp−MERT\|>90s — **confirmed victim: 499-vote/33:1 diagonal, Honest Virtu, 123s error**. SILENT per-span. |
| 2 | `instr_stem_gate_s=90` | infer.py:689 | same guillotine, instrumental lane |
| 3 | `HIT_MIN_VOTES=25` | mix_fp_hits.py | fp hits with ≤24 votes, no gradient. SILENT |
| 4 | `HIT_MIN_SHARPNESS=1.2` | mix_fp_hits.py | high-vote/low-sharpness hits (repeat-heavy refs!). SILENT |
| 5 | z-score<1.0 band gate | mix_fp_hits.py:91 | strong absolute peaks in noisy bands (band-relative calibration). SILENT |
| 6 | `gate_z=1.0` | fp_placement_refine.py:153 | refined fp argmax → coarse fallback. SILENT |
| 7 | `gate_out_frac=0.8` (acap) | joint_ref_decode.py:267 | padded-decode evidence; loud-ish (retry logged), discarded segments silent |
| 8 | `gate_out_frac`/`weave_rate_margin=0.8` (instr) | joint_ref_decode.py:296 | wide-span rungs, cumulative gate. SILENT per rung |
| 9 | tempo_ratio∉(0.9,1.15)→oddratio | path_decode.py:382 | hard zero in fiber-credit path for odd-ratio spans |
| 10 | `_EQ_DELTA=0.04` | continuity_refine.py:59 | repeat-equivalent picks at slight score penalty. SILENT |
| 11 | `place_joint→None` | stem_placement.py:100 | HuBERT placement abstains with no reason emitted |

Class B (soft, keep but watch): stem_placement guard 8s (may be too loose on the
tail — the known <4s regression), Viterbi lam/lam_back, curve weights, monotonic
min_step, fiber min_voiced_frac 0.4, lyrics MIN_DISTINCT. Class C (benign):
duration clamps, DTW corridor, stretch grids, looptrace search bounds.

## Priority (evidence-strength × silence × known victims)

1. **fp gate 90s → strength-conditional override** (votes≥~100 AND sharpness≥~5
   breaks the leash; medley noise floor is ~5 votes/1.2 sharp). Design note: the
   gate WAS GT-validated globally (band sweep, 90s optimal) — the counterexample
   shows it fails in the MEDLEY REGIME specifically; the override must preserve
   the global win (guard p90 on re-run).
2. **fp hit floors (votes 25 / sharpness 1.2 / z 1.0)** → emit as graded
   ProbeFactors instead of pre-filtering (this is the W2 lane, bottom-up).
3. **instr gate 90s** — same fix as #1, instrumental lane.
4. **gate_out_frac cumulative rung gate** — re-examine against wide-span GT.
5. **Loudness pass for every A-gate**: each firing emits a span-level flag into
   the timeline (`gated: {rule, evidence_strength}`) — abstentions become data
   (kernel law: diagnostics as values). Cheap, do alongside #1.

## Ledger

| date | gate | change | board before → after |
|---|---|---|---|
| 2026-07-10 | fp_placement_gate_s (A#1) | strength override (votes≥100 ∧ sharp≥1.2 breaks the leash) + LOUD `placement_gated` span flags | BB12: place med 4.8→**4.4s**, <15s 68→**75%**, p90 **44s** (guard passed; gate-era bar 61s), ref med 15.6→13.0s, headline 45→44 (noise). 2/2 predicted prisoners freed, 0 collateral: Honest Virtu 123s→**2.1s**. Calibration note: sharpness metric = best-vs-second CANDIDATE (decode_placements), not window-histogram — first attempt at 3.0 fired zero; the loud flags diagnosed it in one read. sharp<1.0 = monotonic-decode override (genuine ambiguity, keep gated). |
