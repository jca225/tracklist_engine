# Plan: stem-routed identity + placement (make acappella & instrumental work)

Status: draft 2026-06-29. Living doc.

## Goal

Stop routing stems through the full-song tool. Send **vocal stems → HuBERT
verify**, **instrumental stems → fingerprint**, fuse by the `claimed_stem` axis,
and prove a per-axis improvement on frozen BB12 + BB11 GT.

Both acappella and instrumental are weak today for the *same* reason: they fall
back to the full-mix chroma/MERT path, which is the wrong tool for a stem.

## Coordination note (read first)

The parallel aligner agent already shipped the front half of the acappella lane:
commit `8ad5be5` — "per-stem acappella set_start via banded HuBERT"
(`--stem-placement`, `stem_placement.place_joint`). So:

- **Acappella *placement*** is partly theirs. Do not edit `infer.py` /
  `stem_placement.py` in place while they hold it — wire new behavior behind flags
  in new modules.
- **Default lane split:** this agent takes **Phase 2 (instrumental + stem
  fingerprint index — un-owned)**; coordinate on Phase 1 (acappella *identity*).

## Guiding constraints

- **Per-axis eval, always.** Report acappella / instrumental / regular
  *separately*, abstain-aware. Never a blended median (collapse-ladder lesson — a
  blend hides which channel moved).
- **Frozen GT as judge:** BB12 (`1fsnxchk`) + BB11 (`2nvzlh2k`). Both have mix
  stems on disk.
- **New modules over in-place edits** to the hot files.

---

## Phase 0 — Coordinate + baseline (do first; no research risk)

1. Pull + read the parallel agent's recent commits/log. Confirm what
   `--stem-placement` covers (acappella set_start) vs not (acappella identity, all
   instrumental). Claim un-owned lanes.
2. Freeze the "before" numbers: `infer` + `score_timeline_vs_gt --fibers` on BB12
   and BB11, split by axis, reporting identity %, placement median, offset median
   each. Every later claim is measured against this.

**Exit:** baseline table + confirmed lane split.

---

## Phase 1 — Acappella identity (coordinate; likely partly theirs)

Placement half shipped. The which-song-for-a-vocal half is the gap.

1. Verify-not-decode via HuBERT-L9: reuse `candidate_vocal_gate.py` /
   `similarity_probe.py --feature hubert` to check the claimed acappella matches
   the actual vocal stem, instead of decoding from a big pool (81% retrieval@1,
   +0.233 margin probed).
2. Stem-routed channel: mix-vocal-stem → ref-vocal-stem (lifted acappella id
   0–14% → 84% in `stem_match_probe.py`).
3. Fix the `--stem-placement` precision regression (<15s 61→76% but <4s 44→34%):
   tune the fusion guard so HuBERT placement overrides only where confident; keep
   the precise fp/chroma start where it already won.

**Exit:** acappella identity + placement up vs baseline, <4s no longer regressed.
**Risk: low–medium.** **Overlap risk: high — confirm before touching
`stem_placement.py`.**

---

## Phase 2 — Instrumental channel (un-owned hard part — primary focus here)

Least-solved channel: chroma → 0% recall, because the separated instrumental is
contaminated by other layered tracks. Fix = the missing stem fingerprint index.

1. Build stem fingerprints: populate `track_fingerprints` for
   `stem='instrumental'` (refs) and hash the `mix_instrumental` stem. The
   fingerprint index today is full-mix / full-song only. Doubles as a down payment
   on the corpus-scale index.
2. Route instrumental spans → fingerprint, not chroma. Fingerprint votes for the
   right track despite crosstalk (its strength — ignores contaminating layers).
3. Abstain fallback: weak votes → abstain, never a confident-wrong instrumental.

**Exit:** instrumental recall moves off 0% on BB12/BB11. **Risk: medium–high —
research bet, not a sure win.** Abstain path keeps a failure graceful.

---

## Phase 3 — Repeated-chorus disambiguation (secondary)

1. Fibers declare equivalence when choruses are truly interchangeable (53→59%
   fiber-aware — keep).
2. Wire `continuity_refine.py` (continuity-stack) so where-in-song uses neighbor
   context to break the chorus tie when it matters (different ad-lib / build vs
   drop).

**Exit:** offset median improves on multi-chorus spans without hurting
interchangeable ones. **Risk: medium.**

---

## Phase 4 — Better source stems (optional, parallelizable)

When a real external stem exists, match against it instead of a noisy separation:
staged Discord library (~2,474 real acap/instr on pi) via `match_stem_library.py`,
always additive / `is_reference=0`. Routing, not replacement.

---

## What NOT to do

- Touch `infer.py` / `stem_placement.py` while the other agent is mid-flight.
- Report a single blended "stem accuracy" number.
- Over-invest in instrumental before Phase 0 proves the baseline.

## Recommended sequencing

Phase 0 → Phase 2 (instrumental + stem fingerprint index), coordinating with the
other agent on Phase 1; Phase 3/4 after.
