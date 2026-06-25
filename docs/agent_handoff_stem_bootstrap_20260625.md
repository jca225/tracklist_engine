# Agent handoff — stem→stem bootstrap + alignment standing (2026-06-25)

Pick-up doc for a new agent. Covers what this session did, the measured numbers,
the corrections to prior claims, where the alignment objective actually stands,
and the decided next build (realistic synthetic-mix generator).

## How the session evolved

Started as "are we SOTA on DJ-mix alignment?" (André 2024 multi-pass NMF on
UnmixDB is the line). Pivoted — at the user's direction — to **bootstrap the one
empty column in the prior-art table: vocal/instrumental-aware alignment** ("the
open lane"). User's litmus was explicit: **robustness, not accuracy on any single
example.** The concrete ask: "take acappella→acappella and instrumental→
instrumental matches on a real set, and/or synthetically overlay songs, and find
the breaking point."

## What was built

**`workspaces/alignment_prototype/stem_match_probe.py`** — new module, three arms:

- `--arm synthetic` — sum K real vocal (or instrumental) stems into a DIRTY mix
  channel at known offsets/gains; PERFECT ground truth. For the host stem recover
  (a) its offset within its own ref and (b) its identity vs a distractor pool.
  Sweep K = overlay depth → the breaking-point curve. Compares features
  (chroma/mfcc/hubert). Self-contained (no pi/DB/model).
- `--arm real` — BB12 `mix_vocals.flac` vs ref vocal stems on actual GT acappella
  spans (and `mix_instrumental` vs ref instr on instr spans). Tests survival of
  REAL Roformer separation noise.
- `--arm routed` — route each GT span to its best engine (vocals→hubert,
  instrumental→chroma); reports the stem-routed aligner's per-span scorecard.

Reuses (did NOT rebuild): `refine_ref_offsets.detect_offset/correlate_window/chroma`
(matched filter), `section_hsmm/similarity_probe._mfcc/_hubert` (features). Local
stems live in `data/mashup_compat/stems/<track_audio_id>/{vocals,instrumental}.flac`
(82 vocal, 57 instrumental); recording_id→taid resolved via one cached pi query
(`_resolve_local_stem_dirs`, writes `data/mashup_compat/taid_map_<stem>.json`).

### Run commands
```bash
# breaking-point curve (cheap, no GPU)
venvs/audio/bin/python -m workspaces.alignment_prototype.stem_match_probe \
  --arm synthetic --stem vocals --features chroma,mfcc --depths 1,2,3,4,5,6 --trials 25
# real BB12 acappella, stem-routed identity
venvs/audio/bin/python -m workspaces.alignment_prototype.stem_match_probe \
  --arm real --stem vocals --features hubert --extra-distractors 10
# headline: routed dual-channel scorecard
venvs/audio/bin/python -m workspaces.alignment_prototype.stem_match_probe \
  --arm routed --extra-distractors 12
```
GPU note: HuBERT matched-filter is 768-dim → each correlation ~64× chroma. Keep
`--extra-distractors` ≤ ~12 for HuBERT; the full-pool (81) HuBERT run is an
O(spans·pool·768-FFT) blowup (>30 min) and was deliberately abandoned.

## Measured results (BB12 `1fsnxchk`)

**Synthetic breaking-point** (vocals; margin = host_peak − best_distractor):

| depth | chroma | mfcc | hubert-L9 |
|---|---|---|---|
| 1 | 0.222 | 0.408 | 0.617 |
| 2 | 0.137 | 0.259 | 0.457 |
| 3 | 0.037 | 0.085 | 0.212 |
| 4 | −0.014 | 0.021 | 0.001 |
| 5 | −0.048 | −0.037 | −0.035 |

HuBERT margin 3–6× chroma through depth 3; ALL features collapse at depth 4–5.
Realistic DJ load (1 host + maybe 1 overlay = depth 1–2) sits where HuBERT is
strongest. Timbre (mfcc) > harmony (chroma).

**Real, stem-routed identity** (vs CLAUDE.md baseline: chroma-on-FULL-mix acappella
= 0% exact / 14% repeat-equiv):

| channel | engine | n | chroma id | hubert id | placement |
|---|---|---|---|---|---|
| vocals | HuBERT | 83 | 47% | **84–87%** | median 7.4s, <2s 39% |
| instrumental | chroma | 22 | **86%** | 91% (tie, n=22) | median 0.1s, <2s 67% |
| **ROUTED combined** | routed | 105 | — | **86% (90/105)** | median 5.8s, <2s 44% |

**Hardening:** full-pool distractor test (10 → 81/56 candidates) drops chroma only
modestly (vocals 47→40%, instr 86→77%) → matched-filter peak is genuinely
discriminative, not an easy-pool artifact.

### Three findings that reframe the open lane
1. **Placement is solved when identity is right** — the matched filter on the clean
   isolated stem nails the offset to the frame. The open lane is a **discrimination-
   under-crosstalk** problem, not a localization one.
2. **Vocals and instrumentals are OPPOSITE problems with OPPOSITE engines.** Vocals
   = overlaid layer (crosstalk + re-sung-chorus repeat ambiguity) → need HuBERT
   phonetics (47→84%). Instruments = host/bed (high-SNR) → chroma already gets 86%,
   frame-exact. HuBERT is vocal-specific (neutral on instruments — no lyrics).
3. **The wall (4+ simultaneous distinct vocals) sits just past realistic DJ load.**

### Caveats (do not oversell)
- 86% is a **per-span, ORACLE-position** number (scored at GT set_start).
  **End-to-end (no oracle), BB12 is ~66% id / ~30s placement** — whole-set
  placement + repeat ambiguity dominate. Probes plateau ~55–60% end-to-end.
- Synthetic sums CLEAN stems (isolates crosstalk; excludes separation artifacts/
  reverb/DJ-EQ — the real arm includes those).
- n modest (12–25 synth trials/depth; 83+22 real spans); distractor pool 10–81,
  not the full ~150-track tracklist.

## Corrections to PRIOR claims (important — earlier statements were WRONG)
- **André 2024 code is DELETED**, not public. `github.com/etiandre/icassp2025-dj-
  transcription` AND the whole `etiandre` account 404; never mirrored/archived
  (Software Heritage has 15 of his OTHER repos, proving the account was real — this
  one absent). "Run their public code for the ceiling" is a dead path. Routes to
  their method: email IRCAM (andre@ircam.fr / schwarz@ircam.fr) or reimplement
  multi-pass NMF from the paper. Dataset generator survives:
  `github.com/Ircam-RnD/unmixdb-creation`.
- **Local `~/data/unmixdb-v1.1` is the `-excerpts` subset** (6 base mixes ×3
  timescale ×4 FX ×~44 segments = 2460 renders, 40s refsongs), NOT the full
  1931-mix benchmark. So `eval_bench.py` numbers (fused id 82.6%, placement 2.39s
  median) are on a 6-mix slice, not the published benchmark.
- Fixed in: memory `project_dj_mix_prior_art`, `docs/editable_reconstruction.md`.

## Commits this session
- `d1b236a` docs/editable_reconstruction.md (editable trio: placement/gain/EQ)
- `1dfe942` feat: stem_match_probe (synthetic+real arms) + prior-art corrections
- `0e5fd32` fix: defensive slot_label in real arm
- `8fc315c` feat: --arm routed dual-channel scorecard + full-pool hardening

## Where the ALIGNMENT OBJECTIVE stands (docs/alignment_objective.md)
- **A. GT reader (.als→data model)** — ~80% DONE. Round-trip works (parse∘print
  identity on BB12 152/152); captures BPM/volume/gain/warp/sections. Gaps: key-
  change + FX extraction partial; substitutability flag ad-hoc.
- **B. The aligner** — **NOT built as a trained model.** Only hand-built probes
  measuring the emission signals. B1 (warp) effectively in hand (grid-lock + stretch
  search, ~1.2% err). **B2 (key) NOT done** (only key-INVARIANT matching exists).
- **C. Manual labeling** — thin + PAUSED (BB12 full 166 rows; BB11 partial; paused
  2026-06-17). Spec calls this the gating dependency. Re-sourcing affected audio is
  on the critical path (much bulk audio predates ingest fixes → wrong-version bugs).
- **Success criterion** — far off. Tolerances (±N bars, ±M BPM) still UNSET;
  generalization to 20k unproven (no trained model, measured on 1–2 sets).

## DECIDED next build — realistic synthetic-mix generator (pretrain→finetune)
User proposed (correctly) generating training data by permuting corpus stems. This
IS the planned unlock (`project_alignment_bootstrap_flywheel`: one model, 2 phases
synthetic-pretrain→real-finetune). Agreed plan, with the critical constraint:

**THE TRAP: random permutation ≠ real mixes.** Random stacks are great for
robustness stress-testing but wrong for TRAINING (distribution real mixes never
occupy → poor transfer; this is UnmixDB's known weakness). The value is entirely in
making synthetic mixes REALISTIC:
- beatmatch overlay→host BPM (grids: `set_measures`/`track_measures`),
- key-shift to a compatible key (→ gives B2 key labels for free),
- structured overlap (1 host + 1–2 overlays, gain fades at edges; NOT 6 random),
- curriculum easy→messy.
- emit labels in the **.als GT schema** so synthetic is drop-in with real GT.

**What it does NOT solve:** (1) corpus-scale open-set identity (that's a fingerprint/
ANN index problem, not data); (2) real GT doesn't go to zero — still need a small
real held-out set to finetune the synthetic→real gap AND to PROVE generalization
(C converts mostly from training-data into validation-data).

**Raw material available:** 549 sets on pi with `{vocals,instrumental}.flac`; the
~2,474-file Discord stem corpus (`project_discord_stem_corpus`, staged at
pi:/mnt/storage/staging); grids + key/BPM analysis; the .als GT schema.

**Build in 2 fail-fast steps:** (1) generator → a few hundred labeled synthetic
spans + ear-check they sound like plausible mixes; (2) train the existing aligner
head (`MertAlignHead`) on it, validate on BB12/BB11 held-out real GT = first honest
"does synthetic pretrain transfer?" number.

## Key memory pointers
`project_stem_match_bootstrap` (this session's numbers), `project_dj_mix_prior_art`
(corrected SOTA standing), `project_alignment_bootstrap_flywheel` (pretrain plan),
`project_stemwise_alignment`, `project_hubert_beats_mfcc_vocal`,
`project_collapse_ladder_findings`, `project_labeling_plan_2026_06` (resume GT),
`project_aligner_attention_design`. Objective: `docs/alignment_objective.md`;
program plan: `docs/alignment_program_plan.md`.
