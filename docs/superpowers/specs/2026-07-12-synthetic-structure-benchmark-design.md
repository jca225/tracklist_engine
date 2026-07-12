# Synthetic Structure Benchmark — Design (APPROVED 2026-07-12)

**Date:** 2026-07-12
**Status:** **APPROVED by John** (design + scope: all 4 knobs, OFAT, oracle-identity,
structure-primary metric, validation gate). Resume at **writing-plans → subagent-driven
build** — do NOT re-litigate the design; go straight to the implementation plan.
**Context:** the third leg of a 3-legged alignment dataset (UnmixDB + Big Booties +
this). Grounded in [docs/alignment_recharacterization.md](../../alignment_recharacterization.md).

---

## Why (the claim it earns)

UnmixDB dials only warp+codec (the easy axis) and structurally cannot test
structure. Big Booties has all the hard structure but confounds every factor at
once and can't be manipulated. This synthetic leg is the **controlled experiment**
neither can run: from a fixed "UnmixDB-easy" baseline, sweep each hard dimension
**one at a time (OFAT)** and measure alignment degrading — **causal degradation
curves** showing which dimension breaks which axis. It makes the
synthetic-overstates-real claim *causal*, not merely observed. Spectrum:
`UnmixDB (easy synthetic) → this (controllable synthetic) → BB (real)`.

## Locked decisions (from brainstorming)

- **Knobs: all four**, swept OFAT from baseline `{1 instance, 0 semitones, 1 layer,
  start-aligned}`:
  | knob | levels | stresses |
  |---|---|---|
  | repeat count | 1 → 3 → 6 | structure (which-instance) |
  | re-pitch | 0 → ±2 → ±5 | structure (chroma-break) |
  | medley density | 1 → 2 → 4 → 6 | structure + placement (pileup) |
  | entry point | start → mid-song | placement (heavy tail) |
  ~9 strata × ~20 mixes ≈ 180 short mixes → hundreds of spans → real CIs.
- **Metric surface: structure-primary.** Score on the structure harness
  (`score_spans` → trajectory strict + fiber, same as BB), placement (set_start
  error) secondary as the UnmixDB-overlap axis. Report oracle-placement AND e2e
  (reuse the oracle-vs-e2e decomposition) so each knob's damage shows on its axis
  (entry→placement, repeats→structure).
- **Oracle identity.** Synthetic mixes have KNOWN source tracks → hand the decoder
  the right refs. Drops the expensive MERT/fp identity step **and the pi-storage
  dependency**, and isolates the two walls (placement, structure) cleanly.
- **Short mixes (~30–90 s single mashup windows)**, not 60-min sets → render +
  decode is seconds each; whole sweep runs on the Mac, no cluster.

## Reuse vs build

- **Reuse (exists, verified):** `synthetic_mix/{generate_v2,render_v2,scenario_v2,
  labels_v2,catalog,warp_model,validate,corpus}.py` (renders `mix.flac` + stems
  from `data/mashup_compat/stems/` with tempo/pitch/loops/stacking, emits
  full-structure GT YAML); `score_timeline_vs_gt.score_spans` (structure scoring);
  `experiments/report.py` (span-bootstrap CIs).
- **Build:** (a) expose the 4 knobs as clean independent dials in scenario
  generation; (b) a thin `synth_bench` runner: generate OFAT strata → oracle-identity
  decode → `score_spans` → per-knob degradation table; (c) the validation gate.

## Validation gate (credibility keystone — non-negotiable)

Synthetic was a dead end **as pretrain** (realism); as an eval benchmark the same
risk applies — the causal claim only transfers if the synthetic is a faithful
proxy. Anchor both ends:
1. **Baseline ≈ UnmixDB-easy** — aligner must near-solve the baseline stratum
   (else "easy" isn't easy; abort).
2. **GT stats ∈ BB ranges** — span durations, tempo ratios, repeat structure within
   BB12 distributions (extend `synthetic_mix/validate.py`).
3. **Max-knob → BB wall** — at full knobs, structure score approaches BB's real
   numbers.
Gate fails → the causal claim isn't licensed; report it, do not launder.

## Interfaces to honor (verified this session)

- `score_spans(set_id, timeline_path, *, fibers, hubert_layer, gt_path)` →
  `list[SpanScore]` (score_timeline_vs_gt.py). GT via a GroundTruthSet YAML.
- Generator: `generate_v2.generate(args)` writes `synthv2_*/` + `corpus_manifest.json`;
  `labels_v2.window_to_gt(window) -> GroundTruthSet`; `scenario_v2.sample_window_v2(
  catalog, mix_id, curriculum, rng)`; `render_v2.render_window_v2(window)`.
- eval_bench (placement-only, UnmixDB overlap): `Sample`/`GTSpan(track_idx,
  set_start_s, tempo_ratio)`/`Pred`; `stratum(mix_id)` parses
  `set<NNN>mix3-{warp}-{effect}-<NN>`. Our knobs need their own stratum encoding.

## Success criteria

One command → per-knob degradation curves (structure strict+fiber + placement,
CIs) + the three-legged spectrum table + a pass/fail validation report. Paper
claim: *"holding all else fixed, repeats and re-pitch collapse structure while
entry-point opens the placement tail — the exact dimensions UnmixDB omits."*

## Resume here

- **Design APPROVED.** Next action: invoke `superpowers:writing-plans` to turn this
  design into a bite-sized TDD implementation plan, then `superpowers:subagent-driven-development`
  to build it (mirror the ablation-harness build this session — golden gates, one
  reviewer per task, span-bootstrap CIs). Reuse `synthetic_mix/` heavily.
- Self-review (this session): no placeholders; internally consistent; single-benchmark
  scope; knob levels + n=20/stratum + oracle-identity + validation bar all explicit.
