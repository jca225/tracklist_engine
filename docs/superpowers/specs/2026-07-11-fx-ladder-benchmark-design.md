# FX-Ladder Robustness Benchmark (SP1) — design spec

**Date:** 2026-07-11
**Status:** approved design, pre-implementation
**Owner:** John
**Home:** `alignment/synthetic_mix/` + `external/eval_bench.py`

## North star (session)

Make the DJ-mix alignment work **publishable**. The defensible novelty is not
"we beat André" — it is a **transformation-robustness benchmark** for DJ-mix
alignment: a synthetic ladder of production transforms (clean → pitch/tempo →
FX) with perfect ground truth, plus a real-BB transfer rung, measuring **where
each alignment channel breaks and how abstention converts silent failure into
honest decline**. Nobody has published this for DJ-mix alignment.

This spec covers **SP1 only** — the synthetic FX-ladder benchmark + baseline
degradation profiles. Two follow-on sub-projects get their own cycles:
- **SP2** — real-BB transfer rung (label a few BB sets via the bootstrap flywheel +
  Ableton pipeline; show synthetic→real transfer). Gated on labeling budget.
- **SP3** — the paper, written around SP1 profiles + SP2 transfer + an honest
  "re-instrumentation / live is the open frontier" section.

## Thesis of the benchmark

The contribution is the **degradation-and-abstention map**, not an MAE number:
a profile of each channel's accuracy vs transformation severity, showing which
channel dies where (fp under filter sweep; chroma under key change; HuBERT
survives on vocals; NMF handles superposition), and how the abstaining methods
decline honestly rather than lie as severity climbs.

## Key simplifying insight

**FX is a nuisance transform that colors audio but does not move the notes.**
Reverb / filtering / delay / gating / drive change timbre, not *where* a track
sits in the mix or its warp. So the alignment ground truth (`set_start_s`,
`tempo_ratio`, `ref_segments`) is **invariant under the FX rung**. We reuse
`labels_v2`'s `GroundTruthSet` untouched and vary only the audio coloring — which
is what makes perfect GT free at every FX rung.

## What already exists (reuse, do not rebuild)

- `synthetic_mix/` renders BB-realistic mashups with multi-layer superposition,
  time-stretch + pitch-shift (`_stretch_pitch_vocal`), gain/fader envelopes,
  crossfades, loops, stacked vocals (`render_v2.py`, `scenario_v2.py`). Rungs 0–1
  (clean, pitch/tempo) already exist.
- `labels_v2.window_to_gt` emits `GroundTruthSet(GroundTruthTrack(set_start_s,
  tempo_ratio, ref_segments, ...))` — the GT format the aligner consumes.
- `external/eval_bench.py` — `Sample`/`GTSpan`/`Pred`/`Method` contracts, methods
  `fused`, `fused_resample`, `dtw`, `nmf`, `grid_mf`, chroma/HuBERT identity,
  `score_sample`, `summary_by_stratum` (already stratifies by a 2-tuple).

## Architecture — three new/extended units

### 1. `synthetic_mix/fx.py` (new) — parametrized FX with a severity knob

Each effect is a pure function `apply(y: np.ndarray, severity: float, rng) ->
np.ndarray`, `severity ∈ [0,1]` (0 = identity, 1 = extreme). Effects (v1 set):

- `reverb` — convolution/feedback reverb, severity = wet mix + decay (dry→hall).
- `filter_sweep` — time-varying lowpass or highpass, severity = cutoff excursion.
- `delay` — echo, severity = feedback + wet mix.
- `gate` — amplitude gating/tremolo, severity = depth.
- `drive` — soft-clip distortion / bitcrush, severity = drive amount.

Applied **per track/deck before the sum** (a DJ FX affects one deck), so the mix
tests robustness to *layered* FX, not a single global effect. FX is coloring only
— it never shifts sample timing (no time-variant resampling), preserving GT.

A dispatch `apply_fx(y, fx_type: str, severity: float, rng) -> np.ndarray` and
`FX_TYPES: tuple[str, ...]`.

### 2. `synthetic_mix/ladder.py` (new) — matched-rung generator

`generate_ladder(n_mixes, seeds, fx_types, severities, out_dir) -> LadderManifest`.
For each base mashup (fixed seed → fixed `GroundTruthSet`), render matched
variants:
- **R0** clean (no warp, no FX).
- **R1** +pitch/tempo (existing warp path).
- **R2** +FX: for each `fx_type × severity` in the grid, apply per-deck FX on top
  of R1's audio.

Emits, per variant, a FLAC + a shared GT JSON (the same GT across a base's
variants) + a manifest row `(mix_id, base_seed, rung, fx_type, severity, audio_path,
gt_path)`. `LadderManifest` = the list of rows, written as JSON.

### 3. `eval_bench` adapter `synthetic_ladder_samples` (extend)

`synthetic_ladder_samples(manifest_path: Path, feature: str) -> list[Sample]`.
Loads each manifest row into a `Sample` (mix audio + candidate track features/paths
+ GT spans from the shared GT), tagging `Sample.mix_id` so the stratum key encodes
`(fx_type, severity)`. Every existing `Method` runs unchanged.

## Metrics — the degradation profile

Reuse `eval_bench` metric machinery; stratify by `(fx_type, severity)` instead of
`(warp, effect)`.

- **Degradation curves** — per method: set_start MAE, `<2s` hit-rate, tempo MAE
  vs `severity ∈ {0, 0.25, 0.5, 0.75, 1.0}`, per `fx_type`.
- **Break-point table** — per (method, fx_type): the severity at which `<2s`
  drops below 50% (the "which channel dies where" map).
- **Abstention overlay** — decline-rate vs severity (abstain-not-lie as a curve).

Output: a stratified table (extend `summary_by_stratum` to take a stratum key
function, or add `stratum_ladder(mix_id) -> (fx_type, severity)`) written to
`synthetic_mix/out/fx_ladder_profile.txt`, plus a findings section.

## Baselines

`fused`, `fused_resample`, `chroma` (via `grid_mf`/`method` on chroma features),
`dtw`, `nmf`, and HuBERT identity (`run_identity` mode). All exist; the benchmark
runs them, it does not build them.

## Testing / validation

- `fx.py`: each effect has a unit test — `apply(y, 0.0) == y` (identity at
  severity 0) and `apply(y, 1.0)` changes the signal (e.g. spectral centroid
  moves for filter_sweep; RMS tail grows for reverb). Effects must not change
  length (GT-invariance) — assert `len(out) == len(y)`.
- `ladder.py`: a tiny 2-mix ladder generates without error and every variant of a
  base shares byte-identical GT JSON (the GT-invariance guarantee, tested).
- adapter: `synthetic_ladder_samples` on the tiny manifest yields `Sample`s whose
  `mix_id` parses to the right `(fx_type, severity)` stratum.
- A `--smoke` path generates ~4 mixes × 2 fx × 2 severities and runs 2 methods, to
  keep CI fast; the full profile run is a separate offline command.

## Risks & honest limitations (must reach the paper)

1. **Synthetic FX ≠ real DJ FX.** Parametric DSP effects approximate but don't
   equal a real DJ's hardware/software FX chains. SP2 (real-BB) is what grounds
   this; SP1's synthetic profiles are controlled, not realistic-by-fiat.
2. **Per-deck FX before sum** is one modeling choice; real FX can be on the master
   or send-based. Documented as a v1 assumption.
3. **GT-invariance-under-FX assumes FX is time-preserving** — no FX that resamples
   or time-warps (those belong to rung 1). Enforced by the length-preservation test.
4. **Re-instrumentation / live is out of scope** — named as the open frontier, not
   benchmarked here.

## Non-goals

- Real-BB labeling (SP2).
- The paper draft (SP3).
- New alignment *methods* — SP1 benchmarks existing channels; the resample arm was
  the last method addition (Phase 2).
- Re-instrumentation / live rung.

## Open questions for the plan

- Reverb/delay implementation: pure-numpy convolution vs a light dependency
  (`scipy.signal` only, to avoid new deps). Decide in writing-plans — prefer
  `scipy.signal`/numpy, no new dependency.
- Whether `chroma` and `hubert` baselines run as placement methods or only
  identity — decide per how `eval_bench` currently exposes them.
