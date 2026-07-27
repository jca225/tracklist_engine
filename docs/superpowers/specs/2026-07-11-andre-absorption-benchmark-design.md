# André-absorption benchmark — design spec

**Date:** 2026-07-11
**Status:** approved thesis, pre-implementation
**Owner:** John
**Home:** `alignment/external/`

## Authorization note (freeze exception)

`alignment/CLAUDE.md` declares a **sensor-phase freeze**
("do NOT add new probes/channels/priors"). John has **explicitly lifted the
freeze for this work** (2026-07-11). This spec therefore permits new channels —
specifically the resample pitch-search arm (§Phase 2). An implementation task
will amend the CLAUDE.md freeze note to record this exception so the next agent
doesn't treat the new code as a freeze violation.

## Thesis

We are **not** competing with André et al. 2024 (multi-pass NMF DJ-mix
transcription, arXiv:2410.04198) head-to-head on their leaderboard. We are
demonstrating that **their method is one *mode* of a more general framework** —
the affine-warp, closed-set, always-commit, gain-transcription corner of an
aligner that also does open-set identification, non-affine trajectory decode,
stem-wise multi-axis matching, and calibrated abstention.

The claim is a **subsumption**, earned by a concrete *reduction*, not a
rhetorical reframe:

1. Run our aligner in **"André mode"** — closed-set, affine warp, no-abstain,
   gain-on — on UnmixDB and show it reproduces their setting to within their
   reported warp error.
2. Show the capabilities that light up when the restrictions are *removed*
   (non-affine warps, open-set ID, abstention) — regimes their affine closed
   NMF cannot *represent*.
3. Report, honestly, the axes where the specialist still leads (dense gain
   fidelity, synthetic pitch-resample robustness).

**"More general" does not mean "dominates every number."** A generalist absorbs
a specialist by containing its setting as a special case. The paper must say
where the specialist wins.

## What already exists (do not rebuild)

- `external/eval_bench.py` — the harness. Contracts already present:
  `GTSpan(set_start_s, tempo_ratio)`, `Sample`, `Pred(set_start_s, tempo_ratio,
  score)`, `Method = Callable[[Sample], dict[int, Pred]]`. Registered methods:
  `method_grid_mf`, `method_grid_locked`, `method_nmf`, `method_fused`.
  `score_sample` computes set_start / tempo error; `summary` emits the MAE
  table; `run_identity` scores open-set ID with distractors.
- `external/nmf_baseline.py` — **André's method reproduced as a callable
  component of our repo** (fixed-W KL-NMF, affine warp from the activation
  ridge, gain from the activation envelope). This *is* the physical embodiment
  of "absorption." `method_nmf` already wraps it.
- `external/unmixdb_eval.py` + `external/out/unmixdb_bench.txt` — the shipped
  720-span, 240-mix stratified probe run (per warp×effect stratum). Honest
  standing already written in `external/unmixdb_findings.md`.
- `path_decode.py` — Viterbi over ref-offset (loop/jump/odd-ratio): the
  **non-affine** decoder for the superset demo.
- UnmixDB v1.1 present locally: `~/data/unmixdb-v1.1` (8 GB). Runs on the Mac.

## Goals

1. A **reduction table**: our system in André-mode vs `nmf_baseline` vs André's
   published figures, on the *same* UnmixDB sample, on André's *own* units
   (warp MAE + gain MAE, per warp×effect condition).
2. A **superset demonstration**: score `path_decode` non-affine trajectories and
   the open-set/abstention posture on cases the affine closed NMF cannot express.
3. An honest **per-axis limitations** section (resample, dense gain).
4. Optionally, a **resample pitch-search arm** that narrows the one axis where
   the specialist leads (freeze-exception work).
5. The **internal findings paper** built on the measured reduction, not the belief.

## Non-goals

- A leaderboard "we beat SOTA" claim. Explicitly rejected.
- Physically reorganizing the flat top-level modules. Fan-in is prohibitive
  (`refine_ref_offsets` 43 importers, `path_decode` 27, `mert_store` 21) — a
  100-file rewrite for no capability. Out of scope; a re-export shim is a
  separate, optional cosmetic task.
- Multi-set BB co-train, new GT sets — unrelated to this benchmark.

## Architecture — extend `eval_bench`, don't fork it

Four gaps between the current harness and a full André-metric table. All are
additive changes to existing units:

| Gap | Change | Unit | Cost |
|---|---|---|---|
| **abstain vs no-abstain modes** | a mode flag: André-mode forces best-candidate commit and reports 0 abstentions; open-mode keeps the confidence floor and reports **abstain rate as a first-class column** (a capability, not a penalty). | `eval_bench` | small |
| **per-condition stratification in the MAE table** | parse the warp×effect stratum from `mix_id` (`set<NNN>mix3-<warp>-<effect>-<NN>`); `summary` groups by stratum so the reduction table matches André's Fig. 4 axes. | `eval_bench` | small |
| **DTW baseline** | add `method_dtw` (chroma/feature DTW, mix-window vs ref) — André's *warp-error winner*, so the reduction bar isn't understated by NMF alone. | `eval_bench` | medium |

**Gain — deferred, and honestly scoped.** UnmixDB *has* gain GT (fadein/fadeout
trapezoids in the labels), but (a) `UnmixTrackSpan` doesn't expose the envelope,
(b) `NmfPred` surfaces only a scalar `gain_peak` in un-normalized activation
units, and (c) **our fp/fused methods emit no gain at all** — only the absorbed
NMF does. A dense gain-MAE column would therefore score the NMF component alone,
noisily. Gain is a **named limitation on-thesis** (the NMF *mode* models gain;
our front-end does not) and gets a separate later task that extends
`UnmixTrackSpan` with a `gain_envelope` and `NmfPred` with the curve. It is NOT
in the Phase 0 spine.

### Data flow

```
UnmixDB mix + known sources + affine GT
        │
        ├── method_nmf        (nmf_baseline: warp + gain)      ── André, reproduced
        ├── method_fused      (our fp/chroma, André-mode)      ── our system, restricted
        └── method_fused_open (our fp/chroma, abstain-on)      ── our system, unrestricted
        │
        ▼
score_sample  →  {set_start_err, tempo_err, gain_err, abstained}
        │
        ▼
summary (stratified warp×effect)  →  the reduction table
```

Same `Method` signature for all three columns — that uniformity *is* the
subsumption argument in code: André's method and ours are interchangeable
callables scored on one metric.

## Metrics (André's units)

- **warp error** = set_start MAE + tempo MAE (single-anchor affine warp).
- **gain error** = MAE of predicted vs GT gain (new).
- **abstain rate** = fraction of GT spans declined (0 in André-mode; reported in
  open-mode — this is a *capability*, not a penalty, and the table labels it so).
- **identity accuracy** = `run_identity` open-set with distractors (André does
  not do this at all; reported as a superset capability).

## Phases

### Phase 0 — reduction table (the spine)
Wire gain + tempo-slope + no-abstain + stratification into `eval_bench`. Run
`method_nmf` and `method_fused` (André-mode) on the same 240-mix sample. Emit
the two-column reduction table on André's units. **This is the load-bearing
result; it is mostly plumbing over existing code.**

### Phase 1 — superset demonstrations
Score `path_decode` non-affine trajectories and open-mode abstention/open-set ID
on the common metric. Cheap: the decoders exist; we are scoring them, not
building them. Shows what our framework expresses that André's cannot.

### Phase 2 — resample arm (freeze exception)
The one genuine research build: a pitch-search (or CQT-domain constellation)
wrapper over `landmark_fp` so the fingerprint can follow a pitch-resampled
diagonal. Isolated `Method` variant. **If it lands, it narrows the specialist's
last advantage; if it doesn't, "we abstain on resample" stays an honest, named
limitation.** Either outcome is a valid paper result.

### Phase 3 — cull + shim (cosmetic, optional)
Move the ~3 confirmed orphans (`infer_fused`, `enhance_vocal`, `fp_probe`; verify
`tempo_curve`/`export_mert_from_pi` per-file) to `attic/` and update the ledger.
Skip the physical reorg. Offer a re-export shim only if still wanted after seeing
fan-in numbers.

### Phase 4 — findings paper
Draft the internal methods/findings report around the measured reduction table +
superset demos + honest limitations. Structure: reframe result (placement wall =
decomposition error) · fibers/looptrace as first-class self-repeat objects ·
opinion-audit methodology · the André-absorption reduction · named costs.

## Testing / validation

- `eval_bench --synthetic` must stay green (feature-space smoke test, no audio) —
  the new gain/tempo/abstain code paths get a synthetic fixture first.
- Reduction table reproducibility: fixed sample seed (seed 0, 240-mix stratified),
  matching the shipped `unmixdb_bench.txt` sampling, so numbers are comparable.
- Gain metric sanity: on clean/none mixes, `method_nmf` gain error should be small
  (it computes gain directly); a large error signals a wiring bug, not a finding.
- Resample arm: gated on the synthetic + a held-out resample stratum; report
  before/after detect + warp error on the resample axis only.

## Risks & named limitations (must appear in the paper)

1. **Dense gain fidelity** — we report a single affine warp + coarse gain; André
   produces a dense per-frame gain. On gain fidelity the specialist may lead.
2. **Synthetic pitch-resample** — our fp is pitch-preserving by construction;
   without Phase 2 it abstains on resample (detect 40% on that stratum). This is
   structural, not noise.
3. **Domain mismatch** — our probes are BB-mashup-tuned (pop/vocal); UnmixDB is
   synthetic electronic. The chroma floor already had to be lowered to run at all.
   The reduction is "our framework instantiated in their regime," with this caveat
   stated.
4. **"Absorb" overclaim guard** — the paper states explicitly where the specialist
   wins. Subsumption = contains-as-special-case, not dominates-everywhere.

## Resolved decisions (2026-07-11)

- **Phase 2 method → pitch-search grid over `landmark_fp`, CQT as a documented
  fallback only if the grid plateaus.** A pitch shift is a log-frequency
  translation; the grid exploits the same structure the CQT would make invariant,
  but *reuses the proven constellation matcher* instead of reimplementing landmark
  extraction + hashing (the one component that currently works perfectly). Grid:
  ~±6 semitones at half-semitone steps (~24 bins) covers UnmixDB's 0.75–1.32 warp
  range; find the pitch bin, then local-refine for sub-bin offset precision.
  Embarrassingly parallel (`multiprocessing.Pool`). CQT only if the grid's
  discretization proves insufficient — unlikely given 0.02 s offset precision once
  in the right bin.
- **Language → Python.** Hot loops are already native (scipy FFT, numpy/BLAS NMF
  and correlation). This is offline benchmark code (240 mixes, a few runs);
  correctness and dev velocity dominate runtime. Escape hatch for a profiled
  bottleneck: numpy-vectorize → `numba` → `cython` → single-function `pyo3`, never
  a wholesale rewrite. No C++/Rust adoption for this program.
- **DTW baseline → include it.** André's DTW baseline is his *warp-error winner*;
  reproducing only NMF would understate the bar. Add `method_dtw` (chroma/feature
  DTW between mix window and ref) alongside `method_nmf` in the reduction table.
