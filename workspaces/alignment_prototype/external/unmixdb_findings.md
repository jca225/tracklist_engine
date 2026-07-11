# UnmixDB external benchmark — where our probes stand vs SOTA

**What:** drives this repo's harness probes (`FingerprintProbe`, `ChromaProbe`)
through UnmixDB v1.1 (Schwarz & Fourer 2018) to get a *calibrated standing*
against the published SOTA on the same dataset (André et al. 2024, multi-pass
NMF; and its DTW baseline). Script: `unmixdb_eval.py` (this dir). Not a
leaderboard entry — a sanity check on where DSP probes tuned on Big Bootie
mashups land on a synthetic electronic benchmark.

## The dataset & GT

UnmixDB v1.1 = 2460 synthetic mixotic mixes (6 mixotic sets), each a
beat-synchronous crossfade of **3 known source-track excerpts** (~45 s each) into
a ~1–2 min mix, with perfect ground truth. Two difficulty axes, both in the mix
filename `set<NNN>mix3-<timewarp>-<effect>-<NN>`:

- **timewarp**: `none` (speed 1) · `resample` (pitch+tempo) · `stretch` (tempo only) — 820 each
- **effect**: `none` · `bass` boost · `compressor` · `distortion` — 615 each

GT time map per track is **affine**: `ref_t(mix_t) = (mix_t - start_t) * speed`,
verified against the refsong `.excerpt40.txt` cue points. We evaluate the ref
offset the probes report (fp: ref-time at mix origin; chroma: ref-time at its mix
window start), each against the matching GT scalar.

## What the SOTA (André 2024) actually reports — and the honest framing

Read directly from the paper (arXiv 2410.04198):

1. **André assumes the source tracks are KNOWN** — the NMF dictionary IS the
   source-track spectrograms. This is a *closed* setting. So there is **no
   identification accuracy** to beat; the paper states identification "is well
   handled by fingerprinting [2]–[5]" — i.e. exactly the class of method our
   `FingerprintProbe` is. Our closed-set analog is **detection**: given the 3
   known sources, did we place each one within tolerance.
2. **Their metric is warp error** = MAE of the mix→ref warp function *f* in
   seconds (plus a gain error). Fig. 4 (log scale): warp-error medians run
   **~1 s in the clean (none/none) condition up to ~10 s** under
   compression/distortion, with heavy tails to ~100 s. Their **DTW baseline is
   generally *better* on warp error** than the NMF method — NMF wins on gain, not
   warp, because NMF makes no affine-warp assumption.

So the offset bar to calibrate against is **~1 s median (clean), degrading toward
~5–10 s under effects**. Our probes' `offset_err` (|pred − GT ref-offset|) is the
directly comparable quantity (a single-anchor warp error).

## Method / caveats (read before trusting the numbers)

- **Closed identification setting**, as André's: candidate pool per mix = its 3
  known sources; we score each probe's placement of each GT source.
- **fp** runs whole-mix vs whole-ref once per candidate → ref-offset at mix
  origin (no placement prior).
- **chroma** needs a short mix window sliding over the ref (a whole-mix window
  exceeds the ~45 s ref, killing the correlation), so it's run **per span with a
  12 s window at the GT `set_start`** — i.e. chroma is *given* the coarse
  placement prior ("rough alignment") that fp derives for free. This asymmetry is
  deliberate and matches André's two-stage decomposition + the repo's own
  `refine_ref_offsets` usage; **chroma's numbers therefore flatter it relative to
  fp** and are not directly comparable to fp's.
- **fuse** = higher-confidence non-abstain probe per span, scored in its own frame.
- **Widened stretch grid** (0.75–1.30) vs the BB-tuned 0.98–1.02 default, because
  UnmixDB warps span 0.75–1.32. Offline-only; does not touch shipped defaults.
- **Chroma abstain floor lowered** to 0.2 (`--chroma-min-peak`) — the stock BB
  floor of 0.5 abstains on *all* electronic mixotic content, so the stock probe
  would score zero here. 0.2 surfaces chroma's raw placement (genericness caveat:
  low-peak chroma matches are near-chance on repetitive electronic loops).
- **Domain mismatch (the big one):** our probes are tuned on Big Bootie
  *mashups* (pop/vocal, real DJ edits). UnmixDB is synthetic *electronic* mixotic
  content with synthetic timestretch/effects. In particular `resample` shifts
  pitch, which the fp constellation (spectral peaks) and chroma (pitch classes)
  both key on — while our fp only ever time-stretches the ref *pitch-preserving*.
  So `resample` is expected to be our worst warp axis, and it is.

## Results

See `out/unmixdb_bench.txt` for the full per-stratum table from the headline run
(stratified 240-mix sample, seed 0). Reproduce / re-sample:

```bash
venvs/audio/bin/python -m workspaces.alignment_prototype.external.unmixdb_eval \
  --unmixdb-root ~/data/unmixdb-v1.1 --sample 240 --seed 0 \
  --out workspaces/alignment_prototype/external/out/unmixdb_bench.txt
```

Columns: `detect` = fraction of GT sources placed within 5 s (abstain = miss);
`abst` = # GT spans the probe declined; `med_off`/`<1s`/`<5s` = offset error over
placed spans.

**Headline run — stratified 240 mixes (20 per warp×effect stratum), 720 GT spans,
seed 0.** `detect` = placed within 5 s (abstain counts as miss); offset stats over
committed (non-abstain) spans only.

By EFFECT (each row = 180 spans):

| probe  | effect      | detect | abst | med_off | <1s | <5s |
|--------|-------------|--------|------|---------|-----|-----|
| fp     | none        | 61%    | 67   | 0.03s   | 88% | 96% |
| fp     | bass        | 57%    | 65   | 0.02s   | 83% | 89% |
| fp     | compressor  | 61%    | 65   | 0.02s   | 90% | 95% |
| fp     | distortion  | 57%    | 70   | 0.03s   | 85% | 94% |
| chroma | none        | 76%    | 0    | 0.07s   | 65% | 76% |
| chroma | bass        | 76%    | 0    | 0.05s   | 64% | 76% |
| chroma | compressor  | 78%    | 1    | 0.03s   | 67% | 79% |
| chroma | distortion  | 76%    | 0    | 0.04s   | 66% | 76% |
| fuse   | none        | 83%    | 0    | 0.03s   | 75% | 83% |
| fuse   | bass        | 80%    | 0    | 0.03s   | 72% | 80% |
| fuse   | compressor  | 84%    | 1    | 0.03s   | 76% | 85% |
| fuse   | distortion  | 77%    | 0    | 0.04s   | 68% | 77% |

By TIMEWARP (each row = 240 spans) — **the axis that actually separates us**:

| probe  | warp     | detect | abst | med_off | <1s | <5s |
|--------|----------|--------|------|---------|-----|-----|
| fp     | none     | 83%    | 38   | 0.02s   | 98% | 99% |
| fp     | resample | 40%    | 141  | 0.02s   | 92% | 96% |
| fp     | stretch  | 53%    | 88   | 0.03s   | 69% | 84% |
| chroma | none     | 82%    | 0    | 0.03s   | 74% | 82% |
| chroma | resample | 68%    | 1    | 0.21s   | 54% | 68% |
| chroma | stretch  | 80%    | 0    | 0.09s   | 69% | 80% |
| fuse   | none     | 92%    | 0    | 0.02s   | 88% | 92% |
| fuse   | resample | 70%    | 1    | 0.14s   | 57% | 70% |
| fuse   | stretch  | 81%    | 0    | 0.08s   | 72% | 81% |

OVERALL (720 spans):

| probe  | detect | abst | med_off | <1s | <5s |
|--------|--------|------|---------|-----|-----|
| fp     | 59%    | 267  | 0.02s   | 87% | 93% |
| chroma | 76%    | 1    | 0.04s   | 66% | 76% |
| fuse   | **81%**| 1    | 0.03s   | 73% | 81% |

### Read (headline 720-span run)

- **When our probes commit, the offset is essentially exact — ~0.02 s median,
  87 % of fp placements within 1 s** — i.e. ~50× inside André's ~1 s clean warp
  bar. The fingerprint constellation nails the alignment diagonal to a few frames
  whenever the content survives the transform. Precision is not our problem.
- **Our failure mode is coverage, not sloppiness.** fp *abstains* on 267/720
  spans (weak constellation votes), dragging its detection to 59 %. chroma barely
  abstains (floor 0.2) so its detection is higher (76 %) but its committed offsets
  are looser (66 % <1 s) and its tail blows up on resample.
- **The `effect` axis barely moves us** (fp detect 57–61 % flat across
  none/bass/compressor/distortion; offsets ~0.02–0.03 s throughout). This is the
  *opposite* of André, whose warp error visibly worsens under
  compression/distortion — our fp's peak-picking is largely effect-robust because
  the constellation survives additive/dynamics processing. Good result.
- **The `timewarp` axis is what separates us**, exactly as the domain caveat
  predicts. `none` is easy (fp 83 % detect, 98 % <1 s). **`resample` (pitch+tempo)
  is the wall**: fp abstains on 141/240 (detect 40 %) because pitch-shifting moves
  the spectral peaks off the ref constellation, and chroma's median offset jumps
  to 0.21 s (pitch classes rotate). `stretch` (tempo-only, pitch-preserving) sits
  in between (fp detect 53 %). Our fp only ever time-stretches the ref
  pitch-preservingly, so it structurally cannot follow a resample — a real,
  named limitation, not noise.
- **fuse (81 % detect overall, 92 % on clean) beats either probe** — fp's
  precision fills in where chroma commits loosely, chroma's coverage fills fp's
  abstentions. Cheap, and validates the harness's cross-probe composition.

### Standing vs SOTA — honest

- **On offset/warp precision among committed spans we are at or *below* André's
  clean-condition median** (~0.02 s vs their ~1 s), competitive on the exact
  sub-problem — rough alignment / offset — that André explicitly *delegates to
  fingerprinting* rather than solving with NMF. So on "where does each known track
  sit, to the frame," a plain constellation matcher is already excellent on clean
  and effect-laden (non-resampled) content.
- **We are NOT SOTA on the task the paper is actually about.** André produces a
  *dense* warp function *f* + a gain curve for *every* track under *every*
  transform, with no abstention, and is robust to synthetic pitch-resampling; the
  DTW baseline likewise emits full warps. Our probes (a) emit a single affine
  offset, not a dense warp, (b) *abstain* rather than commit under resample/weak
  content (267 fp misses), and (c) emit **no gain curve at all**. So on the
  headline transcription metric (dense warp MAE + gain MAE over all conditions) we
  do not compete — nor were these probes built to.
- **Net:** the benchmark confirms the fingerprint probe is a *correct, precise,
  effect-robust* localizer in-distribution that **degrades by abstaining rather
  than lying** out-of-distribution (the design intent), but it does not model
  synthetic pitch-resampling and produces no dense warp/gain — so it's a strong
  *rough-alignment front end*, not a transcription SOTA competitor. The clean
  lever if we ever wanted to close the gap: a pitch-search (or CQT-domain
  constellation) for the resample axis, and chroma recalibration for electronic
  content.

## Probes that could not run as-is

- **ChromaProbe** at its shipped `_MIN_PEAK=0.5` abstains on 100 % of UnmixDB
  spans (electronic chroma peaks sit below the BB-tuned floor). Ran only after
  lowering the floor to 0.2; even then its committed offsets are near-chance on
  `resample`. This is a genuine finding: the BB chroma calibration does not
  transfer to electronic content.
- Vocal/language probes (HuBERT) and the segment/path decoders were **out of
  scope** (task-restricted, and UnmixDB is instrumental electronic content with
  no vocal stems — HuBERT has nothing to key on).

## Sources

- André, Fourer, Schwarz (2024), *DJ Mix Transcription with Multi-Pass
  Non-Negative Matrix Factorization*, arXiv:2410.04198.
- Schwarz & Fourer (2018), *UnmixDB: A Dataset for DJ-Mix Information Retrieval*,
  ISMIR — hal-02010431; dataset Zenodo record 1422385.
- Schwarz & Fourer (2021), *Methods and Datasets for DJ-Mix Reverse Engineering*
  (the DTW baseline extended).

## Reduction table (André mode) — 2026-07-11

The subsumption exhibit (spec: `docs/superpowers/specs/2026-07-11-andre-absorption-benchmark-design.md`).
Run our aligner (`fused`) in **André mode** — closed-set, always-commit — on the
same warp×effect strata, next to the two baselines we reproduced inside this
repo: `nmf` (`nmf_baseline.py`, a v0 fixed-W KL-NMF reproduction of André 2024)
and `dtw` (`method_dtw`, a chroma-DTW reproduction of André's DTW baseline). All
three are interchangeable `Method` callables scored on André's warp-error units —
that uniformity *is* the subsumption argument in code. Artifact:
`out/reduction_table.txt`. Sample: **220/240 mixes loaded, 660 GT spans** (20
mixes failed to load), seed-0 stratified, `--feature chroma`.

**André mode (all methods commit; warp error = set_start MAE + tempo MAE):**

| method | set_start MAE | med | <2 s | tempo MAE | abstain |
|--------|--------------:|----:|-----:|----------:|--------:|
| nmf (André NMF, v0 repro) | 19.07 s | 12.35 | 7% | 0.087 | 0% |
| dtw (André DTW baseline) | 6.66 s | 2.80 | 46% | 0.384 | 0% |
| **fused (ours)** | **5.79 s** | **2.39** | 43% | **0.065** | 0% |

- **`fused` wins warp error in their own regime.** Best set_start MAE (5.79 vs
  6.66 vs 19.07) and a tempo MAE (0.065) that *dominates* — 6× tighter than DTW
  (0.384) and below NMF (0.087). DTW edges `fused` on `<2 s` placement (46 vs 43%)
  but its tempo is an order of magnitude looser: DTW recovers *where* well and
  *how fast* poorly.
- **Our v0 NMF underperforms** (19 s placement MAE). This is the fixed-W v0, not
  André's published multi-pass NMF — do **not** read this as "we beat the paper's
  NMF." Against André's *reported* ~1 s clean-condition warp median, our `fused`
  clean-strata set_start medians (~2.3 s; 2.1–2.5 s across strata) are the
  comparable quantity; competitive,
  same order, on the sub-task the paper delegates to fingerprinting.

**Open mode (remove the always-commit restriction — `fused --min-votes 20`):**

| mode | set_start MAE | med | <2 s | tempo MAE | abstain |
|------|--------------:|----:|-----:|----------:|--------:|
| André mode (commit all) | 5.79 s | 2.39 | 43% | 0.065 | 0% |
| open mode (abstain on weak fp) | **2.35 s** | **1.67** | **54%** | **0.039** | **22.9%** |

- **Abstention is a capability, not a penalty.** Declining 22.9% of spans halves
  committed placement MAE (5.79 → 2.35 s) and tightens tempo (0.065 → 0.039). The
  closed NMF *cannot express this* — it commits always.
- **The abstention localizes to the exact structural wall we predicted.** By
  stratum, `fused` open-mode abstains **47–61% on `resample`** (pitch+tempo shift,
  which moves spectral peaks off our pitch-preserving constellation) but **0% on
  `none`** and 1.8–4.4% on three of four `stretch` strata — with one outlier,
  **`stretch/distortion` at 37%**, where distortion appears to confound the
  constellation the way pitch-shift does (an open question worth naming, not
  smoothing over). Where it *does* commit on resample, placement is now tight
  (MAE ~0.9–1.7 s, med <1 s). The model knows where it is blind and declines
  there — the "abstain-not-lie" posture, quantified. Closing the resample
  abstention is exactly the Phase 2 pitch-search arm.

**Honest caveats (must survive into the paper):**

- **Identity was not measured in this run** — no distractors were passed, so
  `identity_acc = 1.0` (a vacuous default) and the `--identity` block's `0.0%`
  (empty distractor pool) are both non-measurements. Open-set identification needs
  a re-run with `--n-distractors > 0`.
- **`nmf`/`dtw` are our reproductions**, not the authors' code; treat them as
  in-repo baselines, and anchor the SOTA claim to André's *published* figures
  separately.
- **`fused` set_start MAE is a single-anchor affine warp error**, comparable to
  but not identical to André's dense-warp MAE; UnmixDB's GT warp is affine, so the
  gap is small, but the paper must state it.
- Gain is not scored here (NMF-only, deferred) — named limitation, on-thesis.
