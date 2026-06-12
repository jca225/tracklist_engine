# Mix structure analysis — findings

## BB12 (`1fsnxchk`) — information-dynamics probe (2026-06-08)

**Setup:** 2054 bar-sync MERT-330M vectors (layer 6) on local `mix.m4a`; VQ
k=24 tokens; adaptive first-order Markov chain; scored against
`labeling/fixtures/bb12_ground_truth.yaml` (154 unique section-start boundaries,
±2 bars).

**Reproduce:**
```bash
venvs/audio/bin/python -m eda.alignment.mix_structure_probe \
  --artifact data/analysis/1fsnxchk_mix_mert.npz \
  --gt labeling/fixtures/bb12_ground_truth.yaml \
  --out data/analysis/1fsnxchk_structure_probe.json --persist
```

### Boundary detection (model-information-rate peaks)

| Metric | Value |
|--------|-------|
| Precision | 0.38 |
| Recall | 0.53 |
| F1 | **0.45** |
| Predicted peaks | 214 |
| GT boundaries | 154 |

The cheap Markov-over-VQ probe finds **about half** of human-labeled section
starts without seeing GT. Useful as a **boundary proposer**, not as the aligner.

### Is the mix "information-optimal"?

Using Abdallah & Plumbley (2008) readouts:

**Predictive information rate (PIR — inverted-U "interestingness"):**
~**78%** of bars sit in the mid-|PIR| band (between 0.25× and 2.5× the median).
By the paper's Wundt curve, the mix spends most of its runtime at *intermediate*
predictive information — not in low-information grooves nor only at shock spikes.
That is consistent with a deliberately varied mashup set, though we have no
corpus baseline yet to call it "optimal" vs other DJ sets.

**Bayesian surprise (model-information-rate) at GT boundaries:**
- GT section starts sit at the **56th** percentile of MIR vs **41st** for
  interior bars (+15 percentile-point lift).
- **12.3%** of GT boundaries fall in the top decile of MIR vs **~10%** random
  baseline — modest enrichment (~1.2×), not a smoking gun.

**PIR at GT boundaries:** no lift (49th percentile) — transitions are better
tagged by *surprise / MIR* than by PIR rate alone for this tokenization.

### Notable surprises (probe-only, not in GT)

Highest MIR clusters **late in the set** (~49–58 min): bars 1652–1941. Worth
listening at **~48:38, 49:58, 51:28** — may be outro chaos, double-drops, or
boundaries the hand pass did not mark as new section starts.

Early mix: bar 1 dominates (cold-start of empty Markov state) — ignore for
interpretation.

### Takeaways for aligner design

1. **Segment-then-label is validated** — bar-wise MIR carries section-start signal;
   dense per-bar song classification is unnecessary at first pass.
2. **Tokenization matters** — VQ k=24 + 75th-percentile peaks beat defaults; chroma
   side-stream still worth adding (plan Phase 1b).
3. **Not "solved"** — F1 0.45 and 1.2× top-decile enrichment mean the probe
   characterizes structure but does not replace manual GT.
4. **One bad MERT bar** (index 36, inf overflow) — now sanitized; root cause in
   float16 accumulate should be monitored on re-embed.

### Artifacts

| File | Description |
|------|-------------|
| `data/analysis/1fsnxchk_measure_times.json` | beat_this downbeats |
| `data/analysis/1fsnxchk_mix_mert.npz` | per-bar layer-6 MERT |
| `data/analysis/1fsnxchk_structure_probe.json` | v1 probe output |
| `data/analysis/1fsnxchk_structure_probe_v2.json` | **v2 dual-stream + local peaks** |
| `data/analysis/2nvzlh2k_*` | BB11 baseline (no GT) |
| `data/analysis/aux.db` | `analysis_results` rows |

---

## v2 — dual-stream (MERT-VQ + chroma) + local MIR peaks (2026-06-08)

**Motivation:** test whether chroma side-stream and locally-normalized surprise
peaks improve boundary detection and whether BB12 is uniquely information-dense
vs BB11.

### BB12 boundary F1 (±2 bars, 154 GT starts)

| Stream | Peak mode | F1 | Notes |
|--------|-----------|-----|-------|
| MERT-VQ | global | **0.446** | still best |
| MERT-VQ | local (32-bar z) | 0.391 | local norm did not help |
| Chroma-VQ | global | 0.390 | worse; GT lift **−6.6** (anti-aligned) |
| Chroma-VQ | local | 0.361 | |
| Combined | local union | 0.407 | more peaks, lower precision |

**Takeaway:** pitch/chroma tokens do not track mashup *section starts* — MERT
(shift-agnostic) is the right stream. Local MIR did not beat global thresholding
on BB12; late-set clustering was not the main bottleneck.

### Corpus baseline — mid-|PIR| band fraction (“interestingness”)

| Set | Bars | MERT mid-PIR | Chroma mid-PIR |
|-----|------|--------------|----------------|
| BB12 `1fsnxchk` | 2054 | **0.740** | 0.674 |
| BB11 `2nvzlh2k` | 1919 | 0.709 | 0.715 |

BB12 is slightly higher on MERT-PIR but **not dramatically** more
“information-optimal” than BB11 — both sit in the ~70–74% mid-band regime.
Cannot claim BB12 is special without more sets and event-level GT.

**Reproduce v2:**
```bash
venvs/audio/bin/python -m eda.alignment.mix_structure_probe \
  --artifact data/analysis/1fsnxchk_mix_mert.npz \
  --audio ~/aligning/1fsnxchk__Two\ Friends\ -\ Big\ Bootie\ Mix\ Volume\ 12/mix.m4a \
  --gt labeling/fixtures/bb12_ground_truth.yaml \
  --out data/analysis/1fsnxchk_structure_probe_v2.json
```

---

## v3 — model ladder M0/M1/M2 + chance baseline (2026-06-11)

Full code + caveats: [`info_dynamics/README.md`](info_dynamics/README.md).
Built a memoryless null (**M0**), kept the adaptive Markov as **M1**, and added a
strictly-prequential causal sequence model (**M2**: Transformer / GRU, discrete
softmax over the same K=24 codebook, expanding-window predict-then-update). Scored
in **seconds** (±3 s / ±10 s) with a **random-peak chance baseline** and a
**temporal-shuffle control** — neither of which the v1/v2 probe had.

**Reproduce:**
```bash
venvs/audio/bin/python -m eda.alignment.info_dynamics.run \
  --artifact data/analysis/1fsnxchk_mix_mert.npz \
  --gt labeling/fixtures/bb12_ground_truth.yaml \
  --out data/analysis/info_dynamics
```

### Prediction — memory clearly helps (prequential NLL, nats; lower better)

| M0 | M1 | M2-gru | M2-attn | uniform (log 24) |
|----|----|--------|---------|------------------|
| 2.92 | **2.26** | 2.29 | 2.66 | 3.18 |

An evolving prior predicts the mix far better than memoryless; most gain is at
**order 1** (GRU ≈ Markov; the Transformer is data-starved on one ~2 k-token mix).

### Boundary localization — mostly chance (lift = F1 − random-peak F1, ±3 s)

| Signal | F1@3s | chance | lift | shuffled F1@3s |
|--------|-------|--------|------|----------------|
| **M0 persist `1−cos`** | **0.284** | 0.160 | **+0.125** | 0.142 ✅ collapses |
| M2-gru surprisal | 0.241 | 0.145 | +0.096 | 0.225 ⚠️ barely moves |
| M1 surprisal | 0.207 | 0.166 | +0.033 | 0.235 ❌ |
| M1 MIR | 0.199 | 0.166 | +0.034 | 0.195 ❌ |
| M1 PIR | 0.199 | — | −0.007 | 0.234 ❌ |

Only **local acoustic novelty** (cosine distance between adjacent bars) beats its
shuffle. The information-dynamics *surprise* signals (surprisal / MIR / PIR) are at
chance once boundary density is controlled.

### ⚠️ This revises the v1/v2 headline

The v1 **F1≈0.45** was scored at ±2 bars with **no chance baseline**. At BB12's
GT density (~1 boundary / 20 s) a ±10 s window tiles the timeline — *random* peaks
score ~0.40 and every model's ±10 s lift is ≈0. So v1/v2's "finds ~half the
section starts" reflects **density, not localization**. MERT + information
dynamics gives a good *predictive* model but **not** a transition detector here.

---

## v4 — drop the VQ + flatten to one stem (2026-06-11)

Reproduce: `info_dynamics.run_grid` (3×2 grid) + `info_dynamics.run_robustness`
(multi-seed null). Hypothesis (user): the paper is right; v3 failed because it
ran info-dynamics on a *polyphonic mashup* through a *24-symbol quantizer*. Two
fixes: (a) **continuous** models (PCA-whitened MERT, Gaussian-NLL surprise, no
codebook); (b) run per **Demucs stem of the mix** (`mix_instrumental.flac`,
`mix_vocals.flac`, same bar grid) so a single coherent stream is modelled.

**±3 s lift over random-peak chance, real vs multi-seed shuffle null:**

| Source | codebook (M1) real / shuf-max | continuous (M2c) real / shuf-max |
|--------|------------------------------|----------------------------------|
| full mix | +0.034 / 0.090 ❌ | +0.098 / 0.068 ~ (suggestive) |
| acappella | +0.062 / 0.088 ❌ | +0.093 / 0.117 ❌ (vocals too sparse) |
| **instrumental** | +0.083 / 0.081 (marginal) | **+0.104 / 0.051 ✅ (~4.5σ)** |

**Result — the user's hypothesis holds, narrowly:**
1. **Continuous ≫ codebook** in every source. VQ (K=24) was the main thing
   crippling v3; dropping it is necessary.
2. **Instrumental stem + continuous is the one robust win** — real lift +0.104
   clears the shuffle distribution (mean 0.037, max 0.051 over 10/4 seeds) by a
   wide margin. The instrumental is also the *most predictable* stream (prequential
   NLL 2.06 vs 2.26 full) — it is the structural anchor, and a memory model's
   forecast-shift (`pred_change`, ≈ predictive-information rate) localizes its
   section starts above chance.
3. **Full mashup & acappella-alone do not robustly localize** — superposition
   masks the seams (full), and intermittent vocals give a high-variance null
   (acappella).

So information dynamics on MERT **does** detect transitions once it sees a single
continuous stream (the instrumental), vindicating both the paper and the
"flatten to a stem" intuition. Caveat: effect is **robust but modest** (absolute
F1@3s ≈ 0.23), n = 1 mix, one bar grid — needs replication across sets before it
is a detector rather than a characterisation.

## v5 — proper significance test + the v4 exclusivity claim fails to reproduce (2026-06-12)

Reproduce: `info_dynamics.run_robustness` (rewritten). Two upgrades over v4's
"~4.5σ", which was an *inferred* descriptor `(real − shuf_mean)/σ` over only 4
continuous shuffle seeds — too few to estimate σ, let alone claim a p-value
(4 permutations floor the achievable p at ~0.2). v5 fixes the statistics **and**
re-runs on the current MERT artifacts (regenerated 2026-06-08…11).

**What changed in the method:**
1. **Exact permutation p-value (primary null).** Take the real model's best-lift
   signal, hold its peak *pattern* fixed (count + spacing), and circularly shift
   its phase within the labeled window 1000× → null F1 distribution.
   `p = (1+#{null≥real})/(N+1)`, one-sided. Tests "do these peaks land on GT
   boundaries better than an arbitrary phase?" — preserving the signal's own peak
   structure, so it is stricter than the uniform-random `random_chance_f1` floor.
2. **Bootstrap 95% CI on the lift** (1000 draws, resampling per-event hit/miss
   indicators — *not* GT timestamps, which would duplicate boundaries the
   one-to-one matcher can't hit and bias the CI down).
3. **Benjamini–Hochberg FDR** across all 6 grid cells; **instrumental × continuous
   pre-registered** as the primary hypothesis.

**±3 s localization, current artifacts (best signal per cell):**

| Source | repr. | F1 | lift [95% CI] | p (perm) | q (FDR) |
|--------|-------|----|----|----|----|
| full | codebook | 0.291 | +0.114 [+0.053, +0.169] | .0020 | .002 |
| full | continuous | 0.342 | **+0.201 [+0.132, +0.262]** | .0010 | .002 |
| acappella | codebook | 0.280 | +0.094 [+0.038, +0.149] | .0030 | .003 |
| acappella | continuous | 0.321 | +0.161 [+0.098, +0.226] | .0010 | .002 |
| instrumental | codebook | 0.336 | +0.157 [+0.094, +0.215] | .0020 | .002 |
| **instrumental ⭐** | continuous | 0.287 | +0.135 [+0.068, +0.193] | .0010 | .002 |

**Primary hypothesis survives:** instrumental × continuous lift +0.135
(CI [+0.068, +0.193] excludes 0), permutation p = .001, FDR q = .002 →
significant. The instrumental's surprise peaks localize its section starts.

**But the v4 *exclusivity* claim does NOT reproduce.** v4 said instrumental +
continuous was "the **one** robust win" (full = suggestive, acappella = fail). On
the current artifacts **all six cells are significant** under the permutation null,
and full-mix continuous is in fact the *strongest* (+0.201, z = 6.1) — not the
instrumental. The secondary model-level input-shuffle null (retrain on scrambled
frames, 10 disc / 4 cont seeds) agrees: every continuous cell now clears its
shuffle-max (full +0.201>.085, acap +0.161>.090, instr +0.135>.059), whereas in
v4 only instrumental did. Two things drove the shift:
- **Artifacts were regenerated.** v4's instrumental-cont lift was +0.104; it is now
  +0.135, full-cont jumped +0.098→+0.201. The MERT/Demucs inputs changed under the
  analysis — the old numbers are stale, and the conclusion was sitting on ±0.05
  margins that didn't survive re-extraction. **Flagged for follow-up:** diff the
  artifact provenance before trusting either set of absolute lifts.
- **The permutation null is more lenient than the input-shuffle null** by design —
  it asks "are these peaks well-phased?" not "did the model learn boundary
  structure from real (vs scrambled) data?" Passing it for full-mix is weaker
  evidence than the v4 input-shuffle test that originally singled out instrumental.

**Honest takeaway:** the *significance machinery* is now sound (real p-values,
valid CIs, FDR) and the headline that info-dynamics localizes transitions on a
single continuous stream holds with q < .01. The sharper claim — that the
**instrumental stem specifically** is privileged — is **not** supported on current
data; full-mix continuous does at least as well. The stem-wise design rationale
([[project_stemwise_alignment]]) now rests on the *prequential-predictability*
argument (instrumental is the most forecastable stream) rather than on a
localization advantage, which has evaporated.

**Scope ceiling (unchanged and decisive): n = 1 mix.** Every test above is
*within-mix* — it shows BB12's surprise is non-randomly aligned to BB12's own
section starts. The unit of replication for a population claim is the **set**, not
the bar-frame (frames are not exchangeable across mixes), so no permutation count
upgrades this to "info-dynamics is a transition detector." Replicating across
≥3–5 hand-labeled sets — treating *set* as the unit — is the only thing that does.

## v6 — first cross-set replication: BB11 vs its tracklist (2026-06-12)

Reproduce: `info_dynamics.run_bb11`. The first step toward the v5 ceiling — a
*second* set, **BB11** (`2nvzlh2k`), scored against an **independent** boundary
source: the 142 scraped 1001tracklists **cue times** (BB11 has no hand-labelled
Ableton GT). Aligning model surprise to a human-but-independent boundary signal is
*stronger* evidence than BB12-vs-its-own-hand-labels — the boundaries weren't made
by us. **Full 3×2 grid** now run: the BB11 mix Demucs stems already existed on
pi-storage (`/mnt/storage/stems/set/6/{instrumental,vocals}.flac`), MERT-embedded
onto the full-mix bar grid (1919 bars) via `prepare_mix_artifact`.

**±3 s localization vs tracklist cues (1000-perm p, FDR q over all 12 cells; ±3 s
is the discriminating tolerance — wide ±10 s saturates on the dense ~1/24 s GT):**

| source | repr. | F1 | lift [95% CI] | p | q | verdict |
|--------|-------|----|----|----|----|----|
| full | codebook | 0.319 | +0.156 [+0.093, +0.216] | .001 | .010 | ✅ |
| full | continuous | 0.297 | +0.164 [+0.092, +0.230] | .004 | .010 | ✅ |
| acappella | codebook | 0.316 | +0.145 [+0.084, +0.200] | .004 | .010 | ✅ |
| acappella | continuous | 0.225 | +0.081 [+0.017, +0.141] | .038 | .057 | ~ marginal |
| **instrumental** | **codebook** | 0.385 | **+0.214 [+0.150, +0.271]** | .003 | .010 | ✅ strongest |
| instrumental | continuous | 0.253 | +0.113 [+0.041, +0.172] | .006 | .012 | ✅ |

**Result — the effect replicates, broadly.** 5 of 6 cells clear FDR at ±3 s (only
acappella-continuous is marginal, q = .057). Boundaries are localizable above
chance from **every** stream — full, acappella, and instrumental — on an
independent boundary source. The effect is real and not stem-exclusive.

**But none of the *finer* claims survive the cross-set test:**
1. **No privileged stem.** Best cell is **full-continuous on BB12** (+0.201) vs
   **instrumental-codebook on BB11** (+0.214). The winner flips by set; there is no
   stable "best stem." The v4 "instrumental only" claim is doubly dead — and note
   the instrumental is *strongest* on BB11, the opposite of weak.
2. **Codebook vs continuous flips too.** BB12 had continuous ≫ codebook ("drop the
   VQ"); BB11 has codebook competitive-to-winning (instrumental-codebook is the top
   cell, continuous loses to codebook in 2 of 3 sources). The v4/v5 "continuous is
   necessary" conclusion was *also* a BB12 single-set artifact.

**Honest synthesis (n = 2).** What replicates: **surprise localizes section
boundaries above chance, robustly across stems and representations.** What does
*not* replicate: any ranking *among* stems or representations — those were
set-specific. The aligner should treat all stem channels as informative boundary
signals, not bet on one. Still n = 2; the v5 ceiling (set as unit, ≥3–5 sets)
stands. Caveat: tracklist cues carry their own small error, partially absorbed by
the ±3 s window.
