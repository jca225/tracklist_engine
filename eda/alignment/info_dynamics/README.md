# info_dynamics — does MERT + information dynamics predict DJ transitions?

A test of Abdallah & Plumbley's *Information Dynamics* (Connection Science 2009)
on a real DJ mix, with **MERT embeddings** as the perceptual representation. The
hypothesis: a sequential predictive model over MERT, evaluated **prequentially**,
should emit surprise / information signals whose peaks land on the DJ's song
transitions.

This is the rigorous follow-up to the [`../findings.md`](../findings.md) "mix
structure probe" — it adds the memoryless null, a strict prequential sequence
model, second-based scoring with a **chance baseline + shuffle control**, and
corrects an over-optimistic earlier reading (see *Findings* below).

## Run

```bash
venvs/audio/bin/python -m eda.alignment.info_dynamics.run \
  --artifact data/analysis/1fsnxchk_mix_mert.npz \
  --gt labeling/fixtures/bb12_ground_truth.yaml \
  --out data/analysis/info_dynamics
```

Outputs (`data/analysis/info_dynamics/`): `config.json`, `metrics.json`,
`metrics.csv`, `summary.md`, `plots/<model>_signals.svg`. ~80 s on a Mac CPU,
fully seeded.

## The model ladder

| Rung | Module | What it is | Memory |
|------|--------|-----------|--------|
| **M0** | `baselines.run_m0` | online marginal token model + embedding persistence `1−cos(xₜ,xₜ₋₁)` | none (null) |
| **M1** | `baselines.run_m1` | adaptive 1st-order Markov chain w/ Dirichlet forgetting (reuses `../adaptive_markov.py`) | order-1 |
| **M2** | `seqmodel.run_m2` | small causal Transformer **or** GRU, discrete softmax head over the same K codebook | long causal context |

All three share **one VQ codebook** (k-means, K=24, fit once on the whole mix) so
they speak the same alphabet. The codebook is a fixed unsupervised quantizer — it
never sees boundaries; only the *sequence model* is held to the prequential bar.

## Prequential protocol (M2)

Expanding-window, predict-then-update:

```
warm-up train on tokens[0:W]            (W=128 bars ≈ 3.8 min, left unscored)
for each block [s, e):                  (block = 32 bars)
    record predictions for tokens[s:e]  using a model trained only on [0:s]
    warm-start train on tokens[0:e]
```

A next-token forecast for frame *t* reads the causal position that attends only
to tokens `< t`, and the model's parameters were fit only on frames `< s ≤ t` —
so nothing at or after *t* ever touches its own prediction. No train/test leakage
on the single ~1 h sequence. M0/M1 are prequential by construction (online counts
/ online Dirichlet updates).

## Information signals (per frame, per model)

- `surprisal` = −log p(xₜ | past) — the prequential NLL.
- `entropy` = predictive entropy H(X | past) before the observation.
- M1 only: `mir` (Bayesian surprise, Dirichlet KL posterior‖prior), `pred_info`
  (the paper's *exact* predictive information I(x|z)), `pir_proxy` (Δ-entropy).
- M2 only: `fwd_kl` — KL between the one-step forecast after vs. before observing
  xₜ. A **proxy** for predictive information, not the exact quantity.

## Evaluation

- **Boundary detection** — peak-pick each signal (global percentile + min 6 s
  spacing), score against GT `set_start_s` times at **±3 s and ±10 s**, greedy
  one-to-one matching. All models scored on the same window (past M2's warm-up).
- **Chance baseline** — F1 of *N* random uniform peaks (40 trials). `lift = F1 −
  chance`. **Report lift, not raw F1.**
- **Shuffle control** — temporally permute the frame sequence, recompute, rescore.
  A genuine localizer's lift collapses; an artifact's does not.
- **Prediction** — mean prequential NLL over labeled frames (does memory help?).
- **Context ablation** — attention restricted to 8 / 32 / full previous bars.

## Findings (BB12, `1fsnxchk`, K=24)

**1. Memory clearly helps *prediction*.** Prequential NLL: M0 **2.92** →
M1 **2.26** → GRU **2.29** (uniform-token baseline = log 24 = 3.18). An evolving
prior predicts the mix far better than the memoryless null — the predictive core
of information dynamics holds for MERT tokens. Most of the gain is captured at
**order 1**; the GRU matches the Markov chain and the small Transformer (2.66)
underperforms — one ~2 k-token mix is too little data to train attention well.

**2. The surprise signals do *not* robustly localize transitions.** At the
discriminating ±3 s tolerance, only **M0 embedding-persistence** beats its
temporal-shuffle null cleanly (F1 0.284 → 0.142 shuffled; lift +0.125 → −0.018).
Every Markov information-dynamics signal (surprisal / MIR / PIR) has a *shuffled*
F1 ≥ its real F1 — i.e. at chance once boundary density is controlled. The GRU's
surprisal is the best model-based signal (lift +0.096) but only marginally beats
its own shuffle (+0.071), so it is weak evidence at best.

**3. ±10 s is saturated — this corrects the earlier `findings.md` F1≈0.45.** With
~1 boundary per 20 s, ±10 s windows tile the timeline: random peaks already score
~0.40 and every model's ±10 s lift is ≈0 or negative. The previously headlined
F1≈0.45 (±2 bars, no chance baseline) was an artifact of density, not evidence of
localization.

**Bottom line (v3, full mix + codebook).** Information dynamics yields a *good
predictive model* (memory helps next-frame prediction) but its *surprise signals
are not a transition detector* on the full mashup through a 24-token codebook — a
plain local-novelty detector beats every model surprisal. See **v4** for the
resolution.

## v4 — continuous + per-stem (`run_grid`, `run_robustness`)

```bash
# 3×2 grid: {full, acappella, instrumental} × {codebook, continuous}
venvs/audio/bin/python -m eda.alignment.info_dynamics.run_grid
# multi-seed shuffle null (hardens the verdict)
venvs/audio/bin/python -m eda.alignment.info_dynamics.run_robustness
```

Two changes test whether v3 failed on *representation*, not theory: **continuous**
models ([continuous.py](continuous.py) — PCA-whitened MERT, Gaussian-NLL surprise,
no VQ) and running per **Demucs stem of the mix** (`mix_instrumental.flac` /
`mix_vocals.flac`, same bar grid via `prepare_mix_artifact`).

**Result:** continuous ≫ codebook everywhere, and **instrumental-stem + continuous
is the one cell whose ±3 s lift (+0.104) clears its multi-seed shuffle null (~4.5σ,
shuffle max 0.051)**. The full mashup and acappella-alone do not robustly localize.
So information dynamics *does* detect DJ transitions once it sees a single coherent
continuous stream (the instrumental anchor) — vindicating the paper and the
"flatten to a stem" intuition — but the effect is robust-yet-modest (F1@3s ≈ 0.23,
n = 1 mix). Full write-up: [`../findings.md`](../findings.md) §v4.

## Caveats

- **n = 1 mix.** BB12 is the only set with local ground truth. Single-shuffle
  null; no cross-mix replication. Treat all numbers as one data point.
- **Codebook & granularity.** K=24, layer-6, bar-synchronous (~0.56 Hz). The
  ~1.8 s bar grid caps ±3 s resolution; a finer fixed-rate grid (re-embed) might
  sharpen localization. VQ discards within-token acoustic change that
  persistence exploits directly.
- **Smoothing sensitivity.** Peaks depend on `smooth_window` / `percentile` /
  `min_distance_s` (all in `config.json`); lifts are small enough to move with
  them. The qualitative ranking (persistence > model surprise; ±10 s saturated)
  is stable, the exact F1s are not.
- **M2 is data-starved.** Attention NLL > GRU/Markov reflects training a
  Transformer on one short sequence, not that attention is wrong in principle.
