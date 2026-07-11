# P0: Information-Dynamics Mashup Surprise — Findings

*Run date: 2026-07-11. Script: `eda/information_dynamics/bb_mashup_surprise_p0.py`.*
*Metrics persisted to `data/analysis/aux.db :: analysis_results / bb_mashup_surprise_p0_v1`.*

---

## Headline

**WEAK GO — with important caveats.**

The chroma KL surprise signature separates real DJ mashups from key/BPM-compatible
but unchosen pairings at AUC 0.576 (best single feature: `mean_kl`) vs a key-only
baseline of 0.498. The signal is real but modest. It clears the pre-registered
threshold (> 0.5 and > baseline + 0.05), but only barely on the multi-feature
classifier (AUC 0.565).

---

## AUC Table

| Predictor | LOSO AUC |
|---|---|
| Key-only baseline (random on key-matched data) | 0.498 |
| `skew_kl` | 0.482 |
| `kl_drop_spike` | 0.478 |
| `invU_coeff` | 0.556 |
| `max_kl` | 0.568 |
| `var_kl` | 0.571 |
| **`mean_kl` (best single)** | **0.576** |
| Multi-feature LR (all 6) | 0.565 |

Evaluation: leave-one-set-out (BB11 held out / BB12 held out, AUC averaged).

---

## Exact Surprise Definition

For each (vocal, bed) pair:

1. Load chroma (librosa `chroma_cqt`, 12 bins, 0.25 s hop, 22 050 Hz mono)
   from both the acappella audio file (sliced at `ref_start_s`, up to 40 s)
   and the mix instrumental (sliced at `set_start_s` for the same duration).
2. Apply 3-frame centred sliding-mean smoothing per bin (regularises silence).
3. L1-normalise each frame to sum to 1; silent frames get uniform 1/12.
4. Per-frame KL divergence (vocal as distribution p, bed as q):

       surprise_t = KL(p_t || q_t) = Σ_k p_k log(p_k / q_k)

   with EPS=1e-7 clipping before normalisation to avoid log(0).
5. Features extracted from the (n_frames,) trajectory:
   - `mean_kl` — mean harmonic tension
   - `var_kl` — variance of tension trajectory (dynamic range)
   - `max_kl` — peak surprise
   - `skew_kl` — skewness (heavy tail?)
   - `invU_coeff` — quadratic coefficient from OLS fit (negative = inverted-U shaped)
   - `kl_drop_spike` — max KL in first 20% of frames minus mean of rest (entry spike)

**Positive pairs**: real acappella span from GT, sliced from the standalone
acappella audio file; bed = the mix_instrumental at the matching set time.

**Negative pairs**: the same vocal, bed from a DIFFERENT time window of the
same corpus (must pass |BPM ratio - 1| <= 6% and +-1 Camelot step compatibility
gate, or +-1 Camelot if BPM unavailable). Drawn at up to 3 negatives per positive.

Counts: n_pos=180 (of 182 attempted), n_neg=540, total=720, 8 pairs failed audio
load (too short after slicing).

---

## Feature Importances (single-feature LOSO AUC, ranked)

1. `mean_kl` — 0.576: the mean harmonic tension is the single best separator.
   Real mashups produce *higher* mean KL than random pairings — good DJ choices
   layer vocals that are *more* harmonically foreign to the bed, not less.
2. `var_kl` — 0.571: real pairings have more dynamic tension trajectories.
3. `max_kl` — 0.568: peak surprise is slightly elevated in real pairings.
4. `invU_coeff` — 0.556: weak evidence for the inverted-U shape, but noise-level.
5. `skew_kl` — 0.482: below baseline — distribution shape does not help.
6. `kl_drop_spike` — 0.478: entry spike not diagnostic.

The multi-feature LR (0.565) does not beat the best single feature (0.576),
suggesting the features are correlated and the classifier is overfitting the
small LOSO validation set.

---

## Verdict and Interpretation

**Pre-registered rule**: AUC clearly > 0.5 AND > key-baseline + 0.05 -> GO.

With multi-LR AUC = 0.565 > 0.5 and > 0.498 + 0.05 = 0.548: **GO by a narrow
margin.**

The sign of the effect is meaningful: real DJ choices pair vocals with beds that
are *more* harmonically tense (higher mean_kl), not less. This runs counter to
a naive "consonance = good" prior — Two Friends choose pairings that create
controlled harmonic friction, not smooth agreement. That is consistent with the
"unexpected combinations become memorable" hypothesis.

However, the margin (AUC 0.576 vs 0.5 baseline) is small enough that:
- A different artist's corpus could reverse the sign.
- The current chroma feature space is almost certainly not the ceiling: CQT
  chroma conflates octaves, is pitch-class-local, and ignores timbre/rhythm.
- The inverted-U trajectory shape (the theoretical heart of the Abdallah/Berlyne
  account) does NOT clearly appear in the data at this resolution.

---

## Caveats

1. **Small-n**: n=720 pairs, 180 positives from 2 sets only (BB11+BB12).
   95% CI half-width ~0.037 on any AUC estimate. The GO verdict is within
   ~1.5 CI half-widths of the NULL threshold. A third set would meaningfully
   sharpen the picture.
2. **Bed proxy is approximate**: the bed is the full-mix instrumental (Roformer
   separation from the mix), not a perfectly clean isolated instrumental. Separation
   artefacts inject noise into the bed chroma.
3. **Temporal alignment is naive**: we slice the vocal at `ref_start_s` with no
   tempo-ratio compensation. For spans where `tempo_ratio` != 1.0 the chroma
   frames drift out of sync.
4. **Key gate is lenient when BPM is unavailable**: ~60% of bed candidates had
   no BPM tag, so BPM compatibility was not enforced for those.
5. **Single artist**: both sets are Two Friends. The finding may not transfer to
   other DJ aesthetics.

---

## Recommendation

**Conditional GO**: build the chroma surprise ranker as a lightweight feature for
the decision model (mean_kl alone is interpretable and fast), NOT yet as a
standalone critic. Before investing in the full ranker, run two cheap experiments:

1. **Clean bed**: use individual track instrumental stems instead of mix_instr
   (already available for ~111 BB12 regular tracks).
2. **Tempo-ratio correction**: resample vocal by `tempo_ratio` before chroma
   extraction.

If the cleaner experiment lifts AUC >= 0.60, build the ranker. If it stays at
0.57, lean on the verb-log preference model instead and archive this lane.
