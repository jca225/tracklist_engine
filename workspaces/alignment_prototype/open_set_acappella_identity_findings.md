# Open-set acappella identity via labeling functions (BB11)

**2026-07-24. Set: BB11 (`2nvzlh2k`). Exploratory; code in scratchpad
(`bb11_acap_id/`), promotable to `evals/` on request.**

## Question

Can we identify *which* acappella plays in each mix span **without the tracklist
prior** — no slot order, no scraped cue times — purely from audio, matching the
mix's vocal stem against the pool of the set's acappella references? Framed as
**programmatic weak supervision**: several identity *labeling functions* (LFs)
with different invariances, then fused.

## Setup

- 91 hand-labeled acappella GT spans → **89 distinct reference files** — the
  *exact* files the annotator placed in Ableton, resolved from the `.als`
  (74 candidate downloads + 17 Baby-rule derived `vocals.flac` stems), not fresh
  downloads.
- Query = mix **vocal stem** (RoFormer) windowed to each GT span; refs = the
  acappella files. Stem-to-stem, both vocal-domain.
- Metric: **89-way top-1** — does the LF rank the true reference first? No
  order/cue prior; span identity only (not placement).

## Labeling functions & invariances

| LF | invariance | raw top-1 |
|----|-----------|-----------|
| landmark fingerprint (Shazam-style, frequency-absolute) | none | 55% |
| chroma cross-correlation | warp-sensitive, key-sensitive | 15% |
| chroma subsequence-DTW | warp-tolerant, key-sensitive | 46% |
| MERT chamfer (vocal-domain, windowed; `mean_q max_ref cos`) | pitch + warp | **86%** |

## Key results

- **Fingerprint dies on pitch.** Raw fp = 55% but **0/32 on ±1-semitone
  transposed spans** — a semitone (~6% freq) moves every landmark peak. On
  *un*-transposed spans fp is 85%.
- **Pitch ⟂ tempo (do not conflate).** Ableton warp is pitch-*preserved*
  time-stretch (`tempo_ratio`, up to 1.5×) **plus a separate ±1-semitone
  transpose** (`pitch_shift_semi` / clip `PitchCoarse`+`PitchFine`). Correcting
  the labeled *semitone* (true pitch-shift) → fp 78%; also undoing the labeled
  *tempo* → 92%. Correcting `tempo_ratio` *as if* it were pitch (an early error)
  tests the wrong knob.
- **MERT covers the transpose blind spot** the whole fingerprint/chroma family
  shares (all three score 0% on pitched spans; MERT ~78%).
- **MERT layer matters — use L3, not L6.** Sweep over all 25 hidden states:
  **L3 = 91%** (best), L4/L7 = 90%, **L6 (repo default) = 88%**, monotonic
  falloff to 80% by L23. Lower/earlier (acoustic-timbre) layers win for vocal
  identity. Validates the adapter keeping all 25 layers (SUPERB/s3prl per-task
  layer pick). *NB: L6 may still be right for other tasks — this is an
  identity-axis, vocal-domain result.*
- **Long-span contamination.** Over long windows the mix vocal stem accumulates
  *other* layered vocals, dragging the chamfer mean. **Conditional top-k** (mean
  of top-25% query bars, applied only to clips >60 s) lifts long-clip identity
  73→82% while leaving short clips on `mean` (top-k hurts them).
- **Best blind single LF: MERT-L3 + conditional top-k = 92%** — no tracklist, no
  pitch label. **Best fusion: borda(MERT-L3, pitch-fp) = 96% = the oracle-of-LFs
  ceiling.** (fp uses the *labeled* semitone → oracle; a deployable fp needs
  blind pitch/key estimation.)
- **Weak LFs hurt.** Adding chroma/dtw to an equal-weight combiner *drops*
  accuracy (56–75%) — they confidently vote wrong and average away MERT's picks.
  Prune to the two strong LFs.
- **The combiner is the lever, not more sensors.** Residual fusion failures are
  cases where one LF has the true ref at rank 0 but equal-weight borda mis-fuses.
  Hand-tuned gating does **not** beat borda at n=91 (overfits) → needs a
  **learned combiner trained cross-set (co-training)**; the LOSO precedent says
  identity signals transfer ~100% across sets ([cotrain_loso_findings.md](cotrain_loso_findings.md)).

## Residual at 96% (4 spans)

- `022w1` Out of Love (137 s, derived `vocals.flac`) — reference quality (ingest).
- `002w2` Sk8er Boi (16 s) — annotator-flagged **bad reference** (cover / different singer).
- `039w1` The Scientist (21 s) & `018w2` Demons (16 s) — MERT-L3-*alone* gets both;
  pure **combiner** losses the equal-weight fusion re-broke.

So: ~2/4 are reference-quality (ingest); ~2/4 are combiner losses (co-train).

## Settled decisions

1. Acappella vocal identity: **MERT L3** (not L6).
2. Stem-to-stem + MERT chamfer + conditional top-k = strong **blind** identity
   (92%, no tracklist prior).
3. Fusion ceiling with these LFs = **96%**; past it = learned combiner
   (co-train) + reference quality + longer-context sensor — **not** a new probe.

## Cross-set LOSO (BB11 ↔ BB12) — measured 2026-07-24

Ran BB12's 101 acappella spans (94-ref pool) through the same 5-LF pipeline and
trained a per-candidate label-model (logistic regression, class-balanced, ~18
set-agnostic features: per-LF within-span z-score / normalized-rank / is-top1 +
cross-LF agreement) on one set, tested on the other.

| train→test | learned combiner | borda(mert,fp_pit) | best single LF | oracle-of-5 ceiling |
|---|---|---|---|---|
| BB11→BB12 | **85%** | 83% | 77% (fp_pit) | **88%** |
| BB12→BB11 | **95%** | 96% | 92% (MERT-L3) | **99%** |

Three findings:

1. **The combiner transfers.** Trained on the *other* set it beats every single
   LF on both held-out sets and does not collapse cross-set → identity-fusion is
   set-agnostic (confirms the co-train hypothesis).
2. **But it ≈ borda, not ≫ borda** (85 vs 83; 95 vs 96). Naive rank-fusion of the
   two strong LFs is already near-optimal, so *learning* the combiner buys
   robustness, not a big accuracy jump. This **revises the earlier "co-training
   closes 92→96"**: with L3-MERT, borda already hits the ceiling on BB11.
3. **The ceiling is the sensors/references, not the combiner.** BB12's oracle-of-5
   ceiling is only **88%** — 12% of spans *no LF* reaches → LF/reference quality is
   the bottleneck on harder sets. (L3 also raised BB11's ceiling 96%→**99%**.)

**Surprise — MERT generalizes worse than BB11 implied:** MERT-L3 drops **92% (BB11)
→ 68% (BB12)**, and on BB12 the pitch-fingerprint (76%) *beats* MERT. LF dominance
**flips across sets**; the combiner absorbs the flip by reweighting. So "MERT is
*the* identity sensor" (decision #20) is set-dependent — the combiner's robustness
is what matters. Why BB12 is harder (contamination / separation quality / ref pool)
is the open question.

## Instrumental chain (contrast) — measured 2026-07-24

Ran the identical pipeline for the **non-acappella** spans (regular full drops +
`claimed_stem:instrumental` overlays; BB11 56 spans/51 refs, BB12 63/58), query =
mix **instrumental** stem. NB "regular" ≠ "vocal song" — BB's full drops are
instrumental EDM; true vocal-regular spans are <5%.

| | acappella | instrumental |
|---|---|---|
| best MERT layer | **L3** (low/acoustic) | **L22** (high/abstract) |
| chroma top-1 | 15–21% | **79–88%** |
| dtw | 43–46% | **75–89%** |
| fp_pit | 78% | 84–86% |
| MERT | 86–92% BB11 / **68% BB12** | **89% / 89%** |
| # strong LFs | 1 (MERT only) | ~5 (all) |
| oracle-of-5 ceiling | 99% / **88%** | **95% / 94%** |
| MERT cross-set | 92→68 (unstable) | 89→89 (stable) |
| LOSO combiner | transfers ≈ borda | transfers ≈ borda (89% both dirs) |

Findings:

1. **LF dominance flips by stem — the "instrumental → chroma+fingerprint" axis rule
   holds hard.** Chroma is useless on acappellas (15%) but top-tier on instrumentals
   (88%); dtw 46→89%. Instrumentals are harmonic/percussive → no single blind spot;
   fp/chroma/dtw/MERT are *all* strong (unlike acappella's lone MERT).
2. **Best MERT layer flips by stem: L3 (vocal) vs L22 (instrumental).** Vocal identity
   = low-layer acoustic-timbre; instrumental/musical identity = high-layer
   abstract-harmonic. Per-stem layer selection — validates keeping all 25 layers.
3. **Instrumental identity is more robust:** oracle ceiling 94–95% on *both* sets (vs
   acappella's BB12 collapse to 88%) — redundant complementary LFs cover each other.
4. **MERT's generalization gap is vocal-specific:** 92→68 cross-set on acappellas but
   **89→89 on instrumentals** → the earlier "MERT doesn't transfer" (§ Cross-set
   LOSO) traces to vocal contamination/separation, not MERT itself.

Caveat: instrumental pools are smaller (~55-way vs ~90), so some of the higher
accuracy is easier classification — but the LF-dominance flip and the layer flip are
pool-size-independent.

## Caveats

- Two sets only; the label-model is small-n but the *cross-set* eval is the honest
  transfer test (train and test sets are disjoint).
- fp pitch-normalization uses the labeled semitone (oracle); deployable fp needs
  blind pitch/key.
- Sensor phase is closed (see [CLAUDE.md](CLAUDE.md)); this is an **eval** of the
  identity axis + a layer/aggregation tuning of an existing channel (MERT), not a
  new channel.
