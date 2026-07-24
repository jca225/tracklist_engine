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

## Caveats

- n = 91, single set; hand-tuned combiners overfit — cross-set (co-train/LOSO)
  is the honest next step.
- fp pitch-normalization uses the labeled semitone (oracle); deployable fp needs
  blind pitch/key.
- Sensor phase is closed (see [CLAUDE.md](CLAUDE.md)); this is an **eval** of the
  identity axis + a layer/aggregation tuning of an existing channel (MERT), not a
  new channel.
