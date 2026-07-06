# Research brief: acappella loop-tracing (the trajectory wall)

**Status:** open research problem, 2026-07-06. The last big weak axis of the aligner.
Everything else (placement, identity, regular/instrumental) is at usable quality; this is not.

## The problem in one line

Given a span of a DJ mix where an acappella (isolated vocal) is playing, and given the full
reference vocal recording, **trace which moment of the reference is playing at each moment of
the mix** — correctly, even when the DJ loops a phrase, jumps to another section, or cuts.

## Formal setup

- **Inputs:**
  - `mix_vocals[t]` — the separated vocal stem of the mix over the span's time window.
  - `R[τ]` — the reference vocal recording (the acappella track), as a feature series.
  - An approximate placement we already have: `set_start` (where the span begins in the mix,
    ~2.2 s accurate from lyrics) and a rough `ref_start` (which part of the song it starts on).
- **Output:** a piecewise-linear path = segment list `[(mix_t, ref_start, ref_end)]`.
  - straight play → one segment (slope ≈ 1),
  - **loop** → `ref_t` jumps *backward* (the vocal repeats an earlier phrase),
  - **section-jump** → `ref_t` jumps *forward* (DJ skips ahead),
  - time-stretch → slope ≠ 1.
- **Score (trajectory accuracy):** sample many mix-times across the span; at each, is the
  predicted `ref_t` within 2 s of the ground-truth `ref_t`? Fraction correct = the metric.
  This is the honest generalization of "did we place it right" — defined for every span shape.

## Why it is hard (the core difficulty)

Acappellas are **highly self-similar**: a chorus repeats 3–4× with near-identical vocals.
Matching a ~12 s mix window against the reference matches *all* chorus instances almost
equally well — a **coin-flip among repeats**. The matched filter finds the right *content*
but cannot tell *which instance* the DJ is playing, and loops/cuts mean the true path is
**non-monotonic** (the query revisits earlier reference positions).

## The evidence (two GT sets, honest numbers)

Metric = trajectory accuracy (fraction of sampled mix-times whose predicted ref-time is <2 s off).

| | real pipeline | **oracle placement (GT set_start)** |
|---|---|---|
| acappella (all) | ~12% | **BB11 35% (n=17), BB12 44–47% (n=21)** |

Oracle-placement breakdown by span shape:

| shape | BB11 | BB12 | read |
|---|---|---|---|
| linear (played straight) | 43% | 62% | placement-limited — decodes well when placed right |
| oddratio (odd stretch) | 8–75% | 32% | mixed |
| **multiseg (A→C jumps)** | **12%** | **27–31%** | **the hard core — fails even with perfect placement** |
| loop | 0% | 81% | tiny n, unstable |

**Key fact:** even with *perfect* placement, multiseg acappella spans decode at 12–31%, and
multiseg is ~41% of acappella spans. So this is **not a placement problem** — it is a
signal-ambiguity + non-monotonic-alignment problem. The ceiling is ~35–47%; getting there
would roughly triple real acappella trajectory, but no further without better features too.

## What has been tried (do not repeat)

1. **Single-window matched filter** (`refine_ref_offsets`): finds content, coin-flips repeats,
   can't represent loops at all.
2. **Viterbi path decode over windowed matched-filter** (`path_decode.decode_path`): represents
   loops/jumps (stay-on-diagonal free, jump costs `lam`), but per-window emissions are ambiguous
   among repeats → wrong instance.
3. **Fiber-aware scoring** (credit any instance of the same repeat class as correct): isolates
   true placement error from repeat ambiguity, but doesn't *resolve* the trace.
4. **Directional jump penalty** (`lam_back`: penalize backward jumps to prefer forward-consistent
   instances): a `lam_back` sweep **hurt** — chasing a specific instance is the wrong frame.
5. **Feature = HuBERT** (phonetic, key-invariant — correct for vocals, beats chroma) but **too
   self-similar within one track** to localize precisely.
6. **Soft `ref_start` prior into the decode** (2026-07-06): **flat** — feeding lyrics' known
   start doesn't help, confirming the bottleneck is the trace itself, not the seed.

## Where a solution might come from (research directions)

- **Problem class / keywords:** audio-to-audio alignment; **subsequence / non-monotonic DTW**;
  music structure analysis; cover-song & performance-to-score alignment; **repeated-section
  disambiguation**; DJ-mix transcription. SOTA baseline to beat: **André, Schwarz, Fourer 2024**
  (multi-pass NMF on UnmixDB, warp + gain). Qfp (Sonnleitner & Widmer) for identity but
  explicitly "not meaningful for repetitive content."
- **A finer/other feature that breaks within-track self-similarity:** the repeats are near-
  identical *phonetically* but differ in the *specific rendition* — timing micro-variation,
  prosody, or (in the mix) the surrounding bed's bleed. A feature sensitive to the specific
  instance, not just the words, could disambiguate.
- **Structural priors the human annotator uses:** loops are almost always **bar-aligned** to the
  DJ's beat grid; phrases repeat on 4/8/16-bar boundaries. A decoder constrained to the grid
  (quantized jump points) has a far smaller, better-posed search than continuous ref-offset.
- **Global/joint decoding** rather than per-window: solve the whole span's path under a
  self-consistency + grid + monotone-within-segment prior, instead of independent window matches.
- **Learned alignment** once GT scales: train a model on the GT segment paths (mix-vocal, ref) →
  path. Gated on more labeled sets (Murph = the 3rd; this is why labeling unblocks it).

## Payoff and cap (be clear-eyed)

Success moves multiseg acappella from ~12% toward the ~35–47% oracle ceiling — a real, large
gain on the last weak axis. **Beyond ~47% requires the feature and placement to improve too**;
loop-tracing alone is capped there. This is a genuine research bet, not a quick win — enter it
deliberately, validate on BOTH BB11 and BB12 (and Murph once labeled) to avoid over-fitting one
set's n≈17–21.
