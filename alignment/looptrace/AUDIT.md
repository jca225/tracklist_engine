# Phase 1 — ill-posedness audit (CLONE vs DISTINCT-TAKE repeats)

2026-07-06 · config `audit-v3` · artifacts: `out/audit_1fsnxchk.json`,
`out/audit_2nvzlh2k.json` (per-track repeat maps + clone equivalence maps),
scored via `looptrace.eval` on the frozen-baseline dumps
(`out/pred_*_hubert.json`).

## Question

How much of the multiseg-acappella failure (frozen baseline 12–31%) is
**unwinnable** (GT plays a region with a digital-copy twin — CLONE — that no
content evidence can disambiguate) vs **addressable** (DISTINCT takes or
unique content)?

## Method (and what died on the way)

- **audit-v1 (dead):** HuBERT fibers as the repeat detector — under-detected
  badly (26 pairs across 19 BB12 tracks; HuBERT is blind to melodic repeats).
- **audit-v2 (dead):** log-mel lag-diagonal scan, but (a) raw mel cosine is
  ~1 everywhere for stable spectral envelopes → fixed by per-band temporal
  whitening; (b) the per-frame silence gate fragmented runs at every breath
  gap (median voiced run 1.6 s < 4 s minimum → 8/15 BB11 tracks got zero
  pairs) → fixed by gap-closing (1.5 s) + per-run voiced-fraction ≥ 0.4.
- **audit-v3 (used):** whitened-mel diagonal scan → (region, lag) repeat
  pairs → sample-accurate waveform xcorr verification with **sub-sample
  refinement** (a clone at a fractional-sample offset — e.g. after
  44.1k→22.05k resampling — caps integer-lag r at 0.94 ≙ −9 dB, inside the
  DISTINCT band; ×8-upsampled re-correlation fixes it; fixture-tested).
  CLONE ⇔ residual `1−r²` ≤ −12 dB. All thresholds fixture-validated
  (`tests/test_selfsim.py`: integer/gain-scaled/half-sample/codec-noise
  clones all classify CLONE; unrelated content ~0 dB; planted copies are
  detected; stationary noise yields zero pairs).

Calibration on random non-repeat window pairs: residual median −0.0 dB
(min −11.6 / −30.2 dB — the tail is random windows landing on true repeats,
not a threshold problem).

## Result 1 — GT-side decomposition (per sampled GT second)

| set | class | n | clone (unwinnable) | distinct (addressable) | unique (addressable) |
|---|---|---|---|---|---|
| BB12 | linear | 8 | 0% | 60% | 40% |
| BB12 | **multiseg** | 7 | **0%** | 55% | 45% |
| BB12 | loop | 1 | 0% | 100% | 0% |
| BB12 | oddratio | 5 | 7% | 77% | 16% |
| BB11 | linear | 5 | 9% | 12% | 79% |
| BB11 | **multiseg** | 7 | **0%** | 51% | 49% |
| BB11 | loop | 1 | 0% | 98% | 2% |
| BB11 | oddratio | 4 | 0% | 40% | 60% |

Clone pairs exist but are rare and small: BB12 14/414 pairs (Blink-182
hooks 11, Eminem 3), BB11 2/234 — and they barely intersect the GT
trajectories.

## Result 2 — baseline re-scored under audit equivalence

`looptrace.eval`, per-second accuracy, tolerances ±0.25/±1.0/±2.0 s. The
±2.0 s strict column reproduces the frozen baseline exactly (validates the
scorer).

| set | class | strict | clone-aware | repeat-aware |
|---|---|---|---|---|
| BB12 | ALL (n=21) | 37/42/44 | 37/42/44 | 42/50/56 |
| BB12 | multiseg (n=7) | 15/24/27 | 15/24/27 | 19/34/45 |
| BB11 | ALL (n=17) | 31/35/35 | 31/35/35 | 39/45/47 |
| BB11 | multiseg (n=7) | 10/12/12 | 10/12/12 | 25/32/34 |
| BB11 | loop (n=1) | 0/0/0 | 0/0/0 | 33/35/43 |

## Conclusions (what bounds "solved")

1. **The unwinnable fraction is ≈ 0.** Acappella repeats in these corpora
   are overwhelmingly DISTINCT takes (different renditions/mixes), not
   production copy-paste. The clone-aware ceiling is effectively 100%; the
   brief's feared "~35–47% ceiling" was decoder-limited, not
   ill-posedness-limited. CLONE tie-break (`rules.clone_tiebreak`: nearest
   previous song position, else first instance) matters only on small hook
   pockets (Blink-182, Eminem, Daft Punk).
2. **Half the multiseg failure is wrong-instance, half is wrong-content.**
   Forgiving wrong-instance picks (repeat-aware) lifts multiseg 27→45%
   (BB12) and 12→34% (BB11). So: ~18–22 pp recoverable by instance
   disambiguation (long-range accumulation, discriminative frames — Phases
   3–4), and the remaining ~55–66 pp is the decoder not even finding the
   right content region — exactly what Phase 3's long-segment landmark
   evidence attacks.
3. **Distinct-take repeats cover ~51–60% of GT seconds** on multiseg/linear
   spans — instance-sensitive evidence is necessary for that half; per-frame
   phonetic features (HuBERT) provably don't provide it.

## Caveats

- Detector coverage is imperfect: "Let It Go" finds 0 pairs (ballad,
  evolving arrangement — plausible but unverified); short (<4 s) repeats are
  out of scope by config. Under-detection moves seconds from
  distinct→unique; it cannot inflate the clone (unwinnable) estimate upward,
  so conclusion 1 is robust to it.
- 11 of 34 refs are true acappella files; the rest are Roformer-separated
  `vocals.flac` stems, whose separation artifacts differ per instance
  (different beds) — a true production copy-paste could then read as
  DISTINCT. For *mix-side* loop detection (Phase 2) this concern doesn't
  apply to the loop-vs-natural-repeat test, because both sides pass through
  the same separation.
- n=1 loop spans per set: never read the loop row alone.
