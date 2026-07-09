# Stem-candidate quality ranking — "which acappella/instrumental sounds better"

Plan of record, 2026-07-09. Grounded in (a) the closed experiment
`workspaces/separation_qa/FINDINGS_acappella_quality.md` (verdicts respected,
nothing re-tested), (b) the 2026-07-09 internal audit (gate = identity not
quality; ~676 preference pairs with documented noise axes; **zero** rich
discern labels collected), and (c) verified external research
(Audiobox-Aesthetics: music-trained quality model with a Production-Quality
axis, pip+public checkpoint; SingMOS: singing-specific MOS predictor;
MERT-L12 embedding distance = best correlate of human singing-separation
quality, arXiv 2507.11427; pairwise > absolute for ranking, MOSPC).
Related: [acquisition_lessons.md](acquisition_lessons.md) §E (revealed policy).

## What exists vs what's missing

- `candidate_vocal_gate` (HuBERT-L9 matched filter vs the track's own
  separated vocal, 0.6 floor, margin 0.0) answers **"same master's vocal?"**
  — authenticity, not quality. Keep as the hard prefilter; never ask it to
  rank ties.
- `separation_qa/bleed_residual.py` ranks contamination **within** the
  online class (face-valid, 63 dB spread; ≤−40 dB clean, ≥−15 dB suspect)
  but collapses cross-provenance (double-separation). Unvalidated vs humans.
- Cheap cleanliness features INVERT cross-provenance (they reward separator
  aggression); only bandwidth (hf16k_ratio, rolloff95) points the right way,
  and it breaks on vintage material. A Bradley-Terry ranker on these went
  100%/100% train/LOO = provenance classifier, not a quality ranker.
- The human's demucs-vs-online choice is driven by **vocal-channel defects**
  (watery musical noise, gating, lost highs) that nothing in-repo scores —
  this is exactly what the MOS-net class measures.
- No bitrate/codec is recorded at candidate fetch; no clipping detector;
  Essentia/MERT are not computed on candidates.

## Phases

**Q0 — scorer battery, no training (≈2 days).**
Assemble per-candidate scores: (1) identity gate verdict (existing, hard
gate); (2) bleed-residual within-class + the instrumental mirror
(`msst_smoke._bleed_score`, NOT subject to the collapse); (3) **SingMOS +
Audiobox-Aesthetics PQ** on the vocal candidates (the closed experiment's
named untried lever — run relative-within-track only, never absolute
thresholds); (4) forensics: existing bandwidth features + NEW clipping rate
+ ffprobe `abr/codec` recorded into `candidate_stems_manifest.json` at fetch
time (fixes the metadata gap); re-pitch by `pitch_semitones` before any
spectral feature.

**Q1 — validate against revealed preferences BEFORE any training (gate).**
Score every candidate in the **107 BB12 human-winner slots** (the only clean
human labels; BB11's 67 are the gate's own picks — circular, sanity only).
Metric: within-slot winner agreement among identity-passing candidates, per
scorer and simple fusions, vs the cand1-prior baseline (67%). Cost: ~a day
of Mac compute. If a scorer/fusion clears the bar → wire it as the tie-break
behind the gate and into the D3 auto-accept bands + acquisition-case records.
If nothing clears → Q2 is mandatory, Q0 scorers become features not deciders.

**Q2 — the data step the closed experiment demanded.**
Collect **within-class** labels with the already-built rich UI
(`review/review_server.py`, port 8800: clean/keep/diff, ⇄ interchangeable,
pitch_semitones) on **BB10's 537 unlabeled candidates** once its stem
backfill lands. These are the uncontaminated preference pairs (same
provenance class, identity pre-gated). Ties are labels too (⇄) — the gate's
margin-0.0 rationale says many slots genuinely tie; per John, some
acappella/instrumental choices are inherently arbitrary — expect and keep ⇄.

**Q3 — learned ranker (only if Q1 underperforms).**
Bradley-Terry/RankNet head over frozen **MERT-L12** + HuBERT-L9 pooled
embeddings + Q0 scalars. Training data: Q2 labels + the 107 BB12 human pairs,
with the audit's noise axes handled explicitly — drop identity-failed losers
(they'd re-teach the gate), provenance-balance or hold out demucs>candidate
pairs (provenance is trivially separable — the documented failure), treat ⇄
as ties. Eval: leave-one-set-out; **ship bar = beat the best Q1 fusion on
held-out winner agreement**. A 100% LOO score is a red flag, per precedent.

## Non-goals / guardrails
- No separator-free absolute quality score across provenance classes (closed
  verdict stands). All scores are relative within a slot.
- No blanket re-ranking of already-human-resolved slots — detect-then-correct.
- Degenerate-axis slots (~10% acap / >50% instr claims): the right answer may
  be the separated stem or even the full track; the ranker must be allowed to
  say "tie / no preference".
