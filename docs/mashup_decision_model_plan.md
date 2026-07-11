# Mashup Decision Model — pretrain → post-train plan

*2026-07-11. Companion to [startup_strategy.md](startup_strategy.md) and the
DJ-agent spec. Frames the mashup compiler's "taste" as a learned policy with
a pretrain → post-train lifecycle, per John's framing: general mashup grammar
is PRETRAINED from DJ behavior at scale; individual taste is POST-TRAINED
from per-user priors + in-app preference signals.*

## The three stages

**Stage 0 — hand rules (this week).** Compiler v2 implements
`bb_mashup_grammar_v1` as deterministic rules (16-bar first-chorus hook,
pickup-led entry ~bar 17, LUFS match). Not the end state — the bootstrap
policy whose job is to (a) clear the ear-test bar, (b) start generating the
preference stream Stage 2 needs. A model with no users has no post-train
data; rules ship first. (Same bootstrap-flywheel shape as the aligner.)

**Stage 1 — pretrain the decision model.** A model that, given two analyzed
songs, predicts the arrangement a good DJ would choose:
`(vocal profile, bed profile) → {hook window, entry bar, span length,
gain offset, loop pattern}`. This is objective-4 (emulation) pulled forward
as the product's brain. The aligner remains the **data engine**: every mix
it can decode becomes a training example of real DJ decisions.

**Stage 2 — post-train on taste.** Two signal classes:
- **Cold-start prior**: the `personalization/` export (SoundCloud cohorts,
  per-user taste priors) conditions the genie before any in-app behavior.
  First honest use is SONG-level: rank the picker / suggest pairs.
  Arrangement-style conditioning (e.g. vocal density by cohort) is
  speculative until verb data exists — don't build it on faith.
- **In-app preference stream**: every refinement verb tap is a labeled pair
  (pre-refinement grant ≺ post-refinement grant) — native DPO data. Keeps,
  re-listens, shares, abandons complete the signal.

## The data ladder (what trains Stage 1)

| Tier | Source | Scale | Quality | Status |
|---|---|---|---|---|
| Gold | BB11+BB12 GT spans | ~300 spans | exact | exists |
| Silver | aligner-decoded mixes (abstention-gated) | unbounded | identity 83%, placement ~7 s median | UNVALIDATED for this use |
| Bronze | synthetic renders | unbounded | by construction | helps decode-level only (closed-experiments ledger) |

Key insight: the decision model needs **coarser labels than the aligner's
hardest cases**. "Which chorus, entered at which bar, for how long" is
16-bar-resolution; 7 s median placement error may already be inside
tolerance. The aligner doesn't have to be SOTA to feed the generator —
it has to be *unbiased at grammar resolution*.

## P0 experiments (falsifiable, cheap, ordered)

1. **Silver-data validation (the load-bearing one).** Run the aligner over
   ~20 non-BB mixes with abstention on; extract the same grammar statistics
   as `bb_mashup_grammar_v1` (span lengths, source position, entry phase,
   density). If silver distributions match gold, the pretrain corpus is
   effectively unbounded and Stage 1 is GO. If they diverge, measure whether
   the divergence is aligner error (fix upstream) or corpus diversity
   (interesting on its own — BB grammar may not be universal).
2. **Tiny-model-vs-rules (lane D reconstruction eval).** Train the smallest
   possible model (GBM / logistic over hand features) on gold only,
   leave-one-set-out: does it beat the Stage-0 rules at predicting held-out
   BB arrangement choices? If a tiny model can't beat rules on gold, a big
   model won't either until silver exists. Guards against premature ML.
3. **Instrument the verbs NOW.** Add refinement verbs + keep/abandon logging
   to seam before any model exists — preference data accrues from day one
   and the schema is trivial (mash_id, verb, before/after params, listener).
4. **Cold-start wiring probe.** Read the personalization export contract;
   confirm the taste bundle can rank a song list for one known user (John).
   Song-level only; no arrangement conditioning.

## What this changes about sequencing

Compiler v2 (Stage 0) stays first — it is both the ear-test fix and the
data-collection instrument. P0-3 (verb logging) lands with it. P0-1 and
P0-2 run in the research repo alongside. Stage-1 training only begins once
P0-1 says the silver corpus is real; Stage-2 personalization conditioning
only once verbs have data. No deep net before the bandit: with early data,
per-user adaptation is a contextual bandit over the ~6 grammar parameters,
upgraded to learned adapters when scale justifies it.

## Honest risks

- Silver data may inherit aligner bias (e.g. under-detecting short stabs →
  grammar skews long). P0-1's distribution comparison is the detector.
- Gold is one duo's style (single-annotator caveat, already flagged in the
  DJ-agent spec): a model that nails BB emulates Two Friends, not "DJs."
  Fine for a taste-model product; flag before any generality claim.
- The product must not wait on Stage 1: rules + verbs + librarian bridge
  make the app lovable while the model gestates.
