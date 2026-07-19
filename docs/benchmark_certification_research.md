# Certifying 48k unlabeled DJ sets you can't listen to

**Status:** research memo, 2026-07-14. Feeds the alignment paper effort
([docs/alignment_paper_draft.md], `project_alignment_paper_effort`). Load-bearing
idea: use the ~48k real 1001tracklists sets (vs 6 eval-bench / UnmixDB) as the
paper's DJ data, with a *certification pipeline* instead of exhaustive listening.

Produced by the `deep-research` workflow (run `wf_5549f175-629`): 16 sources,
73 claims, 25 adversarially verified, **12 confirmed**. The auto-synthesis and
~half the capture-recapture verifications died on an Anthropic weekly rate limit
— those arms are credible-but-not-independently-reverified (flagged inline). Re-run
after the limit resets to get the clean cited report.

---

## The reframe

This is the **test-oracle problem**: we need to test correctness at scale, but
producing the correct answer (a human listening) is the expensive thing we can't
do. It is *the* central open problem for ML systems ("they answer questions for
which no prior answer exists"). Naming it that way makes four decades of
software-testing + ecology-statistics literature directly applicable, and turns
the BB11-cents / BB12-semitone story from "we got lucky by listening" into a
**measurable coverage gap**.

**Do not try to certify all 48k sets.** Wrong goal, unachievable. The right goal:
*certify a metric on a sampled audit with a confidence interval, plus a quantified
estimate of how much failure mass lives in un-audited regions.* That second number
is the thing the team has never had — it is what would have flagged "cents-detune
is out there" **before** anyone listened to BB11.

Two error layers must stay separate (they have different noise sources):
- **identity** — tracklist wrong about *which tracks* (crowd-sourced human error).
- **placement/warp/pitch** — aligner wrong about *where / how* the track sits.
A capture-recapture estimate of unseen *aligner* failure classes is meaningless if
contaminated by *tracklist* errors. Model identity noise with confident-learning;
model placement/warp/pitch with the round-trip residual.

---

## Lever A — Oracle-free validation (catch errors with no human)

Strongest, most repo-ready arm.

- **Metamorphic testing** [MTAM, arXiv 2509.24215, verified 3-0]. No right answer
  needed, only a relation that must hold across runs: hold the label fixed, apply a
  semantic-preserving transform, count any output flip as a failure
  (Error-Finding-Rate = misclassified / generated). **Our version = the round-trip
  we half-have:** align → re-render the mix from recovered {placement, warp, pitch}
  → compare to the real mix. A cents-detune error nobody named shows up as a
  reconstruction residual. Adjacent to existing `als_audit` (GT-vs-mix ~97%) and
  reconstruction-supervision work. **Highest-leverage single idea here.**
- **Confident Learning / cleanlab** [Northcutt, arXiv 1911.00068, verified 3-0 ×3].
  Estimates the joint distribution of noisy-given vs true-latent labels from a
  class-conditional noise model using only predicted probs + noisy labels;
  **provably consistent, exact label-error identification under sufficient
  conditions**, data-derived thresholds (no human hyperparameter). This *is* the
  "model the tracklist noise" answer — ranks which sets are most likely mislabeled
  so we listen to those first.
- **Snorkel / data programming** [Springer s00778-019-00552-1, verified 3-0 ×2].
  Denoises multiple noisy labeling functions and estimates their accuracies with no
  ground truth, learning per-source reliability from agreement structure. Our
  labeling functions = existing channels (fingerprint, HuBERT vocal, lyrics-ASR,
  chroma, cue-detr). Replaces hand-tuned `source_priority` with a *learned*
  per-region reliability.

## Lever B — Spend the listening budget where it pays

- [arXiv 2411.07428, audio-to-score in-the-wild, verified 3-0]. Annotating only the
  cheapest high-leverage signal — **repeat/jump locations** — beat dense labels:
  33%→82% overall, 20%→83% on pieces with repeats. Transfer: humans confirm only
  *decision-critical events* (loop points, mashup seams, transition in/out), never
  whole sets. **Caveat [refuted 0-3]:** the paper's "under 6 sec/page" cheapness
  claim did NOT survive — budget real time per annotation.
- Active learning = point the audit at max-disagreement / thinnest-margin sets. We
  already hold the right instinct: `project_abstention_margin` (margin is the
  signal, absolute cosine useless). That is the acquisition function; apply it to
  set selection.

## Lever C — Estimating unknown-unknowns (the BB11/BB12 question, made rigorous)

The part actually asked about. Also the part the rate limit hit hardest — calibrate
confidence: capture-recapture claim verified, the rest credible-but-abstained
(verifier agents died mid-vote, **not** refuted).

- **Capture-recapture** [ScienceDirect S0164121203000906, verified 3-0]. Estimate
  remaining undiscovered faults from the **overlap between independent detectors**
  (large overlap ⇒ few remain; small ⇒ many undetected). Run N independent
  validation views (fp, HuBERT, lyrics, round-trip residual) as "inspectors."
  **Verified caveat [1-1, kept]:** most estimators *underestimate* — treat as a
  lower bound, calibrate against history.
- **Good-Turing / species-discovery** [ESA 10.1890/14-0550.1, verified 2-0].
  Sample-coverage theory estimates the full rank-abundance distribution *including
  species never seen* — not just how many failure classes remain but their
  frequency mass. Concrete estimator: unseen-class mass ≈ **f₁/n** (singletons /
  sample size). If after auditing K sets we still find new failure modes at rate
  f₁, unseen mass is not yet negligible → keep auditing. When f₁/n → 0 the taxonomy
  is saturated. **This is the meter that would have flagged cents-detune early.**
- Chao1/Chao2, jackknife-best-for-defects, and the "blind-spot mass B_n(τ)"
  pre-deployment metric appeared in fetched sources but were **abstained (rate
  limit), not verified.** Standard and credible; re-verify before citing in paper.

## Lever D — Benchmark construction rigor

Under-covered this run (search under-weighted UnmixDB/MIREX/DCASE construction
methodology — re-run this angle). Verified thread: when exhaustive testing is
impossible, **quantify residual risk from a sampled audit rather than treat a clean
run as proof** [arXiv 1807.10255]. For the paper: report the metric on a stratified
audited subset **with a CI**, plus Good-Turing unseen-mass as an explicit coverage
ceiling.

---

## The design this points to — a *certification pipeline*, not a labeling effort

1. Every set scored by ≥4 independent channels already built (fp, HuBERT,
   lyrics-ASR, round-trip reconstruction residual). No listening.
2. Snorkel-style label model learns each channel's per-region reliability with no
   GT; confident-learning ranks sets by P(mislabeled).
3. Round-trip metamorphic residual = oracle-free correctness proxy. High residual =
   suspect regardless of channel agreement. Catches the classes nobody named.
4. Active-learning queue: humans listen only to high-disagreement / high-residual /
   high-CL-error sets, confirming only decision-critical events.
5. Good-Turing meter on the failure taxonomy: watch f₁/n. Publish the metric with a
   CI **and** the unseen-failure-mass estimate as the honest coverage ceiling.
6. Capture-recapture across channels back-estimates undiscovered failure classes
   (lower bound).

## Caveats

1. Keep the two error layers (identity vs placement/warp/pitch) separate — see
   reframe above.
2. This run is partial: 12 solid claims; the species-discovery arm is
   credible-but-unverified (weekly rate limit). Re-run to confirm Chao / jackknife /
   blind-spot before the paper cites them.
3. Lever D under-covered — one more targeted pass on MIR-benchmark construction.

## Sources (verified subset)

- Confident Learning / cleanlab — https://arxiv.org/pdf/1911.00068
- Audio-to-score in-the-wild (repeats/jumps) — https://arxiv.org/html/2411.07428v1
- Snorkel / data programming — https://link.springer.com/article/10.1007/s00778-019-00552-1
- Metamorphic testing of audio moderation (MTAM) — https://www.arxiv.org/pdf/2509.24215
- Capture-recapture in software inspections — https://www.sciencedirect.com/science/article/abs/pii/S0164121203000906
- Good-Turing generalized rank-abundance (undetected species) — https://esajournals.onlinelibrary.wiley.com/doi/10.1890/14-0550.1

## Immediate proof-of-concept

Prototype the round-trip metamorphic residual on BB11/BB12 first. If it flags the
cents-detune set as high-residual *without being told to*, that's the proof the
scheme catches unknown-unknowns.
