---
title: "The Alignment Learning Plan"
subtitle: "Where we stand, what's learnable, and how we bootstrap our way to a robust aligner"
author: "Tracklist Engine — strategy review"
date: "June 2026"
geometry: "margin=1.1in"
mainfont: "Palatino"
fontsize: 12pt
linestretch: 1.18
colorlinks: true
linkcolor: "RoyalBlue"
toc: true
toc-depth: 2
header-includes:
  - \usepackage{newunicodechar}
  - \newunicodechar{→}{\ensuremath{\rightarrow}}
---

\newpage

# The one-paragraph version

We are not building one model — we are building a handful of learning problems, only one of which (the aligner) is the real deliverable. Today the aligner doesn't exist as a trained model; what exists is a set of hand-tuned decoders sitting around **55–60% identity accuracy on the easy slices** and failing on the hard ones. The path past that plateau is not another clever probe — it is **labels** (which you're about to produce) and **synthetic pretraining** (scaffolding to survive the label shortage), assembled into a **bootstrapping flywheel** that grinds toward robustness on the real corpus. The single biggest risk is that real DJ sets live on a tiny, low-rank sliver of an enormous space, and a naive data generator would drift off it without us noticing. The defenses against that are concrete, and the whole thing is steered by one unbiased ruler: the hand-labeled ground truth.

\newpage

# 1. What learning algorithms are we actually creating?

It helps to sort them by *how they get supervised*, because that — not model architecture — is what determines whether each one can be made robust.

## Self-evaluable (robustly learnable now)

These can be measured and improved without scarce human labels, because the ground truth is in the data itself.

- **Stem retrieval / vocal verification** — "is this the right vocal?" Reframed from decode to *verify*. The strong result: **HuBERT-L9, 81% retrieval@1**, margin +0.233. Lyrics don't transpose, so wrong-track peaks collapse. This is the healthiest model we have.
- **Fiber detection** — clustering the repeat instances of the same content within one mix. Self-supervised by construction. Honest contribution: **+6pp** to decode (53→59%), *not* the inflated 70% from v1.
- **Wrong-*content* detection** — fingerprint localizer gives a sharp (~41pp) signal. Blocked by **data, not modeling**: the fingerprint table is under-populated.

## Human-label-gated (rate-limited by labeling throughput)

- **Full alignment** — the deliverable. Not assembled end-to-end; starved of ground truth.
- **Wrong-*version* detection** (remix vs original, extended vs radio) — same content, different production. Fingerprinting does *not* solve this; needs the identity model + labels.
- **Stem quality** — and this one comes with a warning (below).

## The quality-scoring trap

Every separator-free quality metric we tried **inverts** (separators over-strip, so "less energy" reads as "cleaner"). The 100%-accurate ranker turned out to be a **provenance detector** — "studio vs Demucs," not "good vs bad" — and collapsed across provenance. A universal quality scorer is **not learnable** from the signals we have. What *is* shippable: a narrow within-class gate and DNSMOS for vocal artifacts. Reframe quality as **relative and alignment-derived** ("does stem A align better than stem B?"), not as a standalone classifier.

\newpage

# 2. Where the aligner actually stands (with numbers)

There is **no single "alignment accuracy"** because there is no assembled end-to-end aligner — only decoder probes, each measured on one or two hand-labeled sets (tens of spans). Every number is small-$n$ and slice-specific.

| Probe | Slice | Result | Caveat |
|---|---|---|---|
| Continuity-stack refine (placement) | BB12, **regular** stem | exact <2s: **38→54%** | best case |
| Fiber-aware decode (identity) | BB12, all classes | **53→59%** | honest figure |
| Fingerprint fusion (identity) | BB12, $n{=}29$ | **52→58%** | +6pp |
| Path-decode, chroma (identity) | all classes | **53%** | HuBERT lifts acappella 32→44% |
| Constrained local-refine | one set, ±45s band, mostly acappella | median ~30s, <10s **12%**, <2s **0%** | hard slice |
| Vocal retrieval (verify) | BB12 pool | **81%@1** | retrieval, *not* placement |
| Abstention (precision@coverage) | BB12 | overlay 73%@22%, bed 93%@66% | knowing when *not* to predict |

**Read it plainly:**

- On the favorable slice (regular stem, coherent single track), the best probe hits **~54–60% within 2 seconds.** Better than chance, not done.
- Off that slice it falls apart: acappella placement 12–44%, instrumental worse, short EDM drops broken (chroma can't localize them).
- The one genuinely strong result — 81% vocal retrieval — answers *"is this the right vocal?"*, not *"where does it go?"* Don't let it stand in for alignment.
- **Abstention works**, which matters: a 55%-accurate aligner that knows which 55% it's sure of is far more useful than the raw number suggests.

The +6pp increments are the tell: **hand-tuned DSP has hit diminishing returns.** The way past ~60% is a *trained* model, which means data.

\newpage

# 3. The plan: label, measure, synthesize, train

## What the labeling unlocks

The two weeks of work (the port-8800 UI + alignment GT for BB10/11/12, Murph, Disco Lines) changes the project's *category*, not just its numbers — from "DSP probes on one set" to "a trainable problem across a real distribution." Concretely it produces:

- **Alignment ground truth across multiple DJs and styles** — Murph and Disco Lines matter *more* than another Big Bootie set, because they're the first test of generalization *off* the mashup style we've overfit to.
- **Identity negatives** (the `diff` marks) — confirmed wrong-track/wrong-version examples. Robust, human-confirmed, and exactly what the decode model and wrong-version gate need.

## The sequence after labeling

1. **Measure across the expanded GT.** Get the first end-to-end number, and the first cross-DJ generalization read. This is cheap and it's a fork in the road: it tells us whether to go straight to synthesis or plug a generalization hole first.
2. **Build the synthetic-mix generator.** Highest-leverage build — the only path to a *trained* aligner given that ~6 labeled sets is far too little to train from scratch.
3. **Train the first real aligner.** Stem-wise, one shared MERT/HuBERT backbone, multiple heads (identity / placement / abstain). The `diff` negatives go straight in as hard negatives.
4. **Ship the ingest gates** the labels unblocked (wrong-version; narrow within-class quality) — independent of the aligner, so they land sooner.

\newpage

# 4. The bootstrapping flywheel (and one crossed wire)

There are **two different data engines.** Conflating them is the classic "the model generates its own training data and then mysteriously never improves" mistake.

## Engine A — synthetic generation (hand-built, comes *before* the aligner)

The synthetic mixes are **not labeled by a model.** A deterministic generator takes known catalog tracks, applies known transforms (transition, warp, loop, layer), and *renders* a mix whose ground truth is perfect **because we constructed it.** This is just supervised learning on fabricated data. Its only job: solve the cold-start that ~6 real sets can't.

> **Terminology trap:** "pretrain" here means *a training phase of the aligner*, **not** the step-2 *generation pretrain* (the HRM-Text taste model, a different codebase). The aligner is **one model trained in two phases**: pretrain on synthetic → fine-tune on real GT. Your labeled data still trains the aligner — in phase two. If we had thousands of labeled sets we'd skip synthetic entirely.

## Engine B — self-training on the real corpus (the actual bootstrap, comes *after*)

```
  hand-built generator  -->  synthetic mixes  (perfect GT, high volume)
                                   |   pretrain
                                   v
  real GT (BB + Murph + Disco) --> aligner v0
                                   |   run on the unlabeled corpus;
                                   |   keep only high-confidence + corroborated
                                   v
  pseudo-labels  -->  retrain  -->  aligner v1  -->  v2  -->  ...
```

The rest of the corpus isn't just a *consumer* of the finished aligner — it's the **fuel** that makes it robust. "Align the rest of the sets" and "become robust" are the *same* loop.

## Two ways the loop rots — and the guards

- **Confirmation bias.** Self-training reinforces what the model already believes. *Guards:* never promote low-margin predictions; require an independent corroborator (fingerprint / ACRCloud); keep the hand-GT **frozen** as an eval every round to detect drift.
- **Synthetic–real gap.** If synthetic transitions don't resemble real DJ behavior, the pretrain helps less. *Guard:* the multi-DJ GT calibrates the generator and the decisive ablation (below).

\newpage

# 5. The low-rank risk — the most important objection

Real DJ sets occupy a **tiny, low-rank sliver** of an enormous space of possible track-combinations-with-transforms. A naive generator would put mass *off* that manifold and bias us without our knowing. This objection is correct, and it kills the naive version of Engine A. Here is the architecture that survives it.

## Synthesize the *physics*, never the *grammar*

Split what the generator produces:

- **Mixing physics** — beatmatch, time-warp, EQ transition, layering, loop, and *how each shows up in the features.* This is **not** low-rank; a beatmatch is a beatmatch in any genre. Synthesize aggressively — it's on-manifold by construction.
- **Arrangement / selection grammar** — which tracks, in what order, with which transitions, at what layer-depth. This **is** the low-rank manifold. **Do not fabricate it.**

## Augment real skeletons, don't invent mixes

Make the generator a **content-recombiner over real arrangement skeletons:** take a real GT timeline (its real timing, transitions, layer structure) and swap *different track content* into it. The arrangement stays on-manifold *by construction* — it *is* a real arrangement; only the combinatorics of content expand. Sample whatever you must (transition mix, layer depth, durations) from the **measured** distribution, never an imagined one.

## Detect the bias you can't see directly

- **Real-vs-synthetic discriminator** — if it trivially separates them, the synthetic is detectably off-manifold (and its gradient says *where*).
- **Coverage in MERT space** — precision/recall of distributions: are synthetic samples realistic, and do they span the real variety?
- **The decisive ablation** — on *frozen* real GT, compare *pretrain-then-finetune* vs *finetune-only*. If synthetic pretraining **raises** held-out real accuracy, it's net-useful regardless of how "unrealistic" it looks. If it lowers it, cut it. **Unfakeable.**

## Low-rank is actually good news for Engine B

A low-rank manifold means coverage is *tractable*, not hopeless — and the real corpus is a *dense* sample of a *small* space. Self-training samples the true manifold directly, anchoring the long-run distribution to reality.

\newpage

# 6. Open-set reality: ODESZA, RÜFÜS, Galantis, and "stuff we're not privy to"

Live/hybrid acts **break the core assumption** that a set = an arrangement of identifiable recorded tracks. Two distinct breaks:

1. **Live re-performance** — same *work*, a *recording we don't have*; won't match the studio fingerprint. The honest label is "work = X, recording = live/unknown." (Note: ACRCloud also fails on live, so the fallback is self-reference + human work-level ID.)
2. **Untracked stems / IDs** — no catalog referent at all. The correct label is **"unknown,"** not a track ID.

## Include them — but the label is open-set, not a track ID

A mashup-only training set can never teach **calibrated abstention**; these sets can. What's labelable even on unknown content: segmentation, an **identifiability flag** (catalog / stem / live / unknown / non-musical), **self-reference** (this recurs at 4:12), and layer structure. The worst failure is a model that confidently hallucinates a match on a live guitar solo — and that's exactly what excluding these sets produces.

## The decision is *routing*, not include/exclude

"Has external stems" is a **bucket assignment**, not a keep/discard:

- **Clean-mashup bucket** → bulk of labeling, feeds identity-decode. Keep live sets *out* of this.
- **Stress / open-set bucket** → a small, deliberate set of hard examples, labeled open-set. Keep them *in* the corpus as calibration + held-out robustness.

So don't *exclude* the Galantis EDC set — **reclassify** it. And festival Galantis is probably mostly DJ/tracks+IDs, not a full live band, so it may be more alignable than feared. **The only way to know is to listen** — which makes commute-listening the triage mechanism itself.

## Commute-listening — yes, with one boundary

You are the **ground-truth oracle**; label quality is bounded by how well you perceive the mix. But listening is for **intuition and triage**, *not* labeling — the rigorous version is still the Ableton GT with the real stems. Log the surprising moments; that stream becomes the discovery taxonomy.

\newpage

# 7. The measure-theoretic frame (why it's the right scaffold)

This is more than a metaphor — it imposes the right discipline.

- **Sample space $\Omega$** — the atoms are **segments** (`(set, time-interval)` spans), not whole sets. That's the granularity we label and predict at.
- **$\sigma$-algebra $\Sigma$ = the label schema** — the type taxonomy is a *partition* of $\Omega$; $\Sigma$ is generated by it. A good schema is a partition into events whose membership we can *decide*.
- **Completeness — why the "unknown" bucket is mandatory.** The phenomena we're not privy to are *events not in $\Sigma$*; we cannot measure what $\Sigma$ can't resolve. The explicit **unknown complement** is what makes the partition exhaust $\Omega$. Without it, positive mass sits in no measurable set.

## The real insight: two measures, and the bias is their mismatch

- $\mu_{\text{real}}$ — the true distribution of segment-types. Unknown; what we want.
- $\mu_{\text{train}}$ / $\mu_{\text{synth}}$ — what our data / generator induces.

The synthetic-bias worry is exactly that **$\mu_{\text{synth}}$ is not absolutely continuous w.r.t. $\mu_{\text{real}}$** — it puts mass where real sets have none, so the density doesn't exist there. "Dark matter" is the dual: positive $\mu_{\text{real}}$, near-zero $\mu_{\text{train}}$. Robustness is the requirement $\mu_{\text{real}} \ll \mu_{\text{train}}$ *on the region we care about* — the whole bootstrap is machinery to push $\mu_{\text{train}}$ to dominate $\mu_{\text{real}}$ on the manifold.

## Discovery labeling is a filtration

We can't write down $\Sigma$ fully a priori, so we refine it: $\Sigma_0 \subset \Sigma_1 \subset \Sigma_2 \subset \cdots$, where the "unknown" atom at stage $n$ resolves into named atoms at stage $n{+}1$ (commute-listening + the v0 aligner's low-confidence clusters drive the refinement).

## The actionable payoff — and the trap

**Payoff:** stratify so every type of positive $\mu_{\text{real}}$-measure gets *nonzero empirical measure* in the labeled sample. A tail event with zero samples has an error estimate of infinite variance — you literally can't measure robustness on it. **That is the rigorous reason for the stress slice.**

**Two traps to flag honestly:**

1. **The atoms are fuzzy, not crisp.** "Clean-mashup" vs "some live elements" is a continuum. A hard partition may be the wrong object — consider a measure on a **product attribute-space** (identifiability, liveness, layer-depth as scores) rather than a single category.
2. **Measurability $\neq$ estimability.** Defining $\Sigma$ gives the *language* to state coverage, not the coverage. Estimating the *tail* of $\mu_{\text{real}}$ is hard finite-sample statistics that the measure theory doesn't touch. Use the frame for **conceptual hygiene** — name $\Omega$, make the partition exhaustive, separate the two measures, state robustness as bounded error on a large-$\mu_{\text{real}}$ set — then go do the statistics.

\newpage

# 8. What to do, in order

1. **Label** (next two weeks): bulk = clean mashup sets (BB10/11/12, Murph, Disco Lines) with full Ableton/`.als` rigor; plus a small **deliberate stress slice** of live/open-set examples (Galantis, ODESZA, …) labeled with the open-set schema. Collect `diff`/identity negatives throughout.
2. **Measure** end-to-end across all the new GT, including the stress slice. Get the headline number and the cross-DJ generalization read.
3. **Build** the synthetic generator as a *content-recombiner over real skeletons* — synthesize physics, not grammar.
4. **Train** the aligner: pretrain on synthetic → fine-tune on frozen real GT. Validate with the *pretrain-vs-finetune-only* ablation.
5. **Bootstrap** (Engine B): run on the unlabeled corpus, promote only high-confidence + corroborated predictions, retrain, repeat — with the frozen GT as the unbiased ruler and open-set recognition exploring the dark regions.
6. **Ship** the ingest gates (wrong-version; narrow within-class quality) in parallel.

**The single thread through all of it:** the hand-labeled ground truth is not training volume — it is the *measurement anchor* the entire bootstrap is steered by. That is why these two weeks matter.
