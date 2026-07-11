# The missing dimension: an information-dynamics mashup critic

*2026-07-11, from John's intuition that "our solution lies in information
theory — perhaps a novel methodology based on KL." A research proposal, not a
build order. Companion to [mashup_decision_model_plan.md](mashup_decision_model_plan.md),
[dj_craft_rules.md](dj_craft_rules.md), and the repo's existing info-dynamics
work ([[information-dynamics-bb12]], [[dj-selection-model]]).*

## The dimension v2 is blind to

Compiler v2 obeys **grammar** — syntax: which 16-bar hook, entered where, how
loud. It cannot judge whether a *specific pairing* is **interesting**: whether
the combination builds tension and releases it, whether the drop lands as a
surprise that still coheres. That is semantics, and it is exactly the axis the
Two Friends finding names: *"the most unexpected combinations become the most
memorable."* That sentence is an information-theoretic claim —
**memorability ∝ surprise, bounded by coherence.** Grammar is necessary and
not sufficient; interestingness is the missing term.

## Why information theory is the right mathematics

A mashup is two predictable things combined into a controlled surprise.
Information dynamics (Abdallah & Plumbley 2009, in `papers/`) is precisely the
mathematics of musical expectation and surprise: model music as a stochastic
process, and at each moment compute how much the newly-heard content updates
the listener's belief about what comes next. The update size **is** a KL
divergence:

    surprise_t = D_KL( P(future | past, x_t)  ‖  P(future | past) )
               ≈ −log P(x_t | x_{<t})        (instantaneous surprisal)

Aesthetic response peaks at **intermediate** surprise — the Wundt/Berlyne
inverted-U, formalized for music by Pearce's IDyOM and Abdallah's predictive-
information framework. Too predictable → boring; too surprising → noise/clash.
A great mashup traces a specific arc: surprise rises at the vocal entry (the
drop lands), holds in a coherent band through the section, resolves at the
exit.

## The proposed critic (novel part)

Model the **bed** as a predictive process over frame features. When the vocal
is layered on, measure the **surprise trajectory** — how much the vocal's
presence violates the bed's own expectation, frame by frame:

- predictive model `P(x_t | x_{<t})` over next-frame features of the bed;
- feed it the *mix* (bed+vocal); `surprise_t = −log P(x_t^mix | x_{<t}^bed)`
  (or the symmetric KL between bed-alone and mix posteriors);
- the mashup's **signature** = the trajectory, its integral (total information
  content), and its *shape at structural moments* (does surprise spike at the
  drop, sit mid-band during the hook, resolve at the exit?).

Score a candidate by distance from the "good arc" (the inverted-U band learned
from GT). This is different from the repo's prior use of surprise (boundary
localization): here surprise is the **quality signal on layered stems**, judged
as a trajectory against an aesthetic prior — which, per the whitespace scan
([[research-scan-2026-07-11]]), nobody does.

Three payoffs, one mechanism:
1. **Arrangement critic** → rank candidate pairings/placements → *automates the
   batch-and-whittle* (200→45) that the ground-truth artist does by ear.
2. **Feature** for the Stage-1 decision model.
3. **Reward** for DPO/CRPO — the critic manufactures preference pairs, the
   bridge over the cold-start-data gap the research scan identified.

## Honest priors and risks (do not skip)

- **The repo already found surprise does NOT rank across sets**
  ([[information-dynamics-bb12]]). That was *scalar* surprise for *boundary
  detection*. This proposal is a different object — a *within-mashup
  trajectory* scored against an *inverted-U*, on *layered stems*. Whether it
  ranks mashup quality is OPEN and the P0 experiment below settles it. Do not
  assume it works because it's elegant.
- **Feature space matters and MERT is probably wrong for it.** MERT's ~0.92
  self-similarity floor ([[mert-equivalence-floor]]) and the independent
  mashability-null result (raw MERT ≈ 0 correlation, research scan) mean the
  predictive model likely can't live in MERT-cosine space. Candidates: chroma
  (harmonic surprise), a small learned predictive model, or the
  reconstruction-supervision features. The right space is itself an experiment.
- **Novelty, stated honestly.** The building blocks — predictive information,
  surprisal, the inverted-U — exist (Abdallah, Berlyne, Pearce). The
  contribution is *applying the surprise trajectory as a mashup-quality critic
  on layered stems*, and (possibly) a KL formulation over the bed↔mix posterior
  shift. That is a real, publishable contribution, not a reinvention — but we
  should not oversell it as inventing information theory.

## P0 — the one experiment that decides everything (cheap, existing data)

**Does an information-dynamic signature separate good mashups from bad ones?**

- **Positives:** the BB11+BB12 GT spans (known-good, world-class mashups).
- **Negatives:** render the SAME vocals over *key/BPM-compatible but musically
  wrong* beds (random gate-passing pairings) — we can generate these for free
  with compiler v2 + the corpus.
- **Signal:** compute the surprise trajectory (start in chroma space — cheapest)
  for each; test whether a simple classifier on the signature (mean surprise,
  variance, drop-moment spike, inverted-U fit) separates positives from
  negatives above chance.
- **Verdict rule:** clean separation → the critic is real, promote to a build
  (pick the feature space, wire as ranker). Chance-level → the inverted-U
  doesn't capture mashup quality in this space; pivot (try learned space, or
  abandon and lean on the verb-log preference model instead).

Run P0 before writing any critic code. It reuses the aligner corpus, the GT
loaders, and Essentia/chroma we already have; no new infrastructure.

## Where it sits in the plan

This is the **arrangement-critic** lane of the decision model — it slots
between Stage 0 (rules, shipped) and Stage 2 (taste). It does NOT block the
product: v2 + verbs + the librarian bridge make Appleseed lovable while this
gestates. But if P0 separates, it is the single highest-value research bet,
because it is simultaneously the ranker (product), a decision-model feature,
and a preference-pair generator (training) — and it is the mathematical form
of the one thing grammar can't give us: taste as controlled surprise.
