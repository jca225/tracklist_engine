# Acquisition as a decision model — what our own history says

_2026-07-10. Grounded in the 1,267-row correction ledger, `docs/acquisition_lessons.md`
(41 entries), the BB11/BB12 re-download commit history, and `core/acquisition_case.py`._

## The question

Every song a DJ plays is a slot we must fill with the *right* audio: right
version (original vs a specific remix), right form (full / vocals / instrumental),
right length (radio vs extended), real audio, decent quality. We've paid for this
~1,267 times in corrections and dozens of re-download campaigns. Is there a model
here, and what's the right abstraction?

## What the history actually says

**1. Wrong-version is the dominant failure, and it kept coming back.** 363 of the
corrections are batch `mac_rescue` version fixes; the root cause was one bug
(`2cdb892`, bare `Artist - Title` collapsing every remix onto the studio original —
a ~63k-row class). The instructive part is that the *same hole reopened three more
times in different code paths* (`d0183c1` → `00c1a54` → `e5808c9`) until the lesson
was written down as "gates must live in ONE shared function." **Much of our pain
was not bad ML — it was a correct gate that didn't run everywhere.**

**2. Cheap deterministic gates work — where the signal is cheap.** The
`duration_sane` guard nearly erased the preview-clip / radio-edit class (~9 residual
in the whole ledger). That's the proof of the "detect-then-gate" pattern. But the
*same* class is still open where the gate isn't wired: the pi rescue duration cap is
disabled, and the variant axis (radio-vs-extended) has no gate at all (4 extended
rows corpus-wide).

**3. Candidate selection is real but not a quality problem.** Among human-picked
stem winners the first search result won ~69%, a later one ~31%. Q1 already proved
audio *quality* scoring can't beat "take cand1." What picks the winner is *identity
match* — the HuBERT vocal gate agrees with the human 84% of the time.

**4. Every failure left behind a scalar.** `rank_hits` weights (5/2/4/2), the HuBERT
floor 0.6, `duration_sane`'s 0.5×/3.0× band, chromaprint's uncalibrated threshold,
the 1200s cap. Each is a number someone set to remember one specific failure. That
pile *is* the thing a model would replace.

## The right abstraction already exists

`core/acquisition_case.py` models acquisition as exactly a decision problem, per
slot:

- **`CaseClaim`** — what the slot needs (recording, version, stem, variant).
- **`Attempt`** — each try: `query`, `url`, `actor`, a `checks` map (named:
  `version_gate`, `fingerprint`, `duration_ratio`, `bleed_residual_db`), and a
  **`verdict`** (accept / reject / promote / pending).
- **`Resolution`** — the winner + **`gt_confirmed`** (a human used it in Ableton).
- **`TrainingSignal`** — **`negatives`** (rejected urls) + **`preference_pairs`**
  (winner, loser). A training container, purpose-built for a ranker.

This is the Fellegi-Sunter accept / review / abstain decision record **and** the
training set, already designed. The abstraction question is answered.

## So what's actually missing

Not the abstraction — the **data wiring and the model**:

1. **We record corrections, not decisions.** The 1,267-row ledger is a great
   *negative* set ("this pick was wrong on this axis"), but it's free-text and it
   only fires *when something went wrong*. The `Attempt.checks` that a model needs
   as *features* — what the gates saw at accept time — are not logged on the ingest
   path. We have the outcomes without the inputs.
2. **The positive/preference labels are scattered** across three places
   (`gt_confirmed` cases, `WINNER.txt`, `out/discern/picks.jsonl`) and not
   consolidated.
3. **The `checks` vocabulary isn't frozen**, so features aren't comparable across
   attempts.

## The model to build (and the one NOT to)

**Build:** a per-axis **claim-satisfaction acceptance function** —
`accept(claim, candidate) → verdict` with calibrated **accept / review / abstain**
bands, not one monolith and *not* a quality ranker (Q1 killed that). It decomposes
by how each axis is best solved:

| Axis | Right tool | Status |
|---|---|---|
| real-audio / duration / exact-file | deterministic gate | works; just wire everywhere |
| **version identity** (biggest class) | **content version-ID verifier** vs the claim | **missing — highest value** |
| stem selection | HuBERT identity gate (84%) + abstain | exists; generalize into the function |
| quality among siblings | — | proven un-modelable (Q1); use search-rank prior |

The combiner is the research's **Snorkel label model** (the guards/gates are
labeling functions with coverage+accuracy, replacing the hand-tuned scalars); the
bands are **Fellegi-Sunter**; the review queue is ranked by **cleanlab**.

## The first move (cheap, unlocks everything)

**Turn the correction ledger into a decision ledger:** route every guard/gate
verdict into `Attempt.checks` on the ingest / rescue / replace paths, so *each
acquisition — success or fail — becomes a labeled example automatically*, and freeze
the `checks` vocabulary. This is small and mechanical, needs no ML, and is the
prerequisite for training anything. Do it, and in a few weeks of accumulated cases
(plus the existing ledger) the version-verifier is trainable.

## Honest priority

The acquisition model is the right *long-term* abstraction, but the immediate ROI is
plumbing, not ML: (a) deploy the insert-before-delete fix and re-enable the pi
duration cap + add the variant gate (still-open deterministic holes, no model
needed); (b) start logging decisions into cases; (c) build the version-ID verifier
when labeled volume exists; (d) the full acceptance function is a flywheel/Phase-3
piece. The north-star bottleneck is still the aligner and hand-labeling — this makes
each label count for more, it doesn't replace them.
