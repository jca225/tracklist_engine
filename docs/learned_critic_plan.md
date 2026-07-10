# Learned critic — train the verifier, not (only) the actor

**Status: PROPOSED, not scheduled** (idea captured 2026-07-10). Relates to
[kernel_data_engine_plan.md](kernel_data_engine_plan.md) W2 (fusion v2 is the
hand-built critic this would race), W4 (offboard labeler audit), and W5 (the
auto-accept gate). Nothing here blocks those; this is a candidate upgrade to
the gate, not a new lane.

## The idea

The actor (end-to-end aligner) is hard: it must *search* — which recording,
which repeat instance, what offset, what warp — and the hard tail (acappella
trajectory, placement-under-crosstalk) has known walls. The critic answers a
much smaller question: **given a hypothesis (track, span, ref_offset, warp),
does the audio agree?** That's pattern-matching against a pinned target, not
search. Train a model for *that* role, and use it to (a) rerank proposer
candidates, (b) band actor output into accept / review / reject
(Fellegi–Sunter, same abstraction as the stem pipeline), and (c) gate
pseudo-GT for training the actor.

The honest framing of "automate labeling": a critic converts labeling from
**authorship to auditing**. The human verifies proposed spans in Ableton
(~10x faster than hand-placing them) instead of authoring from scratch. It
does not remove the human — structurally *cannot*, see catches below.

## Why verification is demonstrably easier in this domain

We already have strong evidence that the verify direction works, from
hand-built components:

| evidence | result |
|---|---|
| phase-cancel | a *certificate* for exact-content clones — a perfect critic where it applies |
| fp localizer | "sharp wrong-content" — razor as verifier even where unreliable as proposer |
| HuBERT vocal verify (proposed vs actual vocal) | 81% — explicitly framed as VERIFY, not search |
| WS1 precision fusion (`neuro/`) | correctness AUC **0.75**, monotonic abstention — the hand-built critic baseline to beat |
| `als_audit` | verifies finished GT vs mix at ~97% — verification is near-solved when the hypothesis is right |

## Why training the critic is unusually cheap

The actor is supervision-starved (two fully-labeled sets). The critic is not:

- **Positives:** every GT span in BB11/BB12 — hundreds of
  `(mix window, ref window, hypothesis)` triples.
- **Negatives:** synthesized by perturbation, graded by perturbation size —
  shift the offset ±5 s, swap the sibling recording, wrong stem, wrong repeat
  instance, wrong warp ratio. Dense, *hard* negatives for free.
- **Unlimited synthetic positives:** the mashup renderer
  ([synthetic_mix_plan_v2_bb12.md](synthetic_mix_plan_v2_bb12.md)) produces
  exactly what a critic scores — "does this rendered relationship hold" is
  true by construction. Synthetic already proved net-positive for the
  trajectory decoder (+0.048 held-out); the match is even tighter here.

Classic contrastive/verification setup, orders of magnitude more supervision
density than end-to-end decode.

## What it buys, in value order

1. **Rerank/gate existing proposers.** fp-placement, HuBERT set_start, lyric
   anchors, chroma already generate candidates with decent recall and
   *complementary* failures. A learned critic replaces the hand-tuned
   `source_priority` axis ordering and static precision tables.
2. **Verification labeling.** Actor proposes → critic bands → John audits the
   review band in Ableton (the worst-spans PRED-vs-GT A/B seeder is this
   loop's rendering half already). GT throughput up ~10x even with zero
   fully-automatic labels. Feeds W5's queue directly.
3. **Pseudo-GT gate.** Critic-accepted spans become tiered training data for
   the actor (W4's `agentic auto-accept` tier), FixMatch discipline applies.

## The catches (real, not hedges)

- **Repeat instances break local verification.** A
  locally-correct-but-wrong-instance hypothesis *verifies* — the content
  genuinely matches (the repeated-chorus argmax problem again). A span-level
  critic is not a set-level critic; trajectory-level consistency (warp prior,
  continuity, fiber structure) must be scored too, which reintroduces a slice
  of the actor's difficulty.
- **The flywheel plateaus on the easy manifold.** Probe-gated bootstrap
  already plateaus ~55–60%. A critic auto-accepts what the system already
  handles; the hard tail (acap under crosstalk, loops, segment-list
  trajectories) never enters pseudo-GT. The human review band is where the
  learning signal for hard cases comes from — it is structural, not
  transitional. (This is the low-rank-manifold guard restated.)
- **LOSO transfer risk at n=2 GT sets.** Probe precisions did not transfer
  (fp 0.90→0.53 LOSO). A learned critic can overfit Two Friends' mixing style
  the same way. Synthetic negatives mitigate, don't eliminate; LOSO eval is
  mandatory.
- **Never feed auto-accepted labels back into the critic itself unaudited** —
  that's the drift loop.

## First experiment (contained, one shot)

Span-level correctness classifier on `(mix window, ref window, hypothesis)`:

- **Inputs:** existing probe features (`ProbeFactor` records once W2 lands)
  + HuBERT/chroma embeddings.
- **Data:** GT positives + perturbation negatives + synthetic renders.
- **Bar:** beat the 0.75 precision-fusion AUC held-out, **LOSO across
  BB11/BB12**, abstention curve stays monotonic.
- **If it clears:** wire as the band-gate in the agentic loop; measure
  human-minutes-per-labeled-set before/after (W5 queue metric).
- **If it doesn't:** hand-built fusion stays; cost was one experiment.
