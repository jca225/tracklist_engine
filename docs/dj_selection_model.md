# DJ selection model — "why was this song chosen" (RESEARCH DIRECTION)

**Status:** research direction, not a committed architecture. Created 2026-06-11.
Reframes the information-dynamics work from *descriptive* (detect section
boundaries) to *normative/generative* (explain and predict track selection).

**Relationship to the chain:** this is the **emulation** objective, not the
aligner. The repo's north star is step 1 (align/clean GT); selection modeling is
a downstream objective (see the personalized-mix-generation end goal). We explore
it here because the *same* information-dynamics machinery already built for
boundary probing (`eda/alignment/`) is the natural substrate for a selection
model. Sibling docs: [aligner_attention_design.md](aligner_attention_design.md),
[alignment_objective.md](alignment_objective.md).

## The goal (restated)

We do **not** want better boundaries. We want a mathematical model that predicts
**why a track was chosen** at each point in a mix, and **generalizes** across
sets / DJs / listeners. The deliverable is an *interpretable, decomposable*
utility — "this acapella was picked ~70% for recognition, ~20% for
interestingness-vs-memory; the instrumental was a pure compatibility pick" — not
just a next-token predictor.

The information-dynamics readouts (Abdallah & Plumbley surprise / PIR) stop being
boundary detectors and become **terms in the DJ's implicit objective function**:
the DJ selects tracks to steer the audience's information-dynamic trajectory
through the Wundt "interestingness" band, given what the audience has already
heard.

## Mashup asymmetry — the instrumental anchors, the acapella is the payload

A mashup slot is **not** two symmetric tracks. The two identity roles play
different parts in selection, and the model must reflect that:

- **Instrumental = continuity anchor.** Its BPM never changes — it *sets* the mix
  tempo at that point; acapellas are time-stretched (and pitch-shifted) to sync
  to it. Chosen for *compatibility*, not fame (corpus empirics #6: 73% of BB
  instrumentals are obscure on every popularity metric — picked for fit). Low
  surprise by design.
- **Acapella = payload / hook.** Warped onto the instrumental's grid. Chosen for
  *audience payoff* — recognition, the vocal hook the crowd wants (corpus
  empirics #2: acapellas are 3× chart hits, ~200× the listeners of the
  instrumentals). This is where the chosen surprise lives.

Consequence: **tempo is a slowly-varying state set by the instrumental sequence,
not a per-track selection signal.** The model must not "explain" tempo as a
choice — it's a constraint the instrumental satisfies and the acapella conforms
to. Era proximity is also *not* a selection signal (corpus empirics #1: acapella
vs instrumental release-year independent, r≈0).

This motivates **two-stream information dynamics**:
- **Continuity stream** (instrumental / bed): smooth, anchors tempo+key, low
  surprise — models audience *expectation*.
- **Payload stream** (acapella / hook): where PIR / interestingness is measured,
  scored against **audience memory** of recently-played hooks.

## The model = attention over the catalog, biased by decaying audience memory

The user's two intuitions — "attention with exponential backoff on the prior
state" and "predict why a song was chosen" — are the same object:

- **Query** = current mix/audience state.
- **Keys / values** = candidate tracks in the catalog (retrieval, not a softmax
  over 20k — consistent with the aligner-attention design).
- **Temporal bias** = exponential decay = **audience memory** (a hook played
  recently is "fresh"; after memory decays you may reuse a motif or surprise
  while it's fresh). This is the "exponential backoff on the prior state" — the
  prior state is the *audience's decaying memory*, not a single previous bar.

### Sketch

State at slot *t*:
- `s_cont` = exponentially-weighted recent bed/tempo/key context (expectation).
- `s_mem`  = exponentially-weighted memory of recently-played hooks (recognition
  / fatigue), decay rate λ — the exponential-backoff term.

Choice = (instrumental *i*, acapella *a*). Utility:

```
U(i, a | state, listener) =
  + Recognition(a | listener)        # taste_prior + popularity priors
  + Interestingness(a | s_mem)       # A&P PIR — sweet spot vs audience memory
  − Repetition(a | s_mem)            # can't replay a fresh hook
  + Compatibility(i, a)              # key / energy mashability
  + Continuity(i | s_cont)           # key/energy fit (NOT tempo — anchored)
subject to:
  tempo(i) ≈ mix tempo anchor (fixed); a warped to i   # the BPM fact
  i mashable / present in catalog
```

Policy: `P(choice) ∝ exp(β · U)`. Fit β, λ, and the feature weighting on observed
BB sequences (BB11 `2nvzlh2k`, BB12 `1fsnxchk`, + more sets). Held-out test:
given mix state, **rank the true next track/pair against catalog negatives**
(reuses the `workspaces/alignment_prototype/` retrieval machinery).

### "Why" = decompose U

Because U is a sum of named terms, every observed choice decomposes into its
drivers. That decomposition *is* the deliverable. Generalization: fit across many
sets → parameters become a DJ/style prior; the `Recognition(·|listener)` term
personalizes per user (taste_prior cohorts).

## Open questions

- **Beatless acapella grid** — payload-stream info-dynamics needs a time base for
  the warped acapella; borrow the instrumental's grid (open Q from the variant-
  MERT note).
- **Memory decay λ** — is it fixed, or does it vary with set energy? Fit per set
  first.
- **Surprise of *what*** — VQ tokens (cheap) vs continuous MERT NLL (richer) vs
  hook-identity (recognition is about the *song*, not just the embedding).
- **Circularity** — never build the selection target from GT timing alone; the
  utility features must come from audio + catalog, GT supplies only the observed
  choice (the label).
- **This is emulation, not alignment** — keep it in `workspaces/` / `eda/`; do
  not let it claim aligner scope.
