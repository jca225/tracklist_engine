# Auditory-neuroscience probes for the agentic aligner

**Status:** BUILT 2026-07-02 (`workspaces/alignment_prototype/agentic/auditory.py`
+ belief-layer gates). DSP verified on synthetic signals (8 tests);
**precisions PROVISIONAL — unvalidated until a GT eval pass.**

## The organizing insight (from "Dive into Claude Code", Liu et al. 2026)

That paper reverse-engineers a mature agent and finds the load-bearing ratio:
**~98.4% of the system is deterministic operational harness; ~1.6% is LLM
decision logic.** The harness (permission gates, tool routing, context
management, recovery) creates the conditions under which a thin model layer can
decide well — "the harness enforces, the model reasons."

Our aligner is the same shape, and this is the design target:

| Claude Code | Our aligner |
|---|---|
| deterministic harness (98.4%) | the **probe set** — cheap, calibrated DSP/feature extractors |
| LLM decision logic (1.6%) | **`llm_policy.py`** — Opus adjudicates only the ambiguous residual |
| tool `tool_use` protocol | `Observation` (placement + calibrated precision) |
| permission ladder | `policy.py` autonomy ladder (auto/review/suggest/escalate) |
| append-only session log | `events.py` audit trail |

So: **invest in probes (the harness), keep the model layer thin.** Every probe
added here is deterministic harness the LLM never has to reason inside of — it
only weighs the probes' outputs. This is exactly the paper's bet that
"investing in deterministic infrastructure … yields greater reliability gains
than adding planning scaffolding around increasingly capable models."

## Why auditory neuroscience (the low-rank lens)

The brain recovers *which sources, and where* from one entangled pressure wave —
the same problem as mix→ref alignment — using a small set of irreducible
primitives ([[project_low_rank_worldview]]). Our current probes
(chroma/HuBERT/fp/lyrics) are almost all **identity-and-match** operators. The
ear's primitives are mostly **grouping-and-segregation** operators: *how many
sources, and when did each start*. That is the **missing rank** of the placement
problem — and it's cheap DSP, not new models.

## What was built

**Probes (`auditory.py`) — placement/identity observations:**

| Probe | Neural correlate | What it computes | Fixes |
|---|---|---|---|
| `onset_align` | cochlear-nucleus **onset cells** (fire only at energy onsets) | spectral-flux onset envelope of mix vs ref, cross-correlated → lag + peak + sharpness | placement is fundamentally onset-alignment; a DJ's beatmatch preserves the onset grid. **Flagship.** |
| `onset_async` | **onset asynchrony** — the strongest source-split cue (>~30 ms ⇒ segregate) | fraction of onsets whose low/high bands disagree in time | source-count: "is a 2nd (overlay) onset stream here" — the acappella-over-bed signal the arbiter now infers |
| `harmonic_sieve` | **periodicity coding** (harmonic sieve; partials at integer×f0) | per-frame f0 salience (HPS), pitch-invariant, matched by cosine | identity where chroma breaks — the 31% re-pitched acappellas |
| `modulation` | inferior-colliculus **modulation-rate tuning** | envelope modulation spectrum → dominant rate; ref/mix → stretch | a first-class tempo-ratio/stretch estimate |

**Belief-layer mechanisms (`belief.py`):**

- `masking_gate` — **precedence effect / temporal masking**: an observation
  placing a quiet overlay inside a louder source's shadow is down-weighted
  (precision × floor) before it votes. The LLM shouldn't trust a vocal-under-drop
  placement on thin evidence.
- `surprise_prior` — **predictive coding**: peaks in a boundary-novelty/surprise
  curve (our info-dynamics work, [[project_information_dynamics_bb12]]) become
  weak, **order-independent** placement candidates — "a new track entered here",
  needing no tracklist order.

**Belief-shaping action (`actions.py`):**

- `cocktail_party` — **old-plus-new** (auditory scene analysis's core rule):
  given a committed bed placement, subtract the predicted-old and re-probe the
  residual for the overlay. The reconstruction-supervision idea
  ([[project_reconstruction_supervision]]) as an explicit harness action.

## Safety rail: `validated=False`

All new probes register with `ActionSpec.validated=False` and provisional
precision ≤ 0.65. The autonomy ladder auto-commits only at quality ≥ 0.75
(= share × best-in-cluster precision), so **an unvalidated probe can never
auto-commit on its own** — only when a *validated* probe (fp 0.90, lyrics 0.90)
agrees in its cluster does the cluster's best-precision clear the bar. The math
enforces the discipline; no coupling needed. They are also **absent from
`DOMINANCE`**, so the loop won't spend calls on them until promoted.

## The validation gate (next step — do NOT skip)

Before any of these is trusted (dominance entry + `validated=True`), measure
P(correct | fired) on BB11+BB12 GT, per stem:
1. Bind a runner per probe (compute the DSP on the set's aligning audio).
2. Score each probe's proposal vs GT set_start (the `score_timeline_vs_gt`
   convention) → measured precision replaces the provisional guess.
3. Promote only probes that beat their stem's incumbent on the residual the
   incumbents miss (acappella trajectory, re-pitched identity). Per
   [[feedback_small_sample_regressions]], require n large enough to matter.

Expected first win: `onset_align` on regular/instrumental placement (cheap,
and the one probe that is directly a placement estimator).

## Related

`docs/pomdp_agentic_aligner_design.md` (the harness these plug into),
[[agentic-harness-built]], [[project_low_rank_worldview]],
[[project_information_dynamics_bb12]] (surprise curve source),
[[project_reconstruction_supervision]] (cocktail-party = mix−bed).
