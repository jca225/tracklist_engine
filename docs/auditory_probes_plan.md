# Auditory-neuroscience probes for the agentic aligner

**Status:** BUILT 2026-07-02 (`alignment/agentic/auditory.py`
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

## Validation result — onset_align (2026-07-02, `eval_auditory.py`)

**PARTIAL — real signal, not promotable as-is.** BB11, 25 regular/instrumental
spans, `onset_strength` xcorr of each full ref against the hour mix:

| formulation | <15s | <5s | median |Δ| |
|---|---|---|---|
| cold full-mix search | 24% | 16% | 171s |
| banded GT±90s (refiner ceiling) | 36% | 20% | 29s |

The exact hits are unambiguous — 002/007/015 at ≤0.1s, 010/011 at ≤5s — tracks
played STRAIGHT, where the onset grid matches perfectly. The misses are
edited/looped spans where the full-track onset envelope doesn't appear
contiguously in the mix; correlating the whole 3–4 min ref finds spurious
global matches. **Sharpness (peak/2nd) was flat (~1.0) everywhere and is
useless as an abstention gate; the winners instead had the highest correlation
PEAKS (015: 0.41, 002: 0.33) — peak magnitude is the usable confidence.**

Verdict: `onset_align` stays `validated=False`, precision calibrated to its
measured **0.36** (earned, honest). It is NOT wired into DOMINANCE. The
improvement path — untested, a real-work decision, not an auto-proceed:
1. correlate a short **excerpt** (the ~30–60 s the DJ actually plays), not the
   whole ref — but picking the excerpt is itself the ref_start problem;
2. **gate on peak magnitude** (≥~0.30), not sharpness, so the probe only speaks
   on the straight-played tracks it nails — turning a 36%-always probe into a
   high-precision-sometimes one (the abstention discipline the ladder wants).

`eval_auditory.py` is the reusable **validation gate** — bind a runner + rerun
it for every new probe before promotion.

## The learning harness (`learning.py`) — a probabilistic harness that learns

The hand-set precisions ARE the POMDP observation model, but hand-set — and the
real BB11 log proved several wrong. `learning.py` makes them **learned**: each
probe's precision is a **Beta(α,β) posterior**, seeded from the registry
(validated probes get confident priors, `validated=False` probes wide/explorable
ones) and updated from outcomes.

- **Calibration** — posterior mean feeds the ladder's quality gate.
- **Selection** — `rank()` orders candidate probes by **Thompson-sampled**
  precision-per-cost: exploration is free (uncertain probes occasionally sample
  high and get tried), exploitation from proven probes' tight posteriors. The
  harness thus **learns the DOMINANCE table** instead of us hand-authoring it,
  and the auditory probes **self-promote** as they accumulate correct fires.
- **Safe at n=2** (design doc tier 3): a conjugate Bayesian bandit *calibrates*;
  it never optimizes a learned reward, so there is nothing to hack — dodging the
  reward wall that killed synthetic-pretrain and the fusion model.
- **Learns from the audit trail** — `fit_from_events` replays the append-only
  event log, labels each observation via GT distance, updates posteriors.

**Real calibration (288 BB11 observations, 2026-07-02):** `cue_prior`
0.50→**0.71** (up — the scraped cue is better than guessed), `mert_decode`
0.55→**0.45** (down, correctly below cue_prior), `fp` 0.90→**0.70**, `lyrics`
0.90→**0.80** (still top). The hand-set optimism on fp/lyrics is measured away;
the ordering the harness learned matches every empirical finding. Auditory
probes stay at weak priors until wired to runners.

## Related

`docs/pomdp_agentic_aligner_design.md` (the harness these plug into),
[[agentic-harness-built]], [[project_low_rank_worldview]],
[[project_information_dynamics_bb12]] (surprise curve source),
[[project_reconstruction_supervision]] (cocktail-party = mix−bed).
