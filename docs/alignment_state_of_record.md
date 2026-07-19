# Alignment — State of Record (current best + settled decisions)

> **As of 2026-07-18 @ `8623eac`** (branch `trm-ablation-framework`).
> The current best spans unmerged branches: TRM/flywheel work is on
> `trm-ablation-framework`; the co-training/grammar/transition artifacts called
> out below remain on `cotrain-corpus-harvest` and must be reconciled before use
> from the TRM branch.
>
> **What this doc is.** The single *living* answer to "what is the aligner at its
> best right now, and what have we settled on — so build ON this, don't
> re-litigate it." A new alignment session reads this **first**; a finishing
> session updates it **last** (via `/align-checkpoint`) instead of spawning a new
> dated handoff. It is **undated on purpose** — it is rewritten in place, never
> snapshotted.
>
> **What it is NOT.** It does not restate headline numbers (those live only in
> [alignment_status.md](alignment_status.md)) or dead ends (those live in
> [workspaces/alignment_prototype/attic/EXPERIMENTS.md](../workspaces/alignment_prototype/attic/EXPERIMENTS.md)).
> It cites them. If this doc and an SSOT disagree, the SSOT wins and this doc is
> stale — re-run `/align-checkpoint`.

---

## 0. North star (the frame every decision serves)

**Operative (the gate):** a SOTA, rigorous alignment algorithm that generalizes
across **~40,000 DJ sets** at as close to 100% accuracy as possible. (Earlier
docs say "20,000" — superseded; ~40k is current.) **North-north (deferred behind
the gate):** the DJ-music research lab in `lab/`. Alignment is
necessary-but-not-sufficient for it.

**Interpretive frame (never collapse to one scalar):** alignment is three
near-orthogonal axes — **identity** (which recording), **placement** (where in
mix-time), **structure** (which internal spans, in what order). Read progress as
three curves. Full frame:
[alignment_recharacterization.md](alignment_recharacterization.md).

---

## 1. Current best solution — what to build ON

*(Qualitative pipeline shape only. All accuracy figures live in
[alignment_status.md](alignment_status.md) — cite it, don't copy numbers here.)*

**Objective / output.** The aligner consumes `{tokenized tracklist, track audios,
set audio}` and emits an Ableton-round-trippable structure (per-span
`ref_segments` for non-straight spans), trained on manual Ableton GT. Stem
discovery + version/variant QA are **ingest**, not the aligner. Target output
grammar = "Turing-complete over DJ moves" (see §2, decision #7).

**Per-axis best current approach:**

- **Identity** — strongest axis. Stem-to-stem fingerprinting (mix-stem ↔
  ref-stem) is the current lever: it flips acappella and instrumental
  identification well above the full-mix baseline. HuBERT (L9) beats MFCC/chroma
  for vocal identity and is key/tempo-invariant (matters because acappellas are
  often re-pitched to the bed key). Fingerprint localizer is
  landmark-constellation + offset-histogram, vote-gated as an override rather
  than blindly trusted.
- **Placement** — the wall (roughly tied with structure as the weakest axis).
  Grid-lock (beat-grid snapping) is *the* placement lever; boundary novelty
  (Foote) supplies a `set_start` prior; per-stem HuBERT `set_start` votes a
  diagonal. cue-detr cues are a soft ref-time prior, never a gate. Placement does
  **not** transfer across sets from co-training alone (see decision #6).
- **Structure** — segment-list decode. Piecewise-linear path decode (Viterbi
  over offset) handles loops / jumps / odd-ratio warps; lyric-anchor ref-decode
  helps acappellas but loses loops, so it is fused with abstention. The learned
  trajectory harness remains the primary placement/structure lane. The TRM
  recursive decoder is architecture-validated, but synthetic-only training does
  not transfer to real BB audio; the live data path is real, high-precision
  pseudo-labels, with synthetic realism as the alternative lever (decision #14).

**Cross-cutting machinery (settled, in use):**
- **Scorer** scores over played + gain-audible intervals only (de-inflated; no
  credit for silent gaps), on the `_lt` (looptrace) timeline, `make scorecard`.
- **Fusion** requires an axis prior (`source_priority` = the identity axes
  order); raw cross-axis peak-merge is unsound.
- **Abstention** is driven by *margin*, not absolute cosine.
- **Fibers** (self-repeat classes) credit "right repeated content" via the
  fiber-aware scoring gate.

**Where the engine lives:** `workspaces/alignment_prototype/` (probes, decode,
scorer, trajectory/TRM) + `workspaces/pws_aligner/` (weak-supervision fusion
and co-training seam). Harness: `harness/`. Agentic loop: `agentic/`.

---

## 2. Settled decisions (append-only; status = SETTLED | SUPERSEDED-BY-#N)

> Append new entries at the top. Never rewrite history — supersede it.

**#14 — TRM architecture is viable; synthetic-only substitution is refuted,
so the next data path is real pseudo-labels.** `2026-07-18` · SETTLED
(architecture + diagnosis) / OPEN (flywheel E1). On the strict
`path_decode.trajectory_acc` diagnostic, v0 overfit reached **0.95**, proving
the offset encoding, recursion, decode, and train loop can learn. But
synthetic-only→real stayed flat at **~0.09**, below the **0.306** raw control,
while synthetic train-fit rose: this is a measured sim2real gap, not
underfitting. More GPU on the same synthetic distribution is not the lever.
Use either synthetic realism or, first, the cheaper real pseudo-label flywheel;
run E1 on real AUTO_COMMIT spans before scaling. Full evidence and protocol:
[`trm_decoder_bakeoff.md`](../workspaces/alignment_prototype/docs/trm_decoder_bakeoff.md)
and [`trm_flywheel_design.md`](../workspaces/alignment_prototype/docs/trm_flywheel_design.md).
These are experiment diagnostics, not headline status metrics.

**#13 — Primary bet = learned placement/structure; PWS demoted to the fusion
layer.** `2026-07-17` · SETTLED (learned-decoder direction) /
SUPERSEDED-BY-#14 (synthetic-only data path). The bottleneck is
placement/structure, not fusion: the PWS continuous label model already matches
hand-tuned fusion, while placement dominates the oracle→e2e gap. Synthetic
augmentation had a positive first held-out read and passed its leakage check,
but #14's pure-synthetic diagnostic shows that augmentation is not evidence that
synthetic-only training transfers. PWS v4 (singleton-σ fix) remains a fallback.

**#12 — Transition / gradual-tempo regime is IN SCOPE (representation).**
`2026-07-16` · SETTLED (representation) / OPEN (recovery-by-Aug-1).
The output grammar must express a continuous tempo ride because representation
is a permanent commitment. Whether the aligner can recover gradual tempo by
Aug 1 is empirical; gentle/medium rides passed the synthetic probe, steep rides
remain an abstention regime.

**#11 — Three data sources, three roles.** `2026-07-16` · SETTLED.
*Synthetic* (`.als → audio`, known labels): for MEASURE (alignment) only, never
JUDGE (taste); generate on-manifold from the low-rank knobs + move grammar.
*Download* (large, diverse, unlabeled real sets + constituents): the co-training
substrate / flywheel fuel. *GT (n=2)*: validation anchor + sim-to-real
lie-detector.

**#10 — GT is for VALIDATION, not training; base stays ~n=2.** `2026-07-16` ·
SETTLED. Hand-labeling is too expensive to be the scaling path. At most one more
real set, chosen for max distance from Big Bootie, via
aligner-proposes→human-fixes.

**#9 — "Chosen right" = grammar coverage, NOT popularity.** `2026-07-16` ·
SETTLED. The downloaded corpus is popularity-seeded and EDM/mashup-skewed.
Co-training expansion stratified-samples the DJ-move space and over-pulls
underrepresented moves.

**#8 — Measure vs Judge separation.** `2026-07-16` · SETTLED.
The aligner measures the result of taste; it never models taste. Synthetic needs
only the skeleton. Taste lives in the deferred lab.

**#7 — Aligner target = "Turing-complete over DJ moves" grammar; open DAW tail =
abstain.** `2026-07-16` · SETTLED. Bounded DJ vocabulary is representable; the
open VST/automation tail is not chased for completeness.
```
PLACEMENT/TIME          IDENTITY/LAYER            TIMBRE/FX (fuzzy → abstain)
offset                  straight play             EQ / filter
constant warp           overlay / MASHUP          gain / sidechain
continuous warp (ride)  version (remix/edit/VIP)  echo / delay / reverb throws
loops                   unreleased / ID           gating
jumps / cuts / hot-cue
reverse / spinback
```

**#6 — Placement does NOT transfer across sets from co-training alone.**
`2026-07-12` · SETTLED (finding). Identity transfers across sets; placement is
unstable and the head memorizes per-set placement. n=2 cannot disambiguate the
mechanism — do not over-explain.

**#5 — Pitch: integer acappella offsets are deliberate harmonic key-match to the
bed, NOT varispeed.** `2026-07-12` · SETTLED. A detune-aware channel serves the
aligner; vocal band 200–3500 Hz.

**#4 — GT export drops deactivated clips and is gain/audibility-aware.**
`2026-07-12` · SETTLED (code) / pending canonical write-back.

**#3 — Scorer de-inflation.** `2026-07-12` · SETTLED. Score over played +
gain-audible intervals and report `gap_hallucination_frac`.

**#2 — Alignment is not a scalar → three-axis decomposition.** `2026-07-12` ·
SETTLED. Identity / placement / structure are near-orthogonal, with different
synthetic-vs-real difficulty and generalization.

**#1 — Numbers SSOT.** `2026-07-11` · SETTLED. Every headline alignment number
lives only in [alignment_status.md](alignment_status.md), stamped and regenerated
from scorers. Dead ends live in the EXPERIMENTS ledger.

---

## 3. Open fronts (what's live / undecided right now)

- **TRM real pseudo-label flywheel — E1 is next.** The architecture works but
  synthetic-only transfer fails (decision #14). Verify MERT + fingerprint caches
  for the pool, materialize only agentic `AUTO_COMMIT` spans from BB10 as
  pseudo-GT, train TRM on those real-distribution labels, and evaluate strictly
  on BB11 GT. No pi writes. The first gate is starvation: if too few spans
  survive, calibrate ACCEPT precision before building more machinery. Protocol:
  [`trm_flywheel_design.md`](../workspaces/alignment_prototype/docs/trm_flywheel_design.md).
- **Synthetic realism — alternative, not current first move.** Two measured
  mismatches are drop-from-top starts and regular full-track spans. The
  `bb12-real` curriculum addresses the first; the second is blocked on regular
  catalog diversity. Do not scale the old clean synthetic distribution.
- **Co-training harvest can run on the existing downloaded corpus.** It reads
  analysis outputs and manufactures high-confidence real pseudo-labels, so it
  does not require the larger grammar-coverage pull. Inventory on 2026-07-17:
  1,016 sets had downloaded mixes and ~19.6k reference audios, but mix-side
  analysis was barely run (`set_measures`=0 beat grids, `set_stems`=4 mixes,
  ~3.3k refs analyzed). The limiting prerequisite is a GPU-bound mix-side
  analysis pass, not downloads.
- **Co-training corpus expansion — Tier 1 built on
  `cotrain-corpus-harvest`, not yet reconciled.**
  `eda/alignment/generalization/grammar_coverage.py` maps downloaded-vs-corpus
  coverage and ranks fetchable starvation-fill candidates; the largest starved
  corner is remix-heavy, non-mashup sets. `download_from_coverage.py` emits a
  dry-run ingest job. Actual downloads are a later diversity/scale lever.
- **Synthetic-transition probe — answered for scope on
  `cotrain-corpus-harvest`, not yet reconciled.** Gentle/medium tempo rides
  are recoverable and remain in scope; steep rides are an abstention regime.
  Artifacts: `synthetic_mix/transition.py`, `transition_probe.py`,
  `eval_transitions.py`, and `TRANSITION_PROBE_FINDINGS.md`.
- **Co-training seam — dry-run skeleton built on
  `cotrain-corpus-harvest`, not yet reconciled.**
  `workspaces/pws_aligner/cotrain_seam.py` maps suspects to acquisition cases and
  accepted placements to training signals without canonical mutation.
  `real_probe_scorer` is wired to the shared `capture_votes` harness, including
  absolute→relative offset normalization, and abstains when audio is absent.
  Open: calibrate per-probe confidence on BB GT before accepting pseudo-labels.
- **PWS phase-1b continuous lane** remains unmerged and is a fusion fallback,
  not the primary placement/structure lever.
- **Pending canonical write-back.** The scorer de-inflation + GT-export fixes
  need re-export → pi write-back → rescore → regenerate
  [alignment_status.md](alignment_status.md). Until then, newer trajectory
  diagnostics are provisional.
- **Strategic fork (owner decision):** publish the re-characterization now or
  hold it for a SOTA-at-scale paper after the flywheel runs.

---

## 4. Pointers (SSOTs — this doc defers to these)

| For… | Read | Rule |
|---|---|---|
| Headline numbers | [alignment_status.md](alignment_status.md) | Owns every headline metric; regenerate, don't hand-edit |
| Dead ends / closed experiments | [attic/EXPERIMENTS.md](../workspaces/alignment_prototype/attic/EXPERIMENTS.md) | Read before re-trying anything |
| Current TRM verdict | [trm_decoder_bakeoff.md](../workspaces/alignment_prototype/docs/trm_decoder_bakeoff.md) | Architecture works; synthetic-only sim2real fails |
| Real pseudo-label next step | [trm_flywheel_design.md](../workspaces/alignment_prototype/docs/trm_flywheel_design.md) | E1 before scale |
| Interpretive frame | [alignment_recharacterization.md](alignment_recharacterization.md) | Never collapse to one scalar |
| Objective / round-trip contract | [alignment_objective.md](alignment_objective.md) · [architecture_north_star.md](architecture_north_star.md) | `make align SET=<id>` |
| Set ids | BB11 = `2nvzlh2k`, BB12 = `1fsnxchk` | Only two GT sets exist |
