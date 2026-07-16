# Alignment — State of Record (current best + settled decisions)

> **As of 2026-07-16 @ `223fc68`** (branch `pws-alignment-reframe`).
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
grammar = "Turing-complete over DJ moves" (see §2, decision D1).

**Per-axis best current approach:**

- **Identity** — strongest axis. Stem-to-stem fingerprinting (mix-stem ↔
  ref-stem) is the current lever: it flips acappella and instrumental
  identification well above the full-mix baseline. HuBERT (L9) beats MFCC/chroma
  for vocal identity and is key/tempo-invariant (matters — a large fraction of
  acappellas are re-pitched to the bed key, which breaks chroma). Fingerprint
  localizer is landmark-constellation + offset-histogram, vote-gated as an
  override rather than a blind trust.
- **Placement** — the wall (roughly tied with structure as the weakest axis).
  Grid-lock (beat-grid snapping) is *the* placement lever; boundary-novelty
  (Foote) supplies a `set_start` prior; per-stem HuBERT `set_start` votes a
  diagonal. cue-detr cues are a *soft* ref-time prior, never a gate. Placement
  does **not** transfer across sets from co-training alone (see D-cotrain).
- **Structure** — segment-list decode. Piecewise-linear path decode (Viterbi
  over offset) handles loops / jumps / odd-ratio warps; lyric-anchor ref-decode
  helps acappellas but loses loops, so it is fused with abstention rather than
  used alone.

**Cross-cutting machinery (settled, in use):**
- **Scorer** scores over played + gain-audible intervals only (de-inflated; no
  credit for silent gaps), on the `_lt` (looptrace) timeline, `make scorecard`.
- **Fusion** requires an axis prior (`source_priority` = the identity axes
  order); raw cross-axis peak-merge is unsound.
- **Abstention** is driven by *margin*, not absolute cosine.
- **Fibers** (self-repeat classes) credit "right repeated content" via the
  fiber-aware scoring gate.

**Where the engine lives:** `workspaces/alignment_prototype/` (probes, decode,
scorer) + `workspaces/pws_aligner/` (the PWS/weak-supervision aligner, an agent:
LFs=sensors, label model=belief, actuation+oracle=effectors, abstain=policy).
Harness: `harness/`. Agentic loop: `agentic/`.

---

## 2. Settled decisions (append-only; status = SETTLED | SUPERSEDED-BY-#N)

> Append new entries at the top. Never rewrite history — supersede it.

**#12 — Transition / gradual-tempo regime is IN SCOPE (representation).**
`2026-07-16` · SETTLED (representation) / OPEN (recovery-by-Aug-1).
The output grammar must be able to express a continuous tempo ride (a moving warp
curve), because representation is a permanent commitment. Whether the aligner can
*recover* gradual tempo by Aug 1 is an empirical question the synthetic-transition
probe answers (§3) — don't decide in/out blind.

**#11 — Three data sources, three roles.** `2026-07-16` · SETTLED.
*Synthetic* (`.als → audio`, known labels): for MEASURE (alignment) only, never
JUDGE (taste); generate on-manifold from the low-rank knobs + move grammar.
*Download* (large, diverse, unlabeled real sets + constituents): the co-training
substrate / flywheel fuel — this is *why* 2 GT sets can suffice. *GT (n=2)*:
validation anchor + sim-to-real lie-detector (ace synthetic but flunk BB = the
off-manifold tax).

**#10 — GT is for VALIDATION, not training; base stays ~n=2.** `2026-07-16` ·
SETTLED. Hand-labeling ≈ 3 weeks per mashup set (BB11 `2nvzlh2k`, BB12
`1fsnxchk` done). At most one more real set, chosen for max distance from Big
Bootie (a transition set), via aligner-proposes→human-fixes. Supersedes the
bearings-era "highest-leverage move is grow GT beyond n=2" — the lever is the
unlabeled co-training substrate, not more GT.

**#9 — "Chosen right" = grammar coverage, NOT popularity.** `2026-07-16` ·
SETTLED. The ~1,016 downloaded sets are popularity-seeded → EDM/mashup-skewed.
Co-training expansion stratified-samples the DJ-move space, over-pulling
underrepresented moves.

**#8 — Measure vs Judge separation.** `2026-07-16` · SETTLED. The aligner
measures the *result* of taste (structure = the shadow of taste); it never models
taste. Synthetic needs only the skeleton; taste (flesh) comes from real data +
SoundCloud priors and lives in the deferred lab.

**#7 — Aligner target = "Turing-complete over DJ moves" grammar; open DAW tail =
abstain.** `2026-07-16` · SETTLED. Bounded DJ vocabulary (placement/identity/
timbre table below) is achievable and must be representable. The open
VST/automation tail is not chased for completeness — abstain there.
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
`2026-07-12` · SETTLED (finding). LOSO: identity transfers ~100% both directions;
placement is >75× unstable (the head memorizes placement per-set). n=2 cannot
disambiguate the mechanism — do not over-explain.

**#5 — Pitch: integer acappella offsets are deliberate harmonic key-match to the
bed, NOT varispeed.** `2026-07-12` · SETTLED. Varispeed hypothesis rejected
(H1 R²≈0.005). A detune-aware channel serves the aligner (explains chroma
failure on re-pitched acappellas). Vocal band 200–3500 Hz.

**#4 — GT export drops deactivated clips and is gain/audibility-aware.**
`2026-07-12` · SETTLED (code) / see §3 (not yet written back to pi canonical).
Deactivated-track clips and volume-silenced regions are phantom GT; the scorer
window is the gain-audible extent, not the clip extent.

**#3 — Scorer de-inflation.** `2026-07-12` · SETTLED. Score over played +
gain-audible intervals, report `gap_hallucination_frac`; genuine walls
(acappella oddratio, loop) unchanged. Metric got de-inflated — results did not
get worse.

**#2 — Alignment is not a scalar → three-axis decomposition.** `2026-07-12` ·
SETTLED (framing). identity / placement / structure, near-orthogonal, different
synthetic-vs-real difficulty and generalization. This is the paper's spine.

**#1 — Numbers SSOT.** `2026-07-11` · SETTLED. Every headline alignment number
lives only in [alignment_status.md](alignment_status.md), stamped + regenerated
from the scorers; other docs cite it. Dead ends live in the EXPERIMENTS ledger.

---

## 3. Open fronts (what's live / undecided right now)

- **Co-training corpus expansion (immediate next work).** Run the Tier-1
  metadata-proxy fingerprint over the scraped ~41k (`density`, `w/`-fraction,
  version/stem/ID tag fractions, cue-gap, styles), produce a grammar-coverage map,
  stratified-sample the underrepresented corners. Tier-2 audio-only moves
  (loops/jumps/tempo-ride/key-mix/fx) are invisible in scrape — revealed by
  probes *after* download → top up next round. Seed artists to check (DJ sets not
  live PA): Alesso (mashup corner), RUFUS DU SOL / ODESZA / Galantis (own
  material). Detail: [handoff_pws_cotrain_20260716.md](handoff_pws_cotrain_20260716.md) §4.
- **Synthetic-transition probe (decides Aug-1 transition scope).** (a) verify the
  `.als` round-trip survives a *continuous* tempo curve; (b) build the
  synthetic-transition generator (known curve → rendered audio + labels);
  (c) probe the aligner, report gradual-tempo recovery accuracy. That number
  picks the branch on decision #12.
- **Co-training seam wiring.** suspect-detector → `AcquisitionCase` producer;
  candidate-ref → align-to-mix → `TrainingSignal` + `track_audio_correction`
  ledger, GT-calibrated, ZERO canonical mutation. `bb_reacquire_queue.json` = 879
  fetch_missing-only, no executor wired yet.
- **PWS phase-1b (continuous) build-out.** Lives in worktree
  `~/Desktop/tracklist_engine-pws1b` on branch `pws-phase1b-continuous` (187
  tests). Not yet merged.
- **Pending canonical write-back (makes status numbers stale in a known-good
  direction).** The scorer de-inflation + GT-export fixes (#3, #4) need the "WS1
  step": re-export GT → pi write-back → rescore → regen `alignment_status.md`.
  Until then the improved traj numbers are *provisional*, not canonical.
- **Strategic fork (owner decision, unresolved):** ship the re-characterization
  paper now (defensible, n=2) vs. hold and fold into a SOTA-at-scale paper (needs
  the GT-scaling flywheel to run first).

---

## 4. Pointers (SSOTs — this doc defers to these)

| For… | Read | Rule |
|---|---|---|
| Headline numbers | [alignment_status.md](alignment_status.md) | Owns every number; regen, don't hand-edit |
| Dead ends / closed experiments | [attic/EXPERIMENTS.md](../workspaces/alignment_prototype/attic/EXPERIMENTS.md) | Read before re-trying anything |
| Interpretive frame (3 axes) | [alignment_recharacterization.md](alignment_recharacterization.md) | Never collapse to one scalar |
| Objective / round-trip contract | [alignment_objective.md](alignment_objective.md) · [architecture_north_star.md](architecture_north_star.md) | `make align SET=<id>` |
| Set ids | BB11 = `2nvzlh2k`, BB12 = `1fsnxchk` | Only two GT sets exist |
| Live ops handoff (box teardown etc.) | [handoff_pws_cotrain_20260716.md](handoff_pws_cotrain_20260716.md) | Operational, not strategic-state |
| Prior stock-take (superseded snapshot) | [alignment_bearings_20260712.md](alignment_bearings_20260712.md) | History; not current state |
