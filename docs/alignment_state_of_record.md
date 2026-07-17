# Alignment — State of Record (current best + settled decisions)

> **As of 2026-07-17 @ `da97c00`** (branch `cotrain-grammar-coverage`).
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
  used alone. The **learned trajectory decoder** (`trajectory/`) is the primary
  lever going forward, and **synthetic-mix augmentation of its training set is a
  validated win** (§2, D13): on held-out BB12 it beats real-only training and is
  markedly more stable across epochs. The single-global-stretch limit of
  `decode_path` (constant slope per span) is the transition-recovery wall — lever
  is stretch-in-state Viterbi (§3).

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

**#13 — Primary bet = synthetic-supervised learned placement/structure; PWS
demoted to the fusion layer.** `2026-07-17` · SETTLED (direction) / OPEN (does it
scale — §3). The bottleneck is **placement/structure**, not fusion: the PWS
continuous label model already *matches* hand-tuned fusion (Gate v3 PARTIAL), so
more fusion machinery is diminishing returns, while placement is ~91% of the
oracle→e2e gap. The lever is a **learned placement/structure model trained on
synthetic (manufactured-label) mixes, validated on BB** — synthetic is the only
way to get unlimited *labeled* data given n=2 GT (builds on #11 "synthetic for
MEASURE"). **First synthetic→real transfer read (trajectory decoder, held-out
BB12) → 🟢:** real+synthetic augmentation beats the real-only training ceiling and
is markedly more stable across epochs — synthetic *transfers and helps*. Qualified:
modest lift at only 100 synthetic mixes, this is *augmentation* not *pure-synthetic
substitution*, single direction (eval BB12), single seed, leakage check (BB tracks
in synthetic catalog?) still owed. PWS **v4** (singleton-σ fix — the diagnosed
lever if fusion is revisited) is demoted to a fallback. Spec:
`docs/superpowers/specs/2026-07-17-synthetic-transfer-spike-design.md`; provisional
numbers in the spike log, not yet in the status SSOT.

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

- **Synthetic→real transfer — FIRST READ DONE (🟢); #1 next = volume-scaling curve
  on Vast.** Trajectory decoder, held-out BB12: real+synthetic augmentation beats
  real-only training and is more stable (D13). Reuses the existing scaffold
  (`trajectory/train.py --synthetic-root`, `synthetic_adapter` train-only,
  `path_decode.trajectory_acc`, no-model control) on the 100 `data/synthetic_mixes_v2`
  windows. **Next, in order:** (1) leakage check — confirm the synthetic catalog
  excludes BB recordings; (2) matched-epoch + multi-seed rerun to beat single-epoch
  noise; (3) **Axis-1 volume curve** — scale synthetic generation + featurization
  on **Vast** (GPU-bound HuBERT) and plot held-out acc vs #mixes — the go-signal
  for the learned-aligner program; (4) pure-synthetic-only (train-only-synthetic)
  variant to measure the true transfer gap, not just augmentation. Spec:
  `docs/superpowers/specs/2026-07-17-synthetic-transfer-spike-design.md`.
  **Infra lessons (reusable):** MPS *hangs* trajectory training (run `--device cpu`);
  the Mac is a contended multi-agent box (parallel `race`/`infer` starve + kill
  runs); use `PYTHONUNBUFFERED=1` + `HF_HUB_OFFLINE=1` for observability; persist
  synthetic features (they cache to `.feat_cache`, but a cold pass is ~40 min).
- **Co-training harvest can run on the EXISTING downloaded corpus — NOT blocked on
  the 20k pull or the ingest agent.** To close the synthetic→real gap we need real
  (mix ↔ alignment) pairs; the seam manufactures them by running the probe ensemble
  on already-downloaded+analyzed sets and keeping confident agreements as
  pseudo-labels. It only *reads* analysis outputs → collision-free with ingest. The
  20k grammar-coverage pull demotes to a later *diversity/scale* lever, not a
  prerequisite. Gate on this: validate ACCEPT precision on BB GT first (bad
  pseudo-labels poison training); keep abstain-heavy. Blocked-adjacent: per-probe
  [0,1] calibration (below). Sizing step (read-only): inventory downloaded+analyzed
  sets on pi-storage.
- **Co-training corpus expansion — Tier-1 BUILT (branch `cotrain-grammar-coverage`).**
  `eda/alignment/generalization/grammar_coverage.py` fingerprints all ~41k by
  grammar proxies (w/-frac, version/stem/ID, density), maps downloaded-vs-corpus
  coverage, ranks fetchable candidates by starvation fill, with a **self-fraction
  live-PA filter** (own-material sets excluded). Biggest starved corner: 16k+
  remix-heavy-non-mashup sets ~1% covered. `download_from_coverage.py` emits an
  ingest job file (dry-run). Open: run the actual downloads (rides the ingest path
  — wants the parked ingest bug-fixes first); Tier-2 audio-only moves revealed
  post-download. Traps saved: [[project_corpus_artist_query_traps]] (dj_sets.artists
  empty, diacritics), [[project_grammar_coverage_selection]] (str.splitlines \x1e bug).
- **Synthetic-transition probe — BUILT + RUN → answer: gentle/medium IN scope,
  steep abstain.** `.als` round-trip survives a continuous tempo curve (§5a
  confirmed); `synthetic_mix/transition.py` (ride generator + exact varispeed
  render + LNDS path-proxy) + `transition_probe.py` + `eval_transitions.py`.
  Aggregate (12 rides × refs): gentle 0.06s / medium 0.04s median-of-medians
  (recoverable), steep 19.7s (fails). Keeps decision #12's transition regime IN
  Aug-1 scope for gentle/medium. `generate_transitions.py` emits ride training data.
  Full: `synthetic_mix/TRANSITION_PROBE_FINDINGS.md`, [[project_transition_recovery_finding]].
- **Co-training seam — dry-run skeleton BUILT.** `workspaces/pws_aligner/cotrain_seam.py`:
  suspect → `AcquisitionCase` producer; candidate-ref → align-to-mix → `TrainingSignal`
  + PROPOSED `track_audio_correction`. ACCEPT requires ≥2 independent channels
  agreeing (confirmation-drift guard); ZERO canonical mutation (tested).
  **`real_probe_scorer` now WIRED** — delegates to the same `capture_votes`
  harness-probe machinery (incl. the load-bearing ABSOLUTE→RELATIVE offset-frame
  normalization, single-sourced), gated to all-abstain when audio absent. Open:
  **per-probe [0,1] confidence calibration** (banding currently leans on offset
  *agreement* first, confidence second — needs a live BB-GT probe pass to
  calibrate); BB GT cases from `bb_reacquire_queue.json` (879 fetch_missing) + GT YAML.
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
