# Alignment — State of Record (current best + settled decisions)

> **As of 2026-07-26 @ `b2b9e59`.** **RT1 honest baseline landed on `main`** (PR
> #105): the de-poisoned + form-centric scorecard, regenerated bridge id_maps, and
> human-verified GT completion are now canonical, and [alignment_status.md](alignment_status.md)
> carries the honest numbers (identity 51%/61% per-span vs the old poisoned 82/84).
> This is the honest-as-possible read of the **July `_lt` predictions**; the fully
> honest number still needs a **re-inference on clean canonical** (decision #18's
> remaining step). **Phase 1B identity capability = BUILT + MEASURED e2e, DEFERS on
> BB12 (does not ship)** (branch `phase1b-wire`, PR #109; extraction driver + gate
> added this session). The full override pipeline is now real and validated on GPU
> data: multi-candidate pool (Part A) + blind stem-MERT chamfer LF (Part B) + a
> gated `infer.py` override seam (Part C, **default-off / fail-closed**) + the L3
> stem-MERT extraction runner (`extract_stem_mert.py`) + a GT-anchored acceptance
> gate (`eval_1b_identity.py`). **First honest, GT-anchored acappella number: the
> blind LF does NOT beat the tokenizer claim on BB12** (net-negative — it fixes some
> wrong claims but breaks more correct ones; numbers in [alignment_status.md](alignment_status.md)).
> Per the spec's own rule, **blind-LF-alone defers** — awaits the co-train combiner
> (decision #23). **Load-bearing caveat:** the query windows were the broad
> parent-slot spans (acappella `w`-rows inherit the parent cue), which depress the
> LF (query-query cosine ≈ 0.99); a **tight GT-span re-measure** is the open front
> before concluding the approach's ceiling. Side finding: the tokenizer's acappella
> *claim* is itself wrong on a majority of slots (claim-vs-GT baseline).
>
> Prior settled context: decisions #20/#21/#22 — open-set identity works against a
> real candidate pool, the combiner **transfers** cross-set (BB11↔BB12 LOSO) but ≈
> borda, and **identity is stem-split** (vocal → chroma useless / MERT-L3;
> instrumental → chroma top-tier / MERT-L22; MERT's cross-set instability is
> vocal-specific). #18/#19 — the shipped "identity axis" measures the tokenizer
> claim, not the aligner (Phase 1B is the fix). **Operation Crush has EXITED**
> (decision #15) with GT de-poisoned + content-addressed on canonical; completeness
> Phase A shipped (#17, PR #75), Phases B–E remain. The current best still spans
> unmerged branches: TRM/flywheel on `trm-ablation-framework`; co-training/grammar/
> transition artifacts on `cotrain-corpus-harvest` — reconcile before use.
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
grammar = "Turing-complete over DJ moves" (see §2, decision #7). **The GT substrate
it validates against is now content-addressed and axis-complete** — every identity is
bound by audio content or honestly abstained, never a path/slot guess (decision #15), and
the bind now carries `stem`+`variant` through to the GT row so a right-work/wrong-stem or
wrong-variant label cannot slip (decision #17, Phase A).

**Per-axis best current approach:**

- **Identity** — strongest axis. Stem-to-stem fingerprinting (mix-stem ↔
  ref-stem) is the current lever: it flips acappella and instrumental
  identification well above the full-mix baseline. HuBERT (L9) beats MFCC/chroma
  for vocal identity and is key/tempo-invariant (matters because acappellas are
  often re-pitched to the bed key). Fingerprint localizer is
  landmark-constellation + offset-histogram, vote-gated as an override rather
  than blindly trusted. **Open-set demonstration (BB11 acappellas, no tracklist
  claim — the real-candidate-pool perception #19/#20 flagged as unwired):** against
  the set's full acappella pool, **MERT-chamfer at layer L3 (NOT the L6 default)**
  is the strongest identity LF; fused (borda) with a *pitch-normalized* fingerprint
  it reaches the labeling-function ceiling. Two load-bearing details: the transpose
  is a hard wall for frequency-absolute fingerprints (undo the **labeled semitone**,
  which is decoupled from `tempo_ratio` — do not correct tempo as if it were pitch),
  and long spans need a top-k chamfer (layered-vocal contamination). Weak LFs
  (chroma/DTW) *hurt* equal-weight fusion **on acappellas**. **Route by stem
  (decision #22):** the *instrumental* chain flips this — chroma/fp/dtw are all
  strong (chroma 15%→88%), the best MERT layer moves L3→**L22**, and MERT
  generalizes cross-set (89→89 vs the vocal 92→68). **Two-way stem routing (final,
  no third regime):** acappella spans → the mix *vocal* stem + low-layer MERT (L3);
  *everything else — regular full-song beds AND instrumental overlays* → the mix
  *instrumental* stem + chroma/fp + high-layer MERT (L22). Full-vocal-song beds
  identify by their instrumental backbone, not their vocal, because in a mashup the
  mix vocal stem carries the *overlaid* acappellas, not the bed's own vocal (65/70
  via instr handle, 0 vocal-only). Full method + numbers:
  [`open_set_acappella_identity_findings.md`](../workspaces/alignment_prototype/open_set_acappella_identity_findings.md).
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

**#23 — Phase 1B blind-LF acappella identity override, measured e2e on BB12,
DEFERS — does not ship.** `2026-07-26` · SETTLED (verdict) / OPEN (tight-window
re-measure). The full override pipeline (multi-candidate pool + blind stem-MERT
L3 chamfer + gated `infer.py` seam + extraction runner + GT-anchored gate) is
built and validated on real GPU data. Against clean GT on BB12 acappella slots,
the blind LF **does not beat the tokenizer claim** — net-negative (fixes some
wrong claims, breaks more correct ones). Per the Phase 1B spec's own acceptance
rule ("a capability that regresses does not ship — it waits for the co-train
combiner"), blind-LF-alone **defers**; the override seam stays default-off /
fail-closed. **Caveat that gates the verdict's strength:** query windows were the
broad parent-slot spans (acappella `w`-rows carry cue 0 and inherit the parent
slot's cue — fixed via `resolve_parent_cues`, but still one window per parent,
~127 s), which contaminates the vocal query (query-query cosine ≈ 0.99). The
earlier open-set finding (#20) measured ~68% for this LF with tighter windows, so
32% is the *coarse-window floor*, not the ceiling — **a tight GT-span re-measure
(and saving the full mix-L3 so windows are re-tunable without re-renting) is the
open front** before calling the approach dead. Instrumental identity is untested
here (extraction was acappella-only, the contested axis). Numbers →
[alignment_status.md](alignment_status.md).

**#22 — Identity is stem-split: the LF that works AND the best MERT layer both flip
between vocal and instrumental; MERT's cross-set instability is vocal-specific.**
`2026-07-24` · SETTLED (finding). Ran the open-set identity pipeline on the
**non-acappella** chain (mix instrumental stem vs the full drops + instrumental
overlays; BB11 56 spans, BB12 63). Confirms the CLAUDE.md axis rule empirically:
**chroma is useless on acappellas (15%) but top-tier on instrumentals (88%)**, and
dtw 46→89% — instrumentals are harmonic/percussive so fp/chroma/dtw/MERT are *all*
strong (no single blind spot, unlike acappella's lone MERT). Two consequences: (a)
**the best MERT layer flips by stem — L3 for vocal identity, L22 for instrumental**
(low acoustic-timbre vs high abstract-harmonic) — so per-stem layer selection is
required; validates the adapter keeping all 25 layers. (b) **MERT is stem-dependent
but not universally unstable**: it was 92→68 cross-set on acappellas (#21) yet
**89→89 on instrumentals** → the generalization gap is *vocal contamination /
separation quality*, not MERT. Instrumental identity is also more robust: oracle-of-5
ceiling 94–95% on *both* sets (vs acappella BB12's 88%). Combiner still transfers ≈
borda (89% both directions). Implication for the aligner's real candidate pool
(#19/#20): route by stem — instrumental spans lean chroma/fp + high-layer MERT, vocal
spans lean low-layer MERT. Numbers + method:
[`open_set_acappella_identity_findings.md`](../workspaces/alignment_prototype/open_set_acappella_identity_findings.md)
(§ Instrumental chain). Refines #20/#21.

**#21 — The open-set identity combiner TRANSFERS cross-set (BB11↔BB12 LOSO), but
≈ borda, not ≫ it; the ceiling is the sensors/references, not the combiner —
and MERT-L3 does NOT generalize as #20 implied.** `2026-07-24` · SETTLED
(finding). Ran BB12's 101 acappella spans through the same 5-LF pipeline and
trained a per-candidate label-model (class-balanced LR, set-agnostic features) on
one set, tested on the other. **The combiner transfers** — trained on the *other*
set it beats every single LF on both held-out sets (BB11→BB12 85% vs best-LF 77%;
BB12→BB11 95% vs 92%) and does not collapse. **But it only ties/edges borda** (85
vs 83; 95 vs 96), so *learning* the combiner buys robustness, not a big jump — this
**refines #20's "co-training closes 92→96"**: with L3-MERT, borda already sits at
the ceiling. **The real limit is the LF/reference ceiling:** BB12's oracle-of-5 is
only ~88% (12% of spans no LF reaches) — sensor/reference quality caps harder sets.
**Caveat on #20:** MERT-L3 identity is **set-dependent, not universal** — it drops
sharply BB11→BB12 and on BB12 the pitch-fingerprint *beats* it; LF dominance flips
across sets and the combiner's reweighting is what absorbs it. Open question: why
BB12 is harder (contamination / separation quality / ref-pool). Numbers + method:
[`open_set_acappella_identity_findings.md`](../workspaces/alignment_prototype/open_set_acappella_identity_findings.md)
(§ Cross-set LOSO). Refines #20.

**#20 — Open-set acappella identity works against a real candidate pool: MERT
layer L3 (not L6) is the identity sensor, and the combiner — not new probes — is
the lever past the LF ceiling.** `2026-07-24` · SETTLED (finding). Directly
exercises #19's fix-path-1 (real multi-candidate pool + audio perception overriding
the tokenizer claim): on BB11's 91 acappella spans vs the set's ~89-ref pool with
**no tracklist order/cue**, stem-to-stem MERT-chamfer identifies the played
acappella well, and fused with a pitch-normalized fingerprint reaches the
oracle-of-LFs ceiling. So **perception exists and is strong** — the gap is pool
*wiring*, not sensing. Sub-findings, all settled: (a) **MERT L3 ≫ L6 for vocal
identity** (25-layer sweep, monotonic falloff after ~L8) — validates the adapter
keeping all layers; L6 stays the default only where it hasn't been re-probed. (b)
**Pitch ⟂ tempo** — Ableton warp is pitch-preserved time-stretch + a *separate*
±1-semitone transpose; correct the labeled semitone (not `tempo_ratio`) to unbrick
frequency-absolute fingerprints. (c) **top-k chamfer** beats mean on long spans
(layered-vocal contamination), mean elsewhere. (d) **Weak LFs hurt** equal-weight
fusion; prune to the strong two. (e) The residual failures are combiner losses (a
strong LF already has rank-0) or reference-quality (derived `vocals.flac` /
annotator-flagged bad refs) — **not** sensor limits; hand-tuned gating overfits at
n=91, so the honest lever is a **learned cross-set combiner (co-train)**, which the
LOSO precedent (#6-adjacent) says transfers for identity. Numbers + method:
[`open_set_acappella_identity_findings.md`](../workspaces/alignment_prototype/open_set_acappella_identity_findings.md).
Related: [[project_trm_alignment_core]], [[project_identity_by_string_bug_class]],
[[project_stem_cand_wrong_recording_gap]].

**#19 — The aligner does NOT predict identity — it inherits the tokenizer's claim
100% by construction; the "identity axis" measures the SPINE, not the aligner.**
`2026-07-23` · SETTLED (finding) / OPEN (fix). Proven this session: the emitted
`recording_id` equals `set_track_slots.recording_id` on **every** span — 157/157 on a
fresh clean re-inference, 152/152 on the July-6 full pipeline. Root cause:
`slot_candidates_from_targets` builds a **"naive candidate pool: one recording per
slot = the tokenizer's claim"** ([dataset.py:43]), so `predict_sequence` (mert_model.py)
has a pool of size 1 per slot — it can only "select" the claim; it decides placement,
never identity. `infer.py:124` reads `recording_id` straight from `set_track_slots`.
**Consequence:** the "identity" headline (84% vs poisoned GT, ~62% vs clean GT) is the
tokenizer's *tracklist-claim accuracy*, not an aligner capability. Crush de-poisoned
`set_ground_truth` (GT) but **never touched `set_track_slots` (the claim spine)**, so the
poison relocated to the prediction side — invisible until pred-vs-spine was checked. The
status.md §5 "MERT identity 83–84%" is an **eval-only** capability (real multi-candidate
pool) that is **NOT wired into the shipped pipeline**. **Fix paths:** (1) build a real
per-slot candidate pool (claim + siblings/alternatives, or corpus-wide) and wire
MERT/fingerprint/stem-FP to override the claim when audio disagrees — perception exists
(§5), the pool wiring does not; or (2) fix the spine upstream in ingest (version/variant
QA), per CLAUDE.md's "identity QA is ingest, not the aligner". Nothing currently corrects
it. Supersedes the hopeful read in #18 (the identity deflation is NOT an artifact — it is
the tokenizer claim measured honestly). Related: [[project_stem_cand_wrong_recording_gap]],
[[project_wrong_version_preview_clip]], [[project_identity_by_string_bug_class]]. Experiment
in progress: real-candidate-pool MERT re-pick on BB12.

**#18 — The alignment_status.md numbers are measured on STALE APPARATUS, not
clean GT — do NOT trust them as post-Crush honest numbers.** `2026-07-23` ·
SETTLED (finding) / OPEN (honest re-measure). Attempted the post-Crush honest
re-measure (RT1 starting gun) and found the scoreboard is incoherent with the
de-poisoned GT — the *apparatus*, not the GT, is the problem. Verified on
canonical pi: `set_ground_truth` is genuinely de-poisoned and **drift-free vs the
git fixtures** — abstains stored as NULL `recording_id` (BB12 57/110, BB11 64/84),
counts exact (315 rows), per-label binding diff = **0 drift**; pi code current
with `origin/main` (the "~92 behind" note is stale — autopull caught up). BUT the
predicted timelines (`workspaces/alignment_prototype/out/*_predicted_timeline_lt.json`)
are **from July 6**, generated when the aligner drew candidate identity from the
**unverified claim spine** (`set_track_slots`) — which Crush never touched — and the
**bridge id_maps** (`labeling/fixtures/id_maps/<set>.json`, scrape-id→canonical
recording_id) are **broken: BB12's missing, BB11's from July 2**; their generator
was deleted, so the map silently rotted. Re-scoring stale predictions through a
broken translation table against fresh GT makes the **identity axis look like it
collapsed 82–84%→51–61%** — this is a MEASUREMENT ARTIFACT, not a regression
(placement/structure moved only ±3pp because de-poison never touched them). **The
honest re-measure requires re-inference on clean canonical + a rebuilt bridge
id_map, NOT re-scoring the July-6 files.** Deeper design flag (defer): identity
candidates are bounded by the claim spine, so the aligner cannot identify a
recording the spine never claims. Numbers → alignment_status.md after re-inference.

**#17 — Crush completeness Phase A SHIPPED: sound correctness + axis plumbing.**
`2026-07-22` · SETTLED (Phase A) / OPEN (Phases B–E). First execution of the #16 rule set,
local-only (no pi/DB writes, no new data), on **PR #75** (`gt-binding-completeness`, 5
commits, subagent-driven + opus whole-branch review). A1 = ambiguity hard-abstain keyed on
`(recording_id, stem, variant)` at both `_load_content_catalog` and `from_entries` (an
ambiguous hash abstains, never last-writer-wins); A2 = `stem`+`variant` plumbed
catalog→bind→`GroundTruthTrack.claimed_variant`, sourced from the content bind when bound;
A3 = catalog stem-derivation correctness (`ta.stem='regular'` parent filter, strict
stem-name map, `kind` master/separated); A4 = catalog scope = true `pull_set_for_alignment`
parity (`COALESCE(recording_id,track_id)` + dual-key match); fix wave = separated stems
inherit the parent's `variant` (a real per-axis-soundness bug the whole-branch review
caught). `tests/labeling/` green, gt_als_gate unchanged. **Next = Phase B** (content-history
hash ledger + FLAC-PCM key) — the sound completeness lever, **gated on a coordinated pi
deploy** (pi ~92 behind, issue #73; do not force-deploy). Residual: the catalog still omits
the pull's `dj_set_track_media_links` UNION, so sided/`tlp` slots stay catalog-invisible
(safe — they abstain; candidate Phase-A′ or fold into Phase C prov). Numbers →
alignment_status.md after Phase E re-export.

**#16 — Crush completeness rule set (the final exit): sound multi-channel binding,
certificate-gated.** `2026-07-22` · SETTLED (rules) / SHIPPING (Phase A done — #17; B–E open). Crush's
soundness exit (#15) left ~40% of GT clips abstaining — honest, but recoverable. The
recoverable set is *identity-preserving churn* (retry / re-separate / retag: same song,
new bytes) whose old hashes were discarded. Rule set (twice Fable-reviewed): identity
is a **per-axis** product `Work×Version×Stem×Variant×Remixer` (soundness must hold on
every axis); **bind across a generation boundary only with a CERTIFICATE** (payload-hash
equality / derivation+parent-hash / perceptual+duration), **never** by op name
(retry/rescue is not identity-preserving — the wrong-version-from-preview-clip class);
`relink`/`detach`/re-selection **tombstone** prior generations; a content-history hash
ledger keyed `(recording_id, stem, variant, kind)` + FLAC-PCM/mdat payload keys are the
*sound* completeness lever (recover the churn abstains as byte-exact binds, no new trust
assumption); fuzzy is axis-lossy → rival-relative per-axis gate, ε-sound, excluded from
write-back. Lifts BB12 66%→~82% with **zero** new wrong labels. Spec:
[gt-identity-binding-completeness-design](superpowers/specs/2026-07-22-gt-identity-binding-completeness-design.md)
(v1–v4); plan: [gt-binding-completeness-plan](superpowers/plans/2026-07-22-gt-binding-completeness-plan.md)
(phases A–E). This is the definitive Crush closure; numbers → alignment_status.md after re-export.

**#15 — Operation Crush EXITED: GT is de-poisoned and content-addressed on
canonical.** `2026-07-22` · SETTLED. The `slot_id_map` path/slot-guess binding is
dead. BB11 (`2nvzlh2k`) + BB12 (`1fsnxchk`) were re-pulled → re-exported via the
content-addressed `export_als_to_gt` → `write_back_ground_truth` applied to
canonical `set_ground_truth` + read-back verified. **Every identity binding is now
`id_source ∈ {content, abstain}` — zero path/slot guesses survive** (BB12: 110
content / 57 abstain; BB11: 84 / 64). Slots 028=Beatles, 031=CCR bind
correct-by-content; 144 & 148w1 abstain (148w1 = the `track_audio_id 20911`
wrong-recording fix shipped in PR #72's same-song guard). Poison ids at 028/031/144
= 0. Canonical backup: `pi:~/crush_exit_backups/set_ground_truth_bb11_bb12_*.sql`
(315 rows). This closes decision #4's pending write-back and is the precondition for
Rolling Thunder RT1. (Numbers themselves → `alignment_status.md` after the scorer
re-run.) Related infra filed: PR #72 merged; issues #73 (deploy discipline) + #74
(path-mojibake locale root cause).

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
`2026-07-12` · SETTLED (code) / canonical write-back DONE `2026-07-22` (see #15).

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

- **Phase 1B tight-window re-measure — the immediate open 1B front (decision #23).**
  The e2e gate ran on broad parent-slot windows and the blind LF underperformed the
  claim; the coarse window is the prime suspect (query-query cosine ≈ 0.99, vs the
  #20 finding's ~68% with tighter windows). Next: (1) re-extract BB12 acappella with
  **tight GT-span windows**, and **save the full mix-vocal L3** so windows are
  re-tunable locally without re-renting; (2) extract **BB11** (staged: stems/cache/GT
  present, rows JSON built) so the **full LOSO gate** (identity ≥ RT1 on both sets,
  τ/floor tuned on one and validated on the other) can run; (3) then the ship/defer
  verdict is final. Pipeline built on `phase1b-wire` (PR #109): `extract_stem_mert.py`
  (runner + `resolve_parent_cues`), `eval_1b_identity.py` (GT time-join gate),
  `candidate_pool.py` / `identity_override.py` (default-off seam). GPU driver:
  `scripts/gpubox_extract_1b.py` (needs a `--set-id` param before it lands). This is
  a windowing/reference front, **not** a compute or architecture one.
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
- **Open-set acappella identity — LOSO measured (decision #21); now
  sensor/reference-bound, not combiner-bound.** The combiner transfers cross-set
  (BB11↔BB12) but only ties borda, and BB12's LF ceiling is ~88% — so the live
  levers are no longer the combiner. Priorities: (1) **why MERT-L3 drops 92→68
  cross-set** (contamination / RoFormer separation quality / ref-pool diversity) —
  the biggest generalization gap; (2) **reference quality** — derived `vocals.flac`
  refs are confusable attractors (ingest, not aligner); (3) a *deployable* pitch
  estimate (key/BPM detection) to replace the labeled-semitone oracle in the fp LF;
  (4) longer-context / robust-aggregation MERT for the long-span contamination
  cases. Code in the `bb11_acap_id` / `bb12_acap_id` scratchpads, promotable to
  `evals/`.
- **Crush completeness Phase B — the immediate next build (decision #17 → #16).**
  Content-history hash ledger keyed `(recording_id, stem, variant, kind)` + never-drop-hash
  acquisition hooks + certificate-gated bind-across + FLAC-PCM/mdat payload keys + relink/
  detach tombstones (plan tasks B1–B4). This is the *sound* completeness lever (recovers the
  ~40% churn abstains as byte-exact binds, lifts BB12 66%→~80%+ with zero new wrong labels).
  **GATE: Phase B touches canonical pi**, which is ~92 commits behind with live parallel WIP
  (issue #73) — a real `make deploy` is a **separately coordinated task; do not force-deploy**
  (Phase A stayed local to avoid exactly this). Then re-export → GATE write-back → the
  re-measure below.
- **Audio coverage reality (census 2026-07-22).** For the two GT sets, **every GT row is
  backed by present audio**: all 543 BB11+BB12 referenced files are on disk (8 show "missing"
  only via the issue-#74 mojibake path bug — files present under correct UTF-8). Six truly
  audio-less slots exist (4 are `tlp` sided-rows) but **carry zero `set_ground_truth` rows** —
  un-aligned tracklist spine, never enter the binder, not a GT gap. At corpus scale only
  **~8.6%** (18,805/218,467 distinct tracks) have audio and `is_reference` is essentially
  unpopulated (443 rows) — the ingest frontier, not gating alignment. Do NOT read the 8.6% as
  an aligner problem; it's acquisition.
- **Post-Crush re-measure — RT1 apparatus LANDED on `main` (PR #105); one step
  remains (decision #18).** The de-poisoned + **form-centric** scorecard,
  regenerated bridge id_maps (`labeling/fixtures/id_maps/<set>.json`, both sets),
  and **human-verified GT completion** (spectrogram review) are now on `main`, and
  [alignment_status.md](alignment_status.md) carries the honest numbers (identity
  **51%/61%** per-span, form-centric, audible-weighted — not comparable to the old
  poisoned 82/84). **Caveat:** these score the **July `_lt` predictions** honestly
  against corrected GT; the *fully* honest number still needs the **re-inference on
  clean canonical** (re-infer both sets on a **gpubox** GPU — BB12 first, BB11
  stalls Whisper — validating the id_map via ref-resolution count, then `make
  scorecard` / `make race` + TRM `path_decode.trajectory_acc` → regen status.md).
  That re-inference is the remaining RT1 step. **Recovery note:** the RT1 work was
  recovered from the abandoned `fix/rt1-form-centric-remeasure` branch, whose tip
  commit (`43d6f0e`, mislabeled a guardrails bump) deleted ~1,590 files; the poison
  branch was deleted from origin after the 6 good commits were cherry-picked into
  #105. See [operation_rolling_thunder_proposed.md](operation_rolling_thunder_proposed.md).
- **Phase 1B identity capability — pure cores BUILT, not yet wired (branch
  `provenance-engine-phase1`).** The capability half of decision #19's fix-path-1:
  replace the shipped **size-1 candidate pool** (which forces the aligner to inherit
  the tokenizer claim) with a real multi-candidate pool + a **fail-closed** audio-
  perception override. Done + unit-tested this session: `open_set_identity.py` (blind
  stem-routed MERT-chamfer identity LF — no tracklist prior, no oracle pitch;
  conditional top-k + borda + margin gate) and `stem_mert.py` (stem-domain per-layer
  MERT extraction core). Spec: `docs/engine/phase1b_identity_capability_spec.md`
  (on branch `provenance-engine-phase1` until it merges).
  **Remaining (in order):** (A) `candidate_pool.py` (variant-aware {claim} ∪ stem-ref
  pool); (B) the MERT **data dependency** — a variant-aware L3/L22 stem-MERT extraction
  runner + a GPU pull (deferred this session; `export_mert_from_pi.py` already takes a
  layer arg but exports full-mix/full-track, not stems); (C) the `infer.py` gated
  override seam (touches the live inference path — do LAST). Acceptance gate:
  identity ≥ the RT1 baseline on **both** sets (do-no-harm), numbers → status.md only
  after. Grounded in decisions #19/#20/#21/#22.
- **`labeling/` package reorganization — PARKED by owner (2026-07-22, "do later").** Raised
  after Phase A; assessment: the recurring bug class is SSOT drift (A4 = catalog vs pull
  reimplementing slot→audio resolution), not disorganization broadly. Pilot when resumed =
  extract the ONE shared slot→audio resolver + functional split of the two ~930-line files
  (`export_als_to_gt.py`, `pull_set_for_alignment.py`) + centralize/document tuning constants
  + excise the `BLINK_182_SLOTS` set-specific hack. **NOT** a class rewrite (house style is
  functional); **NOT** whole-repo (module renames break pi systemd entrypoints).
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
