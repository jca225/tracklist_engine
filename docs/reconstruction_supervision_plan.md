# Reconstruction-supervision plan — from 5 GT sets to 20k

**Status:** Step 1 DONE (2026-07-01) → **CONDITIONAL GREEN**. Steps 2–5 proceed,
with one required metric change (below). Raw numbers: `out/recon_probe_bb12.txt`.

## Step 1 result (2026-07-01)

Probe: `recon_probe.py` (per-frame cosine of L2-normalized mel-magnitude; stem-routed).

- **Test A (validity), regular tracks: STRONG.** peak-at-true-placement = **79%**
  vs 8% chance (n=38), median margin +0.055; the score-vs-offset curve peaks
  cleanly at 0 (true 0.742 vs perturbed ~0.60–0.645). The mix audio *does*
  localize regular-track placement with no human GT. Thesis validated for the
  majority stem class.
- **Test B (usability), regular: AUC 0.648** (median match correct 0.703 vs wrong
  0.560, n=96) — at the GREEN threshold; likely a floor (crude metric + coarse
  15 s correctness labels).
- **acappella / instrumental: FAIL with mel** (Test A ~30% ≈ chance-ish, flat
  curve; Test B AUC ~0.5). Understood, not a falsification: re-pitched vocals
  break mel (31 % of acappellas, `project_key_change_breaks_chroma`); repetitive
  instrumentals + choruses are self-similar → high match everywhere → no
  localization. These are exactly the stems the codebase already routes to
  invariant features (HuBERT vocals, chroma/fingerprint instrumental) via
  `harness/axes.py`.

**Verdict:** the free-teacher thesis holds *today, with the crudest possible
metric, for regular tracks*. The hard-stem story is more subtle than "use HuBERT."

### Acappella follow-up (2026-07-01) — HuBERT did NOT rescue it; here's why

Re-ran Test A on acappella with HuBERT-L9 (vocal stem both sides) + fibers, then +
warp-align: peak-at-0 stayed ~30% (mel) → 35% (HuBERT) → 29% (HuBERT+warp), true
score LOW (0.38, dropping to 0.155 under feature-domain warp). Three findings, all
corrective:

1. **Fibers barely fired** (0.2/12 offsets excluded as equivalent) — repeat-chorus
   ambiguity is NOT the acappella bottleneck here. (Earlier claim retracted.)
2. **Feature-domain warp is invalid** — linearly interpolating HuBERT frames smears
   them (true 0.38→0.155). Reconstruction must time-stretch AUDIO then extract, not
   interpolate features.
3. **The real cause — concurrent layers.** BB12 acappella spans are mashup `w`-rows:
   `mix_vocals` at those moments is a SUM of 2–3 overlaid acappellas, so no single
   ref reconstructs it (cosine low + flat). This is the documented root cause
   (alignment_prototype/CLAUDE.md: "a mix moment is a sum of layers ... matches no
   single ref"). My per-span shortcut is valid for regular (one dominant source) but
   BREAKS for layered vocals.

**Corrected Step-2 requirement:** the reconstruction loss must render the **FULL mix
= sum of ALL concurrent layers** (the original "glue all the pieces" thesis) and
warp AUDIO per layer — NOT per-span single-ref matching. Per-span recon is a valid
teacher only where one source dominates (regular host tracks). This does not
contradict the proven acappella tools (lyrics channel 2.2s, HuBERT diagonal-vote
2.1s) — those localize a single vocal in a band; they are placement, not the
summed-reconstruction training signal.

### Solo-vs-concurrent split result (2026-07-01) — PARTIAL confirm, strong form FALSIFIED

Peak-at-0 by concurrent-vocal depth (HuBERT, un-warped, n=66):

| depth | n | peak-at-0 | med true score |
|---|---|---|---|
| solo (1 vocal) | 28 | 43% | 0.159 |
| 2 vocals | 22 | 36% | 0.170 |
| 3+ vocals | 16 | **19%** | 0.166 |

- ✅ **Depth gradient is real** (43→36→19, monotonic): concurrent layers DO degrade
  reconstruction — the summed-layers point stands as *a* factor.
- ❌ **Strong form falsified:** solo acappella does NOT reconstruct like regular
  (43% / true 0.16 vs regular's 79% / 0.74). Layers aren't the whole story.
- **Remaining confounds for vocals (unresolved):** (1) tempo warp not validly applied
  — acappellas are stretched ~1.03; feature-domain interp is invalid, needs AUDIO
  time-stretch; (2) "solo-acappella" ≠ "solo-vocal" — a host regular track's vocals
  also leak into `mix_vocals`; (3) separation quality (mix vocal stem vs clean ref).

### DECISIVE close (2026-07-01) — vocal reconstruction is per-window signal-limited

Truly-solo acappella (no overlapping vocal from any acappella OR host row, n=28) with
proper AUDIO-domain time-stretch: **peak-at-0 = 39%, true = 0.157** — identical to
un-warped (43%/0.16). **Warp was NOT the confound; layer-stacking was not the main
driver either.** Even a solo, correctly-warped, same-song vocal matches at only 0.16
per window (the noise floor).

**Why this does NOT contradict the proven acappella tools:** lyrics (2.2s) and
HuBERT-diagonal-vote (2.1s) work by **accumulating weak per-window evidence across many
tiled windows into a sharp vote** — a 0.16-per-window signal that is consistently
slightly higher at the true offset still produces a decisive peak once summed. A
single-window reconstruction cosine cannot accumulate, so it sits at the floor. The
vocal signal exists; it only survives via vote/accumulation, not per-window distance.

**PERMANENT design call:** reconstruction is a strong teacher ONLY for the **regular/
host channel** (79%, single dominant source). Do NOT build per-window vocal
reconstruction into Step 2. Keep lyrics + HuBERT-diagonal-vote for acappella placement.
If vocal reconstruction is ever pursued, the objective must be **accumulation/vote-
structured** (many tiled windows → diagonal vote), not a per-window rendered-vs-real
distance — and even then it fights separation-stem quality.

## The thesis (one sentence)

A DJ mix is a puzzle whose pieces (the tracklist's songs) are known; the mix
audio itself is a free answer key — a placement is correct to the extent that
re-rendering the placed songs *reconstructs the observed mix*. That reconstruction
error is a supervision signal computable on all ~20,000 sets with **zero human GT**.

Three fields converge on the same objective, which is why it's worth betting on:
- **Neuroscience** — predictive coding / active inference: perception = infer the
  hidden causes that best explain the sensory input; weight each cue by its
  reliability (*precision* = inverse variance). We already have precision-weighted
  abstention (WS1, AUC .75).
- **Signal processing** — analysis-by-synthesis / differentiable rendering (the
  DDSP line); the learnable upgrade of André 2024's multi-pass NMF (current SOTA),
  which explains the mix as a warped, gained sum of source spectra.
- **Statistical learning theory** — self-supervised learning: the label comes from
  the data (the mix), not a human.

## Why this replaces the dead branch

The synthetic-mix pretrain came back **FLAT** (`project_synthetic_pretrain_v2_flat`):
topology-matched synthetic MERT pretrain → BB12 ablation flat, "MERT head can't
localize." Two root causes, both fixed here:
1. **Synthetic ≠ real topology.** Reconstruction runs on the *real* 20k mixes.
2. **Wrong representation.** MERT is identity, not placement (pooled-MERT argmax is
   ~900 s off; fingerprint localizes to 0.2 s). So reconstruction MUST be measured
   in a **time-aware / placement-informative space** (spectrogram / onset /
   fingerprint), never MERT-embedding space.

## The ordered plan

### Step 1 — the load-bearing probe (this week, ~half a day) — GATE

**Question:** does reconstruction error, measured in a time-aware spectral space,
actually track alignment correctness on BB12 (where we have GT)?

- **Test A (validity):** for GT-correct spans, score the match between the mix
  window and the placed ref segment at the *true* placement vs. deliberately
  *perturbed* placements (ref/set offset by ±{6,12,20,30,45,60} s). If the mix
  audio localizes content, the score curve should **peak sharply at offset 0**.
  Headline metric: *peak-at-zero rate* (fraction of spans where true beats all
  perturbations) + mean score-vs-offset curve.
- **Test B (usability):** on the actual predicted timeline, correlate each span's
  reconstruction match score with whether that prediction was correct vs GT.
  Metric: AUC of match-score as a correctness classifier; median match correct vs wrong.

**Verdict rule:**
- **GREEN** (Test A peak-at-zero high AND Test B AUC ≳ 0.65) → reconstruction is a
  usable free teacher; proceed to Step 2.
- **RED** → stop. Reconstruction can't grade placement; the whole plan is dead and
  we've spent half a day, not two months.

Implementation: `workspaces/alignment_prototype/recon_probe.py`
(mel-magnitude per-frame cosine; stem-routed: acappella→vocal stem,
instrumental→instrumental stem, regular→full mix). Output: `out/recon_probe_bb12.txt`.

### Step 2 — pseudo-label the 20k host channel, then supervised pretrain

**Chosen path (over differentiable-render for v1):** use reconstruction as a
**precision gate for pseudo-labeling** — run existing probes on a 20k host track →
candidate placement → score with reconstruction → keep as pseudo-GT only above a
confidence gate → supervised-train the model on the harvest. No differentiable
renderer needed for v1 (defer soft-DTW render to v2).

#### Step 2a result (2026-07-01) — recon alone is a FEATURE, not a standalone gate

Pseudo-label harvest precision/recall on BB12 host spans (`recon_probe.py` Test B):

| gate | top-decile precision | reaches 90%? |
|---|---|---|
| absolute match | 67% | no |
| **recon MARGIN** (chosen vs best-alternative placement) | **78%** | no |

- ✅ **Margin >> absolute** (78 vs 67) — confirms `abstention_margin` ("absolute
  cosine useless, margin is the signal"). Absolute match is track-confounded.
- ❌ **Neither hits 90%** with the crude mel metric. Reconstruction alone can't gate
  clean pseudo-labels — ~1 in 5 of the most-confident host labels is still wrong.

**Reframe:** reconstruction-margin is the **key new label-free correctness FEATURE**,
but the pseudo-label gate must be a **learned confidence fusion** over ALL label-free
signals — {recon margin, fingerprint sharpness, MERT identity margin, cross-probe
agreement, boundary novelty} — i.e. the WS1/C1 arbiter ([[project_ws1_precision_fusion]],
"learned fusion arbiter" on the not-wired list), with reconstruction as its first
signal that directly asks "does this explain the observed mix." Calibrate the arbiter
by **leave-one-set-out CV across the 5 GT sets** — which is exactly why labeling
BB10/Murph/Disco matters (they're the arbiter's CV folds, not just more training data).

#### Step 2b feasibility (2026-07-01) — cheap-feature fusion stalls; 90% is the wrong bar

5-fold CV logistic fusion of {recon_abs, recon_margin, recon_z, recon_rank, mert_conf,
path_conf} on BB12 host (n=96): fused top-decile precision **78%** = best single feature
(recon_z), fused AUC 0.593 < recon_z alone 0.659. Decode-confidence features are
weak/anti-correlated (mert_conf AUC .55, path_conf .36 — below random), so fusion on
small-n just adds noise. Two conclusions:

- **Inconclusive on fusion** — omitted **fp-sharpness**, the feature most likely
  COMPLEMENTARY to recon (recon = content-match; fp = diagonal uniqueness). MERT/path
  are decode confidences, known-weak for placement. Real fusion test needs fp-sharpness wired.
- **90% precision is the wrong bar for PRETRAIN.** Step 2's purpose is pretraining, which
  tolerates noisy labels. ~78%-precision host pseudo-labels at high recall are likely
  adequate to pretrain, then Step 3 finetunes on clean GT. 90% is a direct-supervision bar.

**Decision point reached — probing is exhausted; next is a TRAINING experiment.** The real
Step-2 question is no longer "tune the gate" but "does pretrain-on-noisy-host-pseudo-labels
+ finetune-on-GT beat finetune-alone?" That needs (a) the model + training loop built,
(b) more GT sets (BB10/Murph/Disco) for finetune + held-out eval. This loops back to the
labeling decision: those sets are the finetune/eval/CV substrate, not just more data.

#### Step 2 v1 BUILT (2026-07-01) — recon re-ranks the DP + turnkey held-out harness

Scaffolding map found: the learned model (`MertAlignHead`) does **identity only**;
placement is a non-learned DP (fp + `sequence_decode` + stem/lyrics). `span_placement_loss`
is a stub never called. So "pretrain a placement model on pseudo-labels" has no model to
train — and this is **why synthetic pretrain was flat** (pretrained identity, measured
placement; category error). Chosen path: reconstruction as a **post-infer placement
refiner** on the existing DP, plus a held-out A/B harness. No new model, no `infer.py` edits.

Built (all new modules except one additive flag):
- **`recon_rerank.py`** — for each HOST (regular) span, slide the mix window ±band around
  the predicted set_start, pick the set_start that best reconstructs the mix from the
  (fixed) predicted ref content; GATED (`--gate`, default **0.08** = do-no-harm). Acappella/
  instrumental untouched. Writes `out/<set>_recon_refined_timeline.json`.
- **`run_recon_experiment.py`** — turnkey A/B: BASELINE (pipeline) vs TREATMENT
  (pipeline+recon) scored held-out. Parameterized `--train-set/--eval-set`; only the final
  scorer reads eval GT, so held-out is honest.
- `score_timeline_vs_gt.py` — added optional `--timeline PATH` (additive).

In-domain BB12 A/B (gate tuning): gate 0.02 HURTS (median 6.3→7.7s, moves 33% of host) —
fp is already strong in-domain, loose gate overrides good placements with noisy mel argmax.
gate **0.08** = do-no-harm (median 6.3→6.1s, moves ~4/96). So in-domain recon can't help
(nothing to fix); the payoff test is **held-out BB11**, where the pipeline is weaker.

**TURNKEY — when BB11 GT is exported** (`labeling.export_als_to_gt` → `labeling/fixtures/bb11_ground_truth.yaml`):
```
venvs/audio/bin/python -m workspaces.alignment_prototype.run_recon_experiment \
  --train-set 1fsnxchk --eval-set 2nvzlh2k --infer
```
Gives (a) the first cross-set generalization number (baseline BB12→BB11 — also the
"did-we-overfit-BB12" answer) and (b) whether recon refinement helps held-out. Note:
in-domain do-no-harm ≠ held-out gain; if recon doesn't help held-out either, the mel
comparator is the ceiling → escalate to fp-vote-based recon or differentiable-render v2.

#### Step 2 forward plan

1. Wire recon-margin as a feature; build the small confidence arbiter over the
   label-free features; calibrate leave-one-set-out on the 5 GT sets; find τ@P≥90%.
2. If the fused gate clears ~90% precision at usable recall → harvest pseudo-GT on
   the 20k host channel → supervised pretrain the aligner.
3. v2 only if needed: differentiable render-and-compare loss (soft-DTW warp,
   Cuturi & Blondel 2017) for end-to-end backprop.

### Step 3 — finetune on the 5 hand-GT sets, easy→hard curriculum

Sharpen the pretrained model on BB10/11/12 + Murph + Disco Lines GT. Curriculum:
clean club sets first, mashup-dense sets later. Hold one set out for validation
(leave-one-set-out).

### Step 4 — co-training across independent views (grow labels into the hard tail)

Three conditionally-independent views of the same mix — **content/beat**
(fingerprint, chroma), **voice** (HuBERT), **words** (lyrics/ASR). Where one view
is confident, it pseudo-labels the moments where others are confused (Blum &
Mitchell co-training) — this reaches the *hard* regions plain self-training can't,
because the views don't share blind spots. Gated by the existing precision /
abstention (WS1) so we only teach on genuine confidence, else leave blank.

### Step 5 — active-label the weird sets (spend humans on coverage, not count)

Generalization to 20k depends on distributional **coverage**, not label count. Send
human labeling to the sets *most unlike* what we have — live-instrument sets (Rufus,
Galantis) — chosen by uncertainty × distance (core-set active learning). Cheap easy
sets (Murph/Disco) get machine pseudo-labels, not human hours.

## Known ceiling (decide up front)

If atoms = "catalog songs + edits," a **live re-instrumentation** (Rufus playing
their own parts, Galantis granular additions) is a piece not in the box —
reconstruction can never be perfect there. Requirement: the model must **abstain**
on out-of-vocabulary content, not hallucinate. Decide per-set: in-scope (needs a
richer atom vocabulary) vs. explicit abstain-and-flag.

## Related memory

`project_synthetic_pretrain_v2_flat`, `project_ws1_precision_fusion`,
`project_dj_mix_prior_art` (André 2024 SOTA), `project_placement_wall_was_decomposition_error`
(fingerprint localizes; MERT doesn't), `project_alignment_bootstrap_flywheel`,
`project_aligner_attention_design`.
