# Hand-off: north-star alignment work doable WITHOUT BB10 (n=2)

**Date:** 2026-07-18 · **From:** the oracle-ladder session · **For:** the next aligner agent
**One-line:** the biggest modelling prize (the learned acappella **instance selector**)
is confirmed worth building, but *fitting* it needs a 3rd GT set (BB10, unlabeled).
Everything that **de-risks and pre-builds** it — plus the co-equal placement lever —
is doable at n=2 now. This is that work queue.

---

## 0. Read first (context, in order)

1. `docs/alignment_recharacterization.md` — the three-axes frame (identity / placement /
   **structure**). Read progress as three curves. Structure is the frontier.
2. `docs/alignment_status.md` — the SSOT for every headline number. Never hand-type a metric.
3. `workspaces/alignment_prototype/CLAUDE.md` — "Phase policy (sensor phase is CLOSED)",
   "Design decisions", "Not wired yet / scoped". **Do NOT add new probes/channels/priors.**
4. `workspaces/alignment_prototype/evals/ORACLE_LADDER_FINDINGS.md` — what this session
   established (below).
5. `workspaces/alignment_prototype/attic/EXPERIMENTS.md` + `looptrace/NOTES.md` — dead ends.
   **Read the verdict before re-testing anything.**

## 1. What this session established (start here, don't redo it)

The **oracle ladder** (`evals/oracle_ladder.py`) decomposed the acappella oracle→e2e
trajectory gap, both GT sets, decoder held fixed at looptrace, fixed GT-acap denominator:

- **Instance selection is the binding constraint at the oracle ceiling.** With placement +
  identity + routing all oracle (R3), acappella strict caps at **0.373 (BB12) / 0.300 (BB11)**
  while fiber is **0.591 / 0.536**. The strict→fiber headroom (~+22–24 pp) **survives placement
  being fixed** and is positive both sets → the selector is *necessary* to break ~35%.
- **Co-equal with placement**, which is set-dependent (BB12 +10.8 vs BB11 +20.1 pp) — matches
  cotrain's placement-non-transfer.
- **Routing** is a free +12.6 pp (BB12) hand-off, already filed:
  `docs/agent_handoff_routing_misroute_20260718.md` (ingest/data-engine turf — not yours).

**What the ladder did NOT test, and what you do next:** the headroom is only *realizable* if the
correct repeat instance is **distinguishable** from the wrong-but-same-fiber instances by the
available features. That is Task A.

## 2. The n=2 boundary (what waits for BB10 vs what doesn't)

- **Needs BB10 (do NOT attempt at n=2):** *fitting* the learned instance selector and claiming a
  LOSO win over the classical decoder. A fitted selector validated on n=2 violates the project's
  overfitting bar. Six decode-layer instance threads already died this way (attic).
- **Doable now at n=2:** everything that *measures whether the selector can work*, *builds its
  input pipeline*, and *improves the transferable placement/structure decoder*. That is Tasks A–C.

---

## Task A — Oracle instance-selection separability + cross-set transfer  ★ flagship

**Question:** can `{HuBERT diagonal evidence, fiber μ/ambiguity, fp sharpness}` actually *rank the
correct repeat instance* above its same-fiber rivals — and does that ranking *transfer* BB11↔BB12?
This is the parameter-free gate that decides whether BB10-labeling-then-fitting will pay off.

**Approach (measurement, not a fitted model):**
1. Population = acappella GT spans that are fiber-aware-correct but strict-wrong (the recoverable
   set the ladder's +22 pp lives in). Get them from the fiber-aware scorer
   (`score_timeline_vs_gt --fibers`) + `ref_fibers`.
2. For each such span, enumerate candidate ref instances *within the GT fiber*
   (`ref_fibers.compute_fibers_soft` / `fiber_intervals`).
3. Per candidate compute the three features:
   - **HuBERT diagonal evidence** — the windowed matched-filter / path score at that instance's
     offset (`path_decode._scores_at_stretch` / `trajectory_acc` machinery, HuBERT L9).
   - **fiber μ / ambiguity** — `compute_fibers_soft` membership + `fiber_ambiguity`.
   - **fp sharpness** — landmark offset-histogram sharpness at that instance
     (`landmark_fp.fp_offset` / `mix_fp_hits`).
4. **Oracle ceiling:** rank by each feature alone + a fixed linear combo; does the top pick match
   the GT instance? What fraction of the strict→fiber gap does it recover?
5. **Transfer (the n=2 honesty check):** fit the *simplest* ranker (3-param logistic, or a fixed
   combo) on BB11, score BB12, and vice versa. A 3-param model is low-capacity enough that this
   LOSO is meaningful **iff** the oracle separability is real.

**Success / decision rule (threshold-free where possible):**
- GO to build the selector (once BB10 lands): oracle pick recovers **≥ ~half** the strict→fiber gap
  AND transfer is **non-negative in both directions**.
- STOP / rethink: if features don't separate, or a right answer is genuinely instance-ambiguous
  (both channels agree on the "wrong" pick — the ladder/looptrace forensics found some of these),
  the selector is bounded and BB10 won't rescue it. That's a real finding, not a failure.

**Do NOT re-walk (attic verdicts):** Phase-4 discriminability-weighted fp-support reselection
(NEGATIVE), Phase-5 residual tiebreak (inert), PWS categorical Dawid–Skene over offset bins
(REFUTED twice). This task differs: a *ranker over enumerated fiber instances* using the three
*named* features jointly — the module's own "Not wired yet" lever, un-tried.

**Home:** `workspaces/alignment_prototype/evals/` (next to `oracle_ladder.py`, reuse its
GT-row/fiber plumbing). **Arbiter:** per-set separability + transfer tables (measurement); this
feeds the selector go/no-go, it does not need to move `make scorecard` yet.

**Bonus (fold in):** the feature extractor you build here **is** the selector's input pipeline.
Persist per-candidate `{features, is_gt_instance}` as a dataset keyed by set_id, so that when BB10
is labeled, fitting + LOSO is turnkey (drop-in third set). That is the "build the dataset harness"
task, done for free.

## Task B — Improve the learned trajectory decoder (attacks the co-equal placement/structure wall)

The transferable lever for placement (which the MERT head memorizes per-set) is the learned
decoder, not a hand-tuned prior. It is **n=2-validatable now**.

- Code: `workspaces/alignment_prototype/trajectory/` (data/features/model/targets/decode/train).
  Current: conv-over-cross-similarity actor, **held-out BB11 ~0.42–0.49**. `model.py` documents
  that v2 (fixed diagonal-mean channels) REGRESSED — don't repeat it; the `stretch_slopes`
  channels are the untested knob (slope-aware pooling for tempo-stretched/oddratio spans).
- **Validation is non-negotiable:** `train.py --loso --sets bb11,bb12`, **both directions**. A gain
  on one set that doesn't survive leave-one-set-out is NOT a finding (cotrain: identity transfers,
  placement does not). No feature-engineered R² from tiny n.
- Recon-supervision (`recon_loss.py`/`recon_probe`) localizes REGULAR only — keep it host-only.

## Task C — (optional) Extend the oracle ladder for the full picture

`oracle_ladder.py` is acappella-only. Two cheap completions:
- Run the ladder for **regular / instrumental** axes (same harness, different stem filter) to see if
  instance selection is the ceiling constraint there too, or only for acappella.
- The deferred **legacy-oracle cross-check** (`path_decode --eval --feature hubert --stems
  acappella --fibers`) vs R3-looptrace, to quantify the decoder-difference at oracle placement
  (findings note flagged this as the one open cross-check).

Lower priority than A/B — it deepens understanding, it doesn't build toward the selector.

---

## Guardrails (do not skip)

- **Worktree off `origin/main`** (superpowers:using-git-worktrees). Branch fresh.
- **`make scorecard` is the only arbiter** for e2e claims; separability/transfer for Task A.
  **Axis rule:** take `claimed_stem` from the matched GT row, never the timeline span.
- **n=2 → LOSO both directions, report per-set, never a cross-set CI.** Ruthless on overfitting.
- **Sensor phase frozen:** no new probes/channels/priors. New probe ideas → a NOTE, not code.
- **Off-limits (other session's live surface):** `workspaces/pws_aligner/**`, the
  `acquisition-data-engine` / `cotrain-*` branches. Routing and identity fixes are their turf —
  you only *measure* their slices (already done).
- The single most north-star-consequential thing that unblocks the *next* tier is **labeling BB10**
  (a human/labeling task, not yours) — Task A produces the evidence that it's worth it.

## Session artifacts to build on

- `evals/oracle_ladder.py` (+ `tests/alignment_prototype/test_oracle_ladder.py`) — the ladder harness.
- `evals/ORACLE_LADDER_FINDINGS.md` — both-sets tables + decision.
- `docs/superpowers/specs/2026-07-18-acappella-oracle-ladder-design.md` + `plans/…` — how/why.
- `docs/agent_handoff_routing_misroute_20260718.md` — the routing hand-off (data-engine).
