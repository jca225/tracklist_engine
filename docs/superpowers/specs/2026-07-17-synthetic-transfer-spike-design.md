# Synthetic→Real Transfer Spike — design

**Date:** 2026-07-17
**Status:** approved (brainstorming), pre-implementation
**Home:** `workspaces/alignment_prototype/trajectory/` (+ `synthetic_mix/`) — collision-free with the ingest agent
**Relates to:** `docs/alignment_recharacterization.md` (3-axis frame), `[[project_synthetic_audio_trajectory_positive]]`, `[[project_trajectory_scaffold]]`, `[[project_pws_gate_verdict]]`

---

## 1. Why this spike, and why now

The operative north star is a SOTA aligner that generalizes across ~40k DJ sets. Two candidate paths for "make it better" given our data regime:

- **A — PWS / co-training fusion** (fuse fixed hand-built probes; harvest weakly-labeled real sets). Already near its ceiling: the continuous label model (Gate v3) *matches* hand-tuned fusion but does not clear it. More PWS = diminishing returns.
- **B — synthetic-supervised learned model** (manufacture perfectly-labeled aligned mixes, train a placement/structure-aware model, validate on the 2 real GT sets).

The evidence says the aligner's bottleneck is **placement/structure, not fusion**: the oracle→e2e gap is ~91% placement (`[[project_oracle_e2e_gap_is_placement]]`), and trajectory accuracy sits ~17%. Fusion can't fix that — representation and decoding can. Path B attacks exactly that axis, and our assets (`mashup_compiler`, `warp_prior`, tracklists-as-constraints, this session's ride/transition synthesizers) uniquely enable it.

**The single unproven assumption that gates the entire B program is the synthetic→real gap:** does a model trained on rendered synthetic mixes transfer to real BB mixes? We have one provisional positive point (+0.048 held-out BB11, train-only). This spike turns that one-off into a decisive GO/NO-GO before we spend on scaling synthetic generation.

**Non-goal:** building the final learned aligner architecture. This spike only measures transfer with the *existing* trajectory scaffold. If GREEN, architecture is a separate effort.

## 2. What already exists (we build almost nothing)

- `trajectory/train.py` — trains the trajectory decoder; `--split set` gives honest cross-set holdout; eval scores with `path_decode.trajectory_acc` and prints a **no-model control** (argmax of raw match-similarity). The learned decoder must beat it.
- `trajectory/synthetic_adapter.py` — `build_synthetic_sets()` materializes the `~/aligning/` layout in-place for synthetic dirs; **synthetic is train-only**, scoring stays on real hand GT.
- `synthetic_mix/generate_v2.py` — BB12-realistic synthetic-window generator (curricula, seeds).
- **Data on disk:** 100 `synthetic_mixes_v2/` + 100 `synthetic_mixes/` (v1); trained checkpoints `decoder_{1fsnxchk,2nvzlh2k}.pt`, `pretrain_synthetic_{,v2_}mert.pt`; both BB `~/aligning/` eval folders.

New code is thin: an **experiment driver** that sweeps the matrix below and emits one results table + curve, reusing `train.py`'s existing train/eval path.

## 3. The experiment — a matrix, not a run

This is what makes the read decisive vs. the +0.048 one-off.

**Axis 1 — synthetic volume.** Train on N ∈ {0, ~25, ~50, ~100} synthetic mixes; eval held-out BB. Plot held-out `trajectory_acc` vs N. The "unlimited labels" premise pays only if the curve **rises** with N. Immediate plateau ⇒ the gap caps us.

**Axis 2 — synthetic realism.** Sweep the synthetic distribution: {plain concat} → {+warp/tempo} → {+transitions/rides} (this session's `generate_transitions` / ride augmentation). Does *richer* synthetic close more of the real gap? Tests whether synthetic-fidelity investment buys transfer.

**Brackets — so the numbers mean something:**
- **Floor** = no-model match-similarity argmax (already printed). Synthetic-trained must beat this or it learned nothing.
- **Ceiling** = train-on-*real*-BB, leave-one-set-out (train 2nvzlh2k → eval 1fsnxchk, and reverse). The synthetic-trained-vs-ceiling gap **is** the quantified synthetic→real gap.
- Report both directions (eval BB11 and BB12); n=2 but symmetric.

**Primary metric:** `path_decode.trajectory_acc` (strict, no fibers) — the placement/structure axis, i.e. the 91%-of-gap axis. Identity is out of scope for this spike.

## 4. Decision gate

- 🟢 **GREEN** — synthetic-trained (a) beats floor decisively, (b) `trajectory_acc` rises with volume (Axis 1 not flat), and (c) richer synthetic helps (Axis 2 monotone-ish). ⇒ Commit: scale synthetic generation on Vast, build the placement/structure-aware learned aligner. PWS demoted to fusion/harvest layer.
- 🟡 **YELLOW** — helps but plateaus below the real-trained ceiling. ⇒ Synthetic is a train-time **augmentation/regularizer**, not a standalone path. Keep probe+PWS primary; push the stretch-in-state decoder for placement.
- 🔴 **RED** — fails to beat floor on real BB, or degrades with more synthetic. ⇒ Synthetic→real gap is fatal for the learned bet. Put weight on PWS v4 (singleton-σ) + stretch-in-state decoder.

The gate is honest in all three directions; a null result is a real, publishable finding and redirects effort, it does not "fail."

## 5. Cost, sequencing, collision

- **First read = near-free:** run Axes 1–2 on the 100 existing synthetic mixes + existing checkpoints; training is small (MPS/CPU). No Vast needed to get the initial signal.
- **Scale only if promising:** Axis 1's large bucket (and any synthetic-fidelity expansion) is rendered/featurized on **Vast** — this is the "GPU stuff." Gated behind a non-RED first read.
- **Collision:** touches only `workspaces/alignment_prototype/`. Fully parallel with the ingest agent's bugfixes and with the background coverage-optimal SC pull (which remains staged read-only as tier-2 fuel for *after* a learned model exists).

## 6. Risks / honesty checks

- **Synthetic realism confound:** synthetic mixes are drawn from the same catalog/priors we designed; transfer to BB could be optimistic if BB tracks leak into the synthetic catalog. Verify the synthetic catalog excludes BB11/BB12 recordings before trusting the number.
- **n=2 eval:** only two real sets. Treat GREEN as "worth scaling to a real held-out corpus," not "solved." The 20k pull later provides more real eval sets.
- **Pretrain vs train-only:** existing `pretrain_synthetic_*.pt` suggests a pretrain→finetune path already exists; the spike must distinguish *train-only-synthetic transfer* (the pure gap measurement) from *synthetic-pretrain + real-finetune* (the practical recipe). Report both.
- **Metric strictness:** `trajectory_acc` is strict/no-fiber. A GREEN here understates; a RED is a hard no.

## 7. Deliverable

One results table (volume × realism × {floor, ceiling, synthetic}, both BB directions) + the Axis-1 curve, written to `workspaces/alignment_prototype/trajectory/out/` and summarized in the state-of-record's open-fronts, with the gate verdict (🟢/🟡/🔴) called explicitly.
