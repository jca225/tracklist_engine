# Handoff — TRM decoder graft + ablation framework (2026-07-18)

**Branch:** `trm-ablation-framework`. TRM work ends at `61ab844`; the branch
subsequently gained concurrent ridge-diagnostic commit `f4885a6`. **Not merged
to main.** Origin is still at `64a6a6f`; commits `e34a6b1`, `083c2b6`,
`61ab844`, and `f4885a6` need pushing.
**Tests:** 59 green across `pipeline/tests`, `trajectory/tests`,
`synthetic_mix/tests` (fresh rerun 2026-07-18).
**Numbers here are DIAGNOSTICS, not SSOT** — `docs/alignment_status.md` is
untouched and remains the only home for headline metrics.

---

## TL;DR — the one thing to know

The TRM (Tiny Recursive Model) decoder graft **works as an architecture** but
**synthetic training does not transfer to real** — a *measured* sim2real gap,
not a bug. The live path forward is the **real pseudo-label flywheel** (lever 2,
designed + prototyped this session), whose first experiment (**E1**) is cheap
(no money, no pi writes, ~tens of min). Start there.

---

## What was built (commits, oldest→newest)

| commit | what |
|---|---|
| `375ea1a` | spec: staged pipeline + ablation framework |
| `a1993d7` | **ablation framework** — six-stage spine, registry, two-grain runner, adapters, CLI (`pipeline/`); `make ablate` |
| `4b32dc5` | **TRM core** — `trajectory/trm.py` (TRMCore/TRMDecoder) + `trajectory/offset_coords.py` (fixed offset-bin answer, §2.3) |
| `cef4e74` | wire `train.py --model trm`; **v0 overfit PASSES 0.95** |
| `64a6a6f` | `--synthetic-only` train mode + `--max-train`/`--max-eval` knobs |
| `e34a6b1` | docs: record the measured sim2real verdict (attic ledger + bake-off doc) |
| `083c2b6` | **lever 1** — drop-from-top `ref_start` + `bb12-real` curriculum (measured realism fix) |
| `61ab844` | **lever 2** — pseudo-label flywheel design + prototype (`pseudo_labels.py`) |

---

## The measured findings (the whole arc in three numbers)

Referee = `path_decode.trajectory_acc` (strict, no fibers). Control = raw
match-sim argmax.

| test | result | meaning |
|---|---|---|
| v0 overfit (6 real spans, eval==train) | **0.95** | encoding + recursion + decode + train loop all correct — it *can* learn |
| real-only cross-set (BB12→BB11) | eval **0.075** < 0.239 control | ~150 real spans → memorization (train 0.61↑ / eval 0.075↓) |
| synthetic-only → real (40 windows, 311 spans) | train-fit **0.87** / real eval **flat ~0.09** < 0.306 control | **SIM2REAL GAP** — learns synthetic, none transfers |

**Why GPU/Vast was NOT used:** the synthetic diagnostic showed *underfitting is
not the problem* (train-fit hits 0.87). More compute only memorizes synthetic
harder while real eval stays flat. The wall is **data realism**, not throughput.
This was verified before spending a cent. (Zero money spent this session.)

**The two measured sim2real gaps** (from GT-distribution comparison, real BB11
vs synthetic v2):
1. **drop-from-top:** real spans start a track at ref≈0 **24%** of the time;
   synthetic **0%** (acap `ref_lo=uniform(20,70)`). → the decoder learned a false
   "ref_start never 0" prior that pushed eval *below* control. **FIXED** (lever 1).
2. **regular spans:** real **21%** full-track; synthetic **~0%**. **BLOCKED** —
   the catalog has only **3 regular tracks** + strict compat, so config can't
   force them in. Needs more regular stems (a data task).

---

## Key components (where things live)

- **TRM decoder:** `trajectory/trm.py` — `TRMCore` (recursion: y-once/every,
  full-T backprop, Q-head ACT, answer-latent LayerNorm), `TRMDecoder` (drop-in
  over `(sim, feat_kind, ref_valid)` emitting offset logits),
  `trm_offset_targets`/`trm_offset_ce`/`trm_decode_segments`.
  **Stability fix on record:** first run exploded (CE ~8e7); fixed by the
  answer-latent LayerNorm + grad-clip 1.0. Regression-guarded.
- **Offset answer encoding:** `trajectory/offset_coords.py` — `OffsetVocab`,
  `encode_offset_labels`, `offset_labels_to_ref_bins`. The keystone that makes
  the answer fixed-size (Tr-invariant).
- **Training:** `trajectory/train.py --model {conv,trm}` `--synthetic-only`
  `--synthetic-root DIR` `--max-train N` `--max-eval N`.
- **Ablation framework:** `pipeline/` (stages/registry/runner/adapters/cli/configs).
  `make ablate CONFIG=…`. Compose grain wired; isolate grain (decoder bake-off)
  ships with the TRM build. `make race` left intact.
- **Lever 1 (synthetic realism):** `synthetic_mix/scenario_v2.py`
  (`ref_start_sample`), `synthetic_mix/sections.py` (`drop_from_top_prob`,
  `bb12-real` preset). Generate: `--curriculum bb12-real`.
- **Lever 2 (flywheel):** `docs/trm_flywheel_design.md` (full design),
  `trajectory/pseudo_labels.py` (`pseudo_gt_row`,
  `pseudo_span_to_offset_labels`).

Specs: `docs/trm_decoder_bakeoff.md` (has the MEASURED verdict box at top),
`docs/pipeline_ablation_framework.md`, `docs/trm_flywheel_design.md`.
Verdict ledger: `attic/EXPERIMENTS.md` (bottom entry, 2026-07-18).

---

## NEXT: run the flywheel E1 (the recommended path)

**Why:** a high-confidence prediction span *is* a GT row, so training on it uses
the real distribution — the sim2real gap can't exist by construction. Cheaper to
falsify than chasing synthetic realism.

**E1 recipe:**
1. **Verify prereq (do first):** MERT + fingerprint hit caches exist for the pool
   set on pi-storage, so the agentic run is cheap. Pool sets (all pulled in
   `~/aligning/`): **BB10 `w1mgcjt`** (216 tracks, richest), Disco Lines
   `1rfb0yl9` (32), Murph `pwgrrb1` (73). BB10/Disco Lines have been through
   `infer`; **Murph needs a check.**
2. **Materialize pseudo-GT:** run the agentic driver over BB10 keeping only
   `Mode.AUTO_COMMIT` spans (the rung literally labeled "write pseudo-GT" in
   `agentic/policy.py`) → convert via `pseudo_labels.pseudo_gt_row`. No pi writes.
3. **Train + eval:** TRM on the BB10 pseudo-GT, **eval on BB11 GT** (strict LOSO:
   GT eval-only, pseudo train-only, never the same mix). Print all baselines
   (raw control, synthetic-flat, conv+Viterbi) in the same run.
4. **Target the placement axis first** — the loop's strong axis (fp+HuBERT, the
   37%-of-loss wall where synthetic transfer was flat); abstain on which-instance.

**Kill/viability signals:** (a) *starvation* — too few AUTO_COMMIT spans on BB10
(detected at step 2 for ~zero cost; mitigate by tuning the ACCEPT-precision
bar); (b) *noise floor* — student TRM can't beat its teacher's placement
precision (mitigated by targeting the strong axis). Full gate ladder + risks in
`docs/trm_flywheel_design.md`.

---

## Open items / gotchas

- **The vocal-enhance grind.** Each acappella span runs an *uncached* vocal
  enhancement subprocess during feature build → the "cached → fast" promise only
  holds for raw HuBERT, not this. A synthetic/pseudo corpus that is
  acappella-heavy pays ~20-30 min on the first pass. Caching this step is the
  real pipeline unlock (bigger lever than GPU for iteration speed). The `.feat_cache`
  disk cache (13 GB) IS real and working for HuBERT/chroma/mel — it's the
  vocal-enhance step that isn't cached.
- **Gap B is catalog-blocked** (3 regular tracks). If lever 1 is revisited,
  first add regular stems to `data/mashup_compat/stems` (from the stem library).
- **Lever 1 not yet validated end-to-end.** drop-from-top is committed + unit-
  tested + byte-identity-safe for `bb12-lite`, but whether it moves real eval
  above control needs the ~45-min regenerate(`bb12-real`)+retrain cycle. Banked,
  not proven.
- **Worktree cleanup:** the lever-2 agent's branch
  `worktree-agent-a95ba81b51954e0ac` (based off main, carries unrelated
  SoundCloud changes) can be deleted — only its 3 flywheel files were
  cherry-picked to `trm-ablation-framework`. Worktree at
  `.claude/worktrees/agent-a95ba81b51954e0ac`.
- **Stray local data (gitignored, safe to delete):** `data/synthetic_mixes_v2`
  (40 `bb12-lite` windows), `data/synthetic_mixes_real` (partial `bb12-real`).
- **No Vast box was ever rented.** Nothing to tear down. `gpubox`
  (`~/workspace/gpubox`) is the shared control plane if GPU is needed later —
  but the findings say GPU is not the current lever.

---

## Definition of "did the next session succeed"

E1 prints, in one run on the frozen referee: TRM-on-BB10-pseudo-GT eval on BB11
**vs** the raw control (0.306-ish on the eval subset) **vs** synthetic-flat (~0.09)
**vs** conv+Viterbi — and we learn whether real pseudo-labels beat synthetic
transfer on the placement axis. If yes: the flywheel is the path, scale the pool.
If starved/floored: the design doc's mitigations, then reassess.
