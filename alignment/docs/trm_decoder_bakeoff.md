# Handoff spec — TRM as the alignment core (and mashup generator)

**Status:** BUILT + first numbers in (2026-07-18). Architecture works; **sim2real
gap MEASURED** — see the verdict box below and `attic/EXPERIMENTS.md`.
**Owner of this doc:** whoever picks up the bake-off.

> **MEASURED 2026-07-18 (referee = `path_decode.trajectory_acc`, strict, no
> fibers; control = raw match-sim argmax). Diagnostics, NOT SSOT.**
> - v0 overfit (6 real spans, eval==train): **0.95** → encoding/recursion/decode
>   /train loop all correct; it can learn.
> - real-only cross-set (BB12→BB11): memorization, eval **0.075** < **0.239**
>   control → real is too little data (as predicted here).
> - **synthetic-only → real** (40 windows, 311 spans, real eval-only): train-fit
>   0.095→**0.87** over 200 epochs while real eval stays **flat ~0.09** < **0.306**
>   control → **sim2real gap, not underfitting.** More GPU/scale will not close
>   it (more epochs only memorized synthetic harder). Lever = synthetic REALISM
>   (§3 curriculum: bb12-lite is too clean) or the real pseudo-label flywheel
>   with TRM as the decoder. Stability fix on record: answer-latent LayerNorm +
>   grad-clip 1.0. Code: `trajectory/trm.py`, `offset_coords.py`, `train.py
>   --model trm --synthetic-only`. Branch `trm-ablation-framework`.

**One-line:** A Tiny-Recursive-Model refiner over **frozen sensors** is a candidate for the learnable
alignment core — trained on synthetic mashups (unlimited exact-GT data), the recursive
constraint-satisfaction bias fits placement/structure. **The same architecture, run forward, generates
mashups** (align = inverse problem; generate = constrained forward problem). This doc scopes the align
bake-off first because it has a fixed referee; §6 scopes the generation dual-use.

**Frame correction (important):** do NOT read this as "just a decoder swap." With the sensor freeze
validated, the recursive refiner over the frozen cross-similarity tensor **is** the learnable aligner
(perception stays upstream by design). The trajectory decoder is simply the cleanest place to *test*
the recursive-refinement hypothesis, because its harness, synthetic feed, and strict scorer already
exist. If it wins there, it graduates to the alignment core, not back into a conv-decoder slot.

---

## 0. Why this experiment exists (read before touching code)

The alignment wall is **not** identity/perception and **not** data acquisition — both were audited and
frozen (see memory: *sensor freeze VALIDATED*, *identity-miss decomposition*). The live wall is the
**decoder**: placement/structure. 84% of identity misses decompose to 69% segmentation + 26%
layered-output — a **combinatorial assignment problem** (which mix span maps to which ref segment,
under loops/jumps/odd-ratios/overlays), not a perception problem.

TRM (["Less is More: Recursive Reasoning with Tiny Networks", arXiv:2510.04871](https://arxiv.org/abs/2510.04871);
unofficial impl [lucidrains/tiny-recursive-model](https://github.com/lucidrains/tiny-recursive-model))
is a 7M-param, 2-layer net that iteratively refines a proposed answer `y` against a latent scratchpad
`z`, backpropagating through **all** recursion steps. It is SOTA on discrete constraint-satisfaction
puzzles (Sudoku 87%, Maze 85%, ARC). That inductive bias — recursive refinement toward a
crisp-constraint answer — is a direct match for the placement/structure decode.

**Why the "little data" match is real here (not the usual caveat):** a synthetic mashup is exactly the
puzzle-with-a-generator that makes TRM work on Sudoku/ARC. The forward model
`sources + operations → mix` gives us an **exact-GT generator** (`synthetic_mix.generate_v2`); alignment
is its inverse. And the domain has the rich **augmentation symmetry** TRM leans on — pitch-shift
(≈ transpose), time-stretch (≈ tempo aug), offset translation, stem swap, and slot-permutation (which
track fills which slot ≈ Sudoku digit-permutation). So the "2 real sets" concern is the wrong worry:
training is synthetic, unlimited, augmentable.

**The real risks (relocated, not dismissed):**

- **Sim2real transfer** — a TRM trained on synthetic mashups must generalize to *real* DJ sets. This is
  the genuine open question (the same sim2real gap the whole synthetic program faces), and the cross-set
  BB holdout (§4) is the honest referee. It is NOT "not enough data"; it is "does synthetic transfer."
- **Perception stays upstream** — TRM ingests the frozen sensors' cross-similarity tensor, not raw
  audio. This is by design (sensor freeze validated), not a limitation.
- **Resolvable vs clone tail** — it will help the resolvable fraction (grid-lock, monotone offset) and
  do nothing for the clone-unwinnable tail (looptrace ≈0%). Judge on the resolvable fraction.

---

## 1. The seam (what changes, what does NOT)

The existing decoder is a drop-in target. Reuse the entire `trajectory/` harness; swap only the model
and (optionally) the decode step.

| File | Role | Change |
|------|------|--------|
| `trajectory/data.py` | `TrajectorySpanDataset` — one example per GT play-span; emits `sim (2,Tm,Tr)`, mels, `target_idx (Tm,)`, `target_null`, `abstain` | **REUSE unchanged** |
| `trajectory/features.py` | stem-routed pooled features onto a common `bin_s` raster (acap→HuBERT, instr/regular→chroma) | **REUSE unchanged** |
| `trajectory/targets.py` | GT span → per-frame supervision (ref position / NULL / ignore), rasterized with the eval's own `_gt_pieces`/`_ref_at` | **REUSE unchanged** |
| `trajectory/synthetic_adapter.py` | materializes `~/aligning/<set>/` layout in-place inside each `synthv2_NNNN/`; `build_synthetic_sets()` returns train-only sets | **REUSE unchanged** — this is the data-volume engine |
| `trajectory/model.py` | conv stack → `(Tm,Tr)` grid + NULL logit | **REPLACE** with the TRM refiner (§2) |
| `trajectory/decode.py` | grid → `_viterbi` over clip-start offset → segments | **KEEP as baseline path**; TRM may emit the offset trajectory directly (§2.3), in which case decode is a thin collapse-runs step |
| `trajectory/train.py` | `--split set` (honest cross-set holdout) / `--split slot` (pooled, ~4× signal); eval via `path_decode.trajectory_acc`; prints a no-model control | **REUSE**; add a `--model {conv,trm}` flag |
| `path_decode._viterbi` / `trajectory_acc` | canonical decode coords + strict scorer | **REUSE unchanged** — this is the win-condition oracle, never modify to make TRM look better |

**Invariant:** the scorer (`path_decode.trajectory_acc`, strict, no fibers) and the eval split are the
referee. Do not touch them. The no-model control (argmax of the raw match-similarity channel) must
print alongside every run — TRM must beat it or it has learned nothing.

---

## 2. What to build — the TRM decoder

### 2.1 TRM recursion (the mechanism)
Maintain a proposed answer `y` and a latent scratchpad `z`. Given fixed input embedding `x`:

```
for improvement_step in range(N_sup):          # deep supervision, carry detached between steps
    for _ in range(T):                         # inner recursion, FULL backprop through all T
        z = net(x, y, z)                        # refine reasoning
    y = net_y(y, z)                             # refine the answer
    loss += ce(y, target)                       # supervise every improvement step
    x, y, z = x.detach(), y.detach(), z.detach()
```

- 2-layer `net`. Token mixer: start with **MLP-mixer** (TRM-MLP beat attention on the fixed-grid
  Sudoku task 87% vs 75%); keep an attention variant behind a flag for ablation.
- Backprop through **all** T inner steps (this is the HRM→TRM fix; do not use a 1-step gradient).
- Reference the lucidrains impl for the exact carry/detach bookkeeping.

### 2.2 Input `x`
Per span the dataset already gives `sim (2, Tm, Tr)` (channel 0 = stem-routed match, channel 1 = mel)
plus mels. Project to token sequence over the **mix** axis: `x[t]` summarizes mix-frame `t`'s evidence
row `sim[:, t, :]`. Optionally concatenate tracklist-token embeddings (this ref's identity/version).
Keep it minimal for v0 — the match row is the load-bearing signal.

### 2.3 Answer `y` — encode in OFFSET coordinates (LOCKED; absolute-bin is fallback-ablation only)
**Decision (2026-07-18):** the answer is encoded in **clip-start offset**, the coordinate
`path_decode._viterbi` already decodes in. Absolute-ref-bin is permitted ONLY as a fallback ablation
if the offset encoding fails to train in v0 — do not start there.

Do NOT justify this as "fixed-size" — it isn't, naively: absolute-bin `y[t] ∈ {0..Tr}` has vocab ≈ Tr,
and raw offset `= ref_pos(t) − t` ranges over `[−Tm, +Tr]`, which is *larger*. The real reasons offset
wins for alignment:

1. **Piecewise-constant target, not a ramp.** Within a straight play the offset is FLAT; a hold/loop is
   flat; a tempo-stretch is a constant-SLOPE line; a DJ jump is a sparse discontinuity. In absolute-bin
   space the same straight play is a diagonal ramp (increments every frame) with no flat structure to
   exploit. Low description length → sample-efficient, which is what buys **sim2real** (the real risk).
   This is also *why* the conv-v2 diagonal channels regressed (model.py docstring): they baked in a
   slope-1 diagonal that is wrong for stretched spans. Offset space represents stretch as a *slope*
   without assuming slope-1.
2. **Enables a small, bounded, genuinely fixed-size answer vocab** — *because* the trajectory is
   piecewise-constant, model per-frame **local/relative** offset (a bounded window or delta around a
   running estimate: `{NULL, stay, jump-by-±k for |k|≤W}`), not an absolute bin over the whole ref.
   That bounded local vocab is what actually satisfies TRM's fixed-grid assumption, and it is only
   sound in offset space. Absolute-bin would need a `Tr` cap that truncates long refs.
3. **Zero decode-coordinate mismatch** — lands directly in the coords `_viterbi`/`trajectory_acc`
   consume (runs of constant offset ARE segments; discontinuities ARE the DJ's jumps). Collapse runs of
   equal offset into segments (reuse `targets.frames_to_segments`) and score with `trajectory_acc` — no
   conversion, and the learned jump cost replaces the hand-set Viterbi `lam`.

`targets.py` currently rasterizes GT as ref position; add a thin conversion to offset bins for the TRM
target (mix-frame → ref-position is already known, offset = ref-position − mix-time). NULL stays NULL.
`unalignable` rows stay a positive abstain label with masked placement loss.

### 2.4 Losses
- Cross-entropy on `y` (offset bin incl. NULL) at every improvement step (deep supervision).
- Keep the abstain/NULL handling identical in spirit to `targets.py` (mask ignore, supervise NULL).
- Optional: the reconstruction teacher (`recon_ok` spans only, host/regular) as an auxiliary loss —
  `data.py` already emits `mix_mel`/`ref_mel` and the `recon_ok` flag. Defer to v1.

---

## 3. Training protocol

1. **v0 sanity (overfit-first, TDD):** train `--split slot` on a handful of real spans, confirm the
   TRM refiner can drive train accuracy up and beats the raw-similarity control. If it can't overfit a
   few spans, the recursion wiring is wrong — fix before scaling.
2. **Synthetic scale-up:** `build_synthetic_sets()` for train volume (this is the point — unlimited
   labeled timelines). Real BB stays **eval-only**. Never let a synthetic span into the eval set.
3. **Honest number:** `--split set --train-set 1fsnxchk --eval-set 2nvzlh2k` (and the reverse) — the
   cross-set generalization number is the headline. `--split slot` is a secondary, leakier signal.
4. **Sweep on TRAIN only:** recursion depth `T`, improvement steps `N_sup`, offset-bin resolution,
   MLP-vs-attn mixer. Any hyperparameter touched on eval invalidates the run.

Runtime: 7M params, CPU/MPS-friendly. No GPU box needed for v0. Feature caches (`.feat_cache`) already
exist; `features.py` reuses them.

---

## 4. Win condition & kill criteria

**Referee:** `path_decode.trajectory_acc` (strict, no fibers), on the cross-set holdout.

**Baselines to beat, in order:**
1. Raw match-similarity argmax (no-model control) — must beat, else it learned nothing.
2. The current conv+Viterbi scaffold on the same split (its held-out BB11 number lives in the training
   logs, e.g. `out/trajectory_train_20260707.log`; **regenerate it in the same run**, do not trust a
   stale figure — cite `docs/alignment_status.md` for any headline number).

**Kill criteria (cheap-death, honor them):**
- Cannot beat the raw-similarity control after the v0 overfit → wiring/encoding bug or wrong task
  framing; stop.
- Beats the control but not the conv+Viterbi scaffold on cross-set holdout → the recursive bias didn't
  buy anything here; record the verdict in `attic/EXPERIMENTS.md` (closed-experiments ledger) and stop.
- Wins on `--split slot` but not `--split set` → memorizing the mix/DJ, not generalizing. Not a win.

**Success:** beats the conv+Viterbi scaffold on cross-set holdout by a margin that survives the
train/eval reverse (both directions of the 2-set holdout). Then it's a real decoder replacement —
promote per the workspace graduation rule.

---

## 5. Traps (learned the hard way in this repo)

- **Do not modify the scorer or the split to flatter the model.** The referee is fixed.
- **Synthetic is train-only.** `synthetic_adapter` enforces this; don't route synth into eval.
- **Skips must be LOUD** (`Dataset.skipped`) — never silently zero-fill an unservable span (hid the
  slot-039 miss once).
- **Offset scale is checkpoint-specific.** If you keep a jump penalty anywhere, it lives on the logit
  scale of the checkpoint, not the matched-filter scale — sweep on TRAIN.
- **7M params on 2 real sets = memorization.** The whole design leans on synthetic volume; if you find
  yourself training on real BB, you've lost the plot.
- **Headline numbers belong ONLY in `docs/alignment_status.md`.** Don't hand-type accuracies into this
  doc or memory — regenerate and cite.

---

## 6. Dual-use — the same model generates mashups (the strategic reason to do this)

Alignment and generation are the **inverse and forward** of one structured-timeline problem, both solved
by recursive refinement of `y` over the offset/assignment grid subject to hard constraints:

- **Align (inverse):** `mix + sources → arrangement operations`. (This spec, §1–§6.)
- **Generate (constrained forward):** `sources + constraints → a novel valid arrangement`. Same `y`
  representation (§2.3), same recursion (§2.1); the input is target constraints (key/tempo compatibility,
  phrase-grid alignment, energy arc, slot budget) instead of an observed mix. This is **Sudoku over the
  timeline**: fill the grid so the DJ-grammar constraints hold. TRM's hard-constraint machinery is the
  right tool for the *feasible set*.

**Why this matters:** it collapses north-star step-1 (align) and step-5 (mix generation) onto one
architecture, and closes a **flywheel** — the generator feeds TRM-as-aligner training data; TRM-as-
generator produces harder/better synthetic mashups (curriculum); which sharpen TRM-as-aligner. Connects
to `mashup_compiler/`, the Appleseed studio, and `bb_mashup_grammar_v1` (memory: *Mashup grammar prior*).

**The honest ceiling — feasible ≠ tasteful.** Sudoku has one correct answer; a good mashup does not.
TRM gives the *feasible set* (arrangements that don't violate the grammar), not the *good* one.
Selecting the good arrangement is a **ranking/taste** problem — that's the taste-prior / info-dynamics /
personalization layer's job, as a reward on top of TRM's feasibility core, NOT something TRM decides.
Do not conflate "TRM produces valid arrangements" with "TRM produces good arrangements."

**Sequencing:** prove the align direction first (fixed referee, §4). The forward/generation direction
reuses the trained weights and the same grid representation — scope it as a follow-on once the shared
`y` encoding and constraint channels are validated on align. Do not build both at once.

## 7. Pointers

- Existing decoder + harness: [`alignment/trajectory/`](../trajectory/)
  (`model.py`, `data.py`, `features.py`, `targets.py`, `synthetic_adapter.py`, `decode.py`, `train.py`).
- Canonical decode coords + scorer: [`alignment/path_decode.py`](../path_decode.py)
  (`_viterbi`, `trajectory_acc`).
- Synthetic generator: `alignment/synthetic_mix/` (`generate_v2`).
- Closed-experiments ledger (READ before starting, WRITE the verdict after): `attic/EXPERIMENTS.md`.
- Status SSOT: `docs/alignment_status.md`. Set ids: BB11 = `2nvzlh2k`, BB12 = `1fsnxchk`.
- TRM: paper arXiv:2510.04871; impl github.com/lucidrains/tiny-recursive-model (TRM-MLP variant).
