# Handoff: stem-axis (acappella/instrumental) findings — 2026-06-29

For the parallel aligner agent. I spent a session trying to find a **non-colliding
solo DSP lane** to improve the acappella + instrumental axes (the known weak
spots). The conclusion: **there isn't one — the remaining lever is your segment
decode (`joint_ref_decode`/`path_decode`) or the learned model.** Below is what I
established so you don't repeat it, with the measured numbers.

I touched **no hot files.** New artifacts: `instrumental_ref_offset_eval.py` (a
probe), `docs/stem_routing_plan.md`, this doc. The continuity result reused your
`continuity_refine.py` read-only.

## The five findings (all measured on BB12, `1fsnxchk`)

1. **Per-axis baseline** (`infer` + `score_timeline_vs_gt --fibers`): acappella is
   the **worst axis** — segment traj-acc **10%** (vs regular 25%), set_start median
   **42.5s**. Instrumental is **unmeasurable**: only 1–2 instrumental GT spans
   survive to the placement scorecard (rest skipped/mis-id'd upstream). Memory:
   `project_bb12_per_axis_baseline`.

2. **Instrumental fingerprint ref-offset** (new probe, n=7): chroma median **0.3s**
   / 57% <2s vs fingerprint **30.6s** / 43%. **Fingerprint does NOT beat chroma**
   on instrumental ref-offset — and the "instrumental 0%" claim was about
   stem-presence/set_start-under-crosstalk, a *different* sub-task. No stem-fp
   instrumental channel is justified. Memory: `project_instrumental_refoffset_chroma_ok`.

3. **Acappella HuBERT ref_start** (`acappella_ref_offset_eval`, n=80): HuBERT
   median **38.7s** vs chroma 57.7s — HuBERT wins (40/27) but the **BB11 2.1s
   median does NOT generalize** to BB12 (denser, more repeat-ambiguity). A plain
   feature-swap into the ref_start path buys ~57→39s — marginal, still unusable.
   Memory: `project_hubert_vocal_ref_offset` (updated).

4. **Continuity-stack on acappella** (`continuity_refine --feature hubert --stems
   acappella`): rigid stack (band 0) is an **exact no-op** (baseline 50% = stack
   50%, 0 fixed / 0 broke); the warp band only hurt. The decisive line:
   **`104 of ~112 acappella spans excluded as loop/segment/odd-ratio`** — 93% of
   acappella is **non-linear**, which a single-line stack structurally can't touch.
   Memory: `project_continuity_stack_acappella_deadend`.

5. **Synthesis:** three optimistic premises (instr-fp, acappella-HuBERT,
   continuity-stack) all falsified by cheap probes. The acappella bottleneck is
   **non-linear segment decode + placement** — i.e. **your `joint_ref_decode`
   territory** (you already lifted acappella traj 11→18% with HuBERT routing). The
   instrumental axis can't be worked until more instrumental GT exists to measure
   against. This is the predicted DSP plateau (`project_alignment_bootstrap_flywheel`).

6. **ROOT CAUSE of the acappella weakness = WARP, not "density"** (John's
   diagnosis, measured from the .als clip rows — `tempo_ratio` +
   `ref_segments`/`is_loop`):

   | set | stem | n | stretched% | non-linear% | med\|ratio-1\| |
   |---|---|---|---|---|---|
   | BB11 | acappella | 91 | 89% | **16%** | 0.083 |
   | BB12 | acappella | 91 | 74% | **55%** | **0.482** |
   | BB11 | instrumental | 2 | 100% | 0% | 0.577 |
   | BB12 | instrumental | 7 | 43% | 71% | 0.027 |

   BB11 acappellas are mostly STRAIGHT (linear offset nails them → HuBERT 2.1s);
   BB12 acappellas are heavily WARPED — 55% non-linear AND ~48% median stretch
   (vocals from slow songs sped onto an EDM grid, warped phrase-by-phrase) → the
   linear matched filter can't track them and the extreme stretches fall outside
   its grid (→ HuBERT 38.7s). This is WHY HuBERT didn't generalize and WHY
   continuity-stack is a no-op (it excludes the non-linear spans = the actual 55%).
   **Implication for your decode:** acappella placement needs a warp-aware /
   wide-stretch path decode, not a linear matched filter — the linear stretch grid
   is too narrow for ~48% acappella stretches. Memory: `project_hubert_vocal_ref_offset`.

## What this means for you

- **Acappella improvement = better non-linear segment decode**, not a feature swap
  and not continuity. The 104 loop/segment/odd-ratio acappella spans are where the
  10% traj-acc lives. That's yours.
- **Don't build an instrumental placement channel yet** — no GT to validate it
  (n=1–2 survive scoring). Instrumental ref-offset already works via chroma.
- The HuBERT vocal feature is right for *identity/verification*, but on BB12 it does
  **not** solve ref_start placement alone — repeat-ambiguity dominates.

## Reproduce
```
venvs/audio/bin/python -m workspaces.alignment_prototype.instrumental_ref_offset_eval \
  --als "$HOME/aligning/1fsnxchk__*/BB12 align.als" --set-dir "$HOME/aligning/1fsnxchk__*"
venvs/audio/bin/python -m workspaces.alignment_prototype.continuity_refine \
  --eval --feature hubert --stems acappella --acap-band-s 0.0
```
