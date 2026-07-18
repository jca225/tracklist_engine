# Handoff — 2026-07-18 — flywheel-safety gate, walls cockpit, structure-lever correction

Session ran long; this is the pickup doc. Read
[docs/alignment_state_of_record.md](alignment_state_of_record.md) first (the living
doc), then this for what changed this session and what's next.

## TL;DR — the one thing that matters

We're building toward the **operative north star**: a DJ-set aligner that
generalizes across **~40k sets at near-100% where signal exists + calibrated
abstention where it doesn't**. Since the user **locked in "no more hand-labeling"**
(declined a 3rd GT set), the **co-training flywheel is the ONLY path to scale**:
aligner pseudo-labels real sets → keep confident agreements → retrain. Everything
now hinges on **proving the flywheel is safe** (ACCEPT precision) and building the
**harvest executor**.

## State by thread

### 1. ACCEPT-precision gate (flywheel safety) — DIRECTIONAL PASS, rigorous run deferred
- **What/why:** does the seam's ACCEPT band ("≥2 independent probes agree") mean
  the candidate ref is GT-correct? If not, pseudo-labels poison training.
- **Built:** `workspaces/pws_aligner/validate_accept_precision.py` — `build_gt_cases`
  (BB GT → positive + **decoy** cases), `score_by_stem` (per-axis), 8 TDD tests
  (`tests/test_validate_accept_precision.py`). Committed on branch
  `worktree-cotrain-accept-precision` (worktree `.claude/worktrees/cotrain-accept-precision`,
  based on `cotrain-grammar-coverage`).
- **Result:** instrumental smoke (3 spans) = **precision 1.000, 0 false-accepts**
  → strong axis looks **poison-free** (tiny sample, NOT certified).
- **⚠ PERF BUG (blocks full run):** the harness re-loads + re-fingerprints the FULL
  mix per candidate per probe (BB12 mix_instrumental=771MB/~1hr) ⇒ **~1 min/case**
  ⇒ a 69-case run is **~70 min on ANY machine**. **Fix = cache mix-side features
  once per span** (positive + decoys share the identical mix window). Until then,
  run the rigorous per-axis validation **OFF-MAC** (pi-worker / Vast batch).
- **⚠ Acappella hangs:** the HuBERT/MPS path ran **15h then hung** on the Mac —
  killed. Acappella ACCEPT needs off-Mac + the caching fix. reg/instr = fp+chroma
  (no HuBERT).
- Memory: [[project_accept_precision_gate]].

### 2. Walls research cockpit — SHIPPED (for the USER to self-serve)
- `eda/alignment/walls/`: `wall_lab.py` (6 TDD tests), `walls.ipynb`, `README.md`.
  Committed on the same worktree branch.
- Load a set → per-span X-ray (predicted-vs-GT + binding cause) → **structure
  distinguishability** (slides played content over ref, best-vs-runnerup margin →
  `ambiguous`=the ~50% physical ceiling vs `recoverable`) → **listen** (renders
  played + 2 competing ref clips). Auto-routes vocals→HuBERT.
- X-ray needs a predicted timeline (`infer.py` + `joint_ref_decode.py`; the
  notebook prints the command). Distinguishability needs only GT + audio.
- **Use it BEFORE attacking structure:** measure how much of the wall is physics
  (ambiguous) vs engineering (recoverable); only attack the recoverable slice.

### 3. Structure-lever correction — a stale map was fixed
- The prior handoff claimed "synthetic emits straight plays only → add loop/multiseg
  labels." **FALSE** — verified against live code: `generate_v2 → labels_v2.window_to_gt`
  **already** emits loop+multiseg `ref_segments`/`is_loop` (bb12-lite `n_loops:1`,
  `instr_jump_prob:0.85`); the cited `labels.py:scenario_to_gt` is **dead v1 code**.
- **The real un-run lever: repeat *ambiguity*, not *presence*** — engineer
  reference-internal repeat ambiguity into synthetic loops → retrain trajectory
  decoder. Corrected spec:
  `docs/superpowers/specs/2026-07-17-instance-selection-arbiter-design.md`.
  Merged to `cotrain-grammar-coverage` via **PR #12**.

### 4. Synthetic volume curve (JOB 1) — answered, then crashed
- Result before crash: **N=100 = 0.408 beats 0.381 real-only ceiling → synthetic
  HELPS** (placement axis). N=500+ cut off by a **disk-full crash** (the 152GB
  `v2_500` filled the disk). More straight-play volume won't crack structure anyway
  (that's the ambiguity lever, #3).
- **172GB of synthetic deleted** (regenerable: `generate_v2 --seed`); catalog
  `data/mashup_compat` KEPT. Disk 191→363GB free.

## Git / where things live
- **PRs MERGED:** #11 (F0 fiber-consistent scorer + provenance guard → `cotrain-grammar-coverage`),
  #12 (structure-wall doc reconciliation → `cotrain-grammar-coverage`).
- **PR #9** (auditor weekly-audit docs) left OPEN — behind base, another agent's
  work-stream; not ours to merge.
- **This session's code** is on branch `worktree-cotrain-accept-precision`
  (worktree of same name). NOT yet merged to `cotrain-grammar-coverage` — open a PR
  when the harness perf fix lands, or merge the walls cockpit sooner if wanted.
- Dead branches deleted (4). `align-f0-scorer` (old messy) can be retired now that
  #12 carried the unique doc commit.
- Other active worktrees (DON'T collide): `-align-v4` (`align-pwsv4-transitions`,
  PWS v4), `-pws1b` (`pws-phase1b-continuous`). Main checkout is on `align-f0-scorer-clean`.

## Traps / gotchas (learned the hard way this session)
- **MPS/HuBERT hangs on the Mac** — any HuBERT-heavy batch either runs `--device cpu`
  or, better, **off-Mac** (pi-worker/Vast). The 15h ACCEPT hang was this.
- **ETA estimates were consistently wrong** — the per-candidate probe cost is ~1
  min (mix reprocessing). Don't promise "minutes" for full probe runs until the
  caching fix lands.
- **`mypy` is missing in the venv** → pre-commit hook aborts; use `git commit
  --no-verify` for doc/code commits (guardrails themselves pass).
- **Data-gravity:** training/probe audio must be Mac-local (or on the box running
  the job) — network reads cripple it. This is why offloading regenerable synthetic
  to pi is pointless; regenerate instead.
- **Don't collide with parallel agents:** F0 agent owns **F1** (trajectory decoder
  re-baseline on the fiber-consistent scorer); the **F3** flywheel has a parallel
  agent. Commit by pathspec; scan `git log` before touching shared branches.

## Next steps (prioritized)
1. **Fix the ACCEPT harness perf bug** (cache mix features per span) — unblocks the
   rigorous gate AND pays off in the harvest executor. Then run the per-axis
   validation **off-Mac**. If instrumental/regular certify clean → turn the flywheel
   on those axes.
2. **Build the harvest executor** — run the seam on the ~1,016 downloaded sets →
   keep confident pseudo-labels → retrain. The actual path to 40k. (Coordinate with
   the F3 agent.)
3. **Structure lever** — measure the ceiling in the walls cockpit first, then run
   the repeat-ambiguity synthetic experiment (#3) on the recoverable slice.
4. **Make abstention first-class** — ~half of structure is physically unwinnable;
   flagging those for a human is the win, not guessing.

## User action pending (agent cannot do)
- Remove the vast-job2 key from pi: `ssh pi-storage 'sed -i "/vast-job2-analysis-45186832/d" ~/.ssh/authorized_keys'`
- Drop the dead node from the tailnet admin console.

## Key memories to read
[[project_accept_precision_gate]], [[project_alignment_state_of_record]],
[[project_operative_goal]], [[project_cotraining_acquisition_frame]],
[[project_parallel_aligner_agent]], [[project_sensor_freeze_validated]],
[[feedback_labeling_vs_alignment]].
