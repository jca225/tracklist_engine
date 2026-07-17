# Alignment Remediation Plan (handoff for a fresh agent)

> **For agentic workers:** This is a multi-front remediation roadmap, NOT a single
> feature. **Begin with Front F0** (concrete, TDD, bite-sized below). Fronts F1–F3
> are scoped research directives with a concrete first action + success criterion
> each — they are open-ended by nature; do not fake bite-sized TDD steps for them.
> Read [docs/alignment_state_of_record.md](../../alignment_state_of_record.md) and
> the EXPERIMENTS ledger FIRST.

**Author:** session 2026-07-17. **Numbers:** cite
[docs/alignment_status.md](../../alignment_status.md) (SSOT) — do not hardcode
metrics into this plan or anywhere else; they drift.

---

## 0. Honest position (why this plan exists)

Read through the three near-orthogonal axes
([recharacterization](../../alignment_recharacterization.md)):

- **Identity ~84%, generalizes → effectively done.** Stop investing.
- **Placement: median strong (agentic ~sub-3 s on BB12), tail open** (<15 s only
  ~78%, p90 ~50 s). Beats naive baselines 100–300×.
- **Structure: THE binding wall — ~85% of GT-seconds lost, and it does NOT
  generalize** (LOSO n=2: identity transfers 100%, structure decoder is +4.7 pp
  one direction, −2.1 pp the other).

**We are NOT close to the operative goal** (SOTA, ~40k sets, near-100%). The gate
is bottlenecked on **structure + GT scale**, which is months out, not the two
weeks to Aug 1. "99%" is an *aspirational* north star, not an Aug-1 deliverable
(status doc §7). The Aug-1-shippable milestone is an honest, well-measured
system + the recharacterization finding — bank that, and attack the wall in
parallel.

**Signal from the 2026-07-17 session:** the entire day went to *measurement
integrity* — a "agentic loses BB12" scare that turned out to be a stale-timeline
artifact, and three rebuilds of the placement metric. When the ruler is still
being fixed, structure-decoder results cannot be trusted. **That is why F0 is
first.**

---

## Global constraints / SETTLED — do NOT re-litigate

Copied so a fresh agent does not waste cycles re-deriving them. Each is banked;
re-opening one needs new evidence, not a fresh opinion.

- **Sensor phase is FROZEN (2026-07-09).** The channel inventory
  (fp/HuBERT/lyrics/fibers/chroma/surprise/recon/warp-prior) is rich enough. Do
  NOT add new probes/channels/priors. The wall is the *actor*, not perception.
  New probe ideas → `looptrace/NOTES.md` or the attic ledger, not code.
- **Numbers live ONLY in `docs/alignment_status.md`** (dated+SHA stamped,
  regenerated from scorers). Never hand-type metrics elsewhere; cite the doc.
- **Timelines are stale-prone artifacts.** `out/*_timeline.json` are un-tracked
  and overwritten by any `infer`/`make race`. NEVER compare timelines of
  different vintages. The cohort guard in
  `workspaces/alignment_prototype/experiments/bb_baselines.py` (mtime spread >
  6 h → hard fail) exists for this — F0 generalizes it to the main scorecard.
- **Placement is fiber-fair by nearest-instance match** (`score_spans` matches
  each span to its nearest same-recording GT instance). Do NOT exclude
  "pileup"/fibered spans from *placement* (a density-based attempt on 2026-07-17
  hid a genuine 746 s single-instance miss). Ref-fiber ("which repeat")
  ambiguity is a **trajectory-axis** concern, forgiven by `trajectory_acc`'s
  fiber-aware mode (`headF%`/`facc`), NOT by the placement number.
- **NMF/DTW (André line) are a DIFFERENT regime** (short synthetic excerpts);
  they blow up on full-length real mixes and are omitted from BB comparison by
  design. Do NOT try to force them onto BB.
- **BB11 (`2nvzlh2k`) hangs Whisper on Mac MPS.** Run its race on CUDA (Vast) or
  force CPU; never leave it to hang. Seed MERT caches from the Mac
  (`.cache/mert/{set}_mert.npz`) to skip the slow pi export.
- **Fibers are PRECISE but LOW-RECALL** (SALAMI P .88 / R .06). The detector
  misses most real repeats, so any fiber-based scoring is a *lower bound* — a
  span flagged `n_instances=1` may still be an unflagged repeat. F0 must treat
  this honestly, not paper over it.
- Dead ends: read `workspaces/alignment_prototype/attic/EXPERIMENTS.md` before
  re-testing ANY idea.

---

## Prioritized fronts

| front | what | why | shape |
|---|---|---|---|
| **F0** | Fiber-consistent (DP) scorer + generalize the cohort guard | trust the ruler before trusting decoder results | **code/TDD — BEGIN HERE** |
| **F1** | Learned trajectory decoder — dent the 85% structure loss | the sanctioned lever for the binding wall | research directive |
| **F2** | Third GT set (BB10) prep → unlock instance selector + generalization test | n=2 can't validate transfer; the lever needs n≥3 | agent-preps, human-labels |
| **F3** | Pseudo-label flywheel (co-train) toward scale | the only path to 40k without hand-GT | research directive (parallel agent active — coordinate) |

---

## Front F0 — Fiber-consistent scorer + guard generalization (BEGIN HERE)

**Goal:** one canonical scorer that (a) scores the trajectory axis by *optimal*
fiber-consistent assignment (not greedy per-span), exposing strict AND
fiber-aware honestly, and (b) refuses to score stale/mismatched timeline cohorts.
This makes every downstream structure-decoder result trustworthy.

**Architecture:** new pure module `fiber_assignment.py` holding the equivalence
extractor + optimal assignment solver; `trajectory_acc` (path_decode.py:508)
gains a `fiber_consistent=True` path that routes through it; the provenance/cohort
guard is lifted out of `experiments/bb_baselines.py` into a shared helper the main
`score_timeline_vs_gt` uses. Pure functions, table-driven tests, no audio.

**Tech stack:** Python, numpy, `scipy.optimize.linear_sum_assignment` (Hungarian).

### Task F0.1: fiber equivalence classes from GT

**Files:**
- Create: `workspaces/alignment_prototype/fiber_assignment.py`
- Test: `workspaces/alignment_prototype/test_fiber_assignment.py`

**Interfaces:**
- Produces: `equivalence_classes(gt_rows: list[dict], fibers) -> dict[int, int]`
  — maps each GT row's **0-based index** to a fiber-class id (row-index keying
  chosen over a `(track_id, ref_start bucket)` string during F0.1: no unspecified
  bucket resolution, no silent collisions, and it feeds F0.2's `assign` which is
  already in `gt_idx` terms); single occurrences get a unique singleton class.
  Low-recall caveat: only VALIDATED fibers (`fiber_id>=0` AND `n_instances>=2`)
  collapse; everything else stays singleton.

- [ ] **Step 1: failing test** — two GT rows of the same recording whose
  ref-starts fall in one validated fiber share a class; a third, unflagged
  single-instance row is its own class.
- [ ] **Step 2:** run `pytest workspaces/alignment_prototype/test_fiber_assignment.py::test_equivalence_classes -v` → FAIL.
- [ ] **Step 3:** implement `equivalence_classes` (group by fiber id; singletons for `fiber_id<0` / `n_instances<2`).
- [ ] **Step 4:** run test → PASS.
- [ ] **Step 5:** commit `feat(scorer): fiber equivalence classes (validated-only, low-recall-honest)`.

### Task F0.2: optimal fiber-consistent assignment

**Files:**
- Modify: `workspaces/alignment_prototype/fiber_assignment.py`
- Test: same test file

**Interfaces:**
- Produces: `assign(pred_segments, gt_rows, classes) -> list[tuple[int,int,float]]`
  — returns `(pred_idx, gt_idx, cost)`; cost 0 when pred lands on ANY member of
  the correct fiber class, else the real ref-time error. Uses
  `scipy.optimize.linear_sum_assignment` on a cost matrix built so within-class
  swaps are free.

- [ ] **Step 1: failing test** — a prediction landing on the wrong occurrence of
  a validated fiber costs 0; landing outside the class costs the true gap; a
  single-instance miss costs the true gap (guards the 746 s-miss regression).
- [ ] **Step 2:** run the test → FAIL.
- [ ] **Step 3:** implement the cost matrix + Hungarian solve.
- [ ] **Step 4:** run → PASS.
- [ ] **Step 5:** commit `feat(scorer): optimal fiber-consistent assignment (Hungarian)`.

### Task F0.3: wire into trajectory_acc, keep strict as diagnostic

**Files:**
- Modify: `workspaces/alignment_prototype/path_decode.py:508` (`trajectory_acc`)
- Test: `workspaces/alignment_prototype/test_fiber_assignment.py`

**Interfaces:**
- Consumes: `assign` from F0.2.
- Produces: `trajectory_acc(..., fiber_consistent: bool = False)` — when True,
  scores via the assignment; strict path unchanged. Returned tuple keeps its
  existing shape (do not break `score_timeline_vs_gt` callers).

- [ ] **Step 1: failing test** — on a hand-built span with one wrong-occurrence
  validated-fiber segment, `fiber_consistent=True` scores higher than strict, and
  a single-instance miss scores identically under both.
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** add the branch; default False (no behavior change for existing callers).
- [ ] **Step 4:** run → PASS; also run the existing `score_timeline_vs_gt` on BB12 and confirm strict numbers are byte-identical.
- [ ] **Step 5:** commit `feat(scorer): fiber_consistent trajectory scoring (opt-in, strict unchanged)`.

### Task F0.4: lift the cohort guard into the shared scorer

**Files:**
- Create: `workspaces/alignment_prototype/timeline_provenance.py` (move
  `_git_sha`/`driver_provenance`/`cohort_spread_s` out of `experiments/bb_baselines.py`)
- Modify: `experiments/bb_baselines.py` (import from the new module — no logic change),
  `score_timeline_vs_gt.py` (stamp provenance + warn on stale cohort in its report)
- Test: `workspaces/alignment_prototype/test_timeline_provenance.py`

- [ ] **Step 1: failing test** — a cohort spanning >6 h is flagged incoherent; the
  extracted functions return the same values the bb_baselines tests already pin.
- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3:** move the functions; re-point bb_baselines; add a provenance line to the scorer's output.
- [ ] **Step 4:** run the new test + the existing bb_baselines tests → PASS.
- [ ] **Step 5:** commit `refactor(scorer): shared timeline-provenance guard (scorecard now stamps + warns)`.

### Task F0.5: regression-pin the canonical numbers

**Files:**
- Create: `workspaces/alignment_prototype/test_scorecard_regression.py`

- [ ] **Step 1:** write a test that runs `score_spans` on the committed BB12
  coherent timelines and asserts the headline placement medians match
  `docs/alignment_status.md` within tolerance (pin the ruler so the next stale-
  artifact scare is caught by CI, not by a day of debugging).
- [ ] **Step 2:** run → PASS (or FAIL loudly if timelines are stale — then regenerate coherently first).
- [ ] **Step 3:** commit `test(scorer): regression-pin canonical BB12 placement`.

**F0 success criterion:** `make scorecard` emits strict AND fiber-consistent
trajectory, stamps timeline provenance, and hard-warns on a stale cohort; a
regression test guards the headline. Then — and only then — decoder experiments
are measurable.

---

## Front F1 — Learned trajectory decoder (the binding-wall lever)

**Directive, not a TDD plan.** Sensor phase is frozen; the actor is the wall.

- **Where:** `workspaces/alignment_prototype/trajectory/` (`train.py`, `model.py`,
  `decode.py`, `targets.py`, `recon_loss.py`, `synthetic_adapter.py`). Current
  state: scaffold built, held-out BB11 ~0.42/0.49 (verify vs status doc).
- **First action:** with F0's fiber-consistent scorer live, re-baseline the
  decoder LOSO (train BB12 → decode BB11 and vice-versa) and report the delta vs
  the classical decode **on the trustworthy metric**. The 2026-07-12 LOSO result
  (+4.7 / −2.1 pp) was on the old scorer; it must be re-measured.
- **Success criterion:** a decoder config that reduces the structure-loss share
  (currently ~38% decode-residual + the 85% GT-seconds-lost figure — cite status
  doc) on **held-out** data, both LOSO directions, without regressing placement.
- **Guardrail:** every claimed gain regenerates the status doc and lands in the
  EXPERIMENTS ledger (win or dead end).

---

## Front F2 — Third GT set (BB10) to unlock generalization

**Directive.** n=2 cannot validate transfer or train the instance selector.

- **First action (agent can do):** `make check-inventory SET=<BB10 id>` then
  pull for labeling via the `alignment-pull` skill / `pull_set_for_alignment.py`.
  Prep the seeded review artifacts. **Labeling itself is John's — do NOT pitch
  seeded sessions as GT** (settled; see labeling CLAUDE.md).
- **Success criterion:** BB10 GT written back, then re-run F1's LOSO across THREE
  sets — the first honest test of whether structure *starts* to generalize. If
  yes, wire the learned instance selector (`{HuBERT diagonal, fiber μ/ambiguity,
  fp sharpness}`) that n=2 could not validate.

---

## Front F3 — Pseudo-label flywheel toward scale (COORDINATE)

**Directive.** The only path to 40k without hand-GT.

- **Where:** `workspaces/alignment_prototype/cotrain.py`,
  `workspaces/pws_aligner/` (`cotrain_seam.py`, `label_model.py`, `verifier.py`,
  `run_phase1.py`). **A parallel agent was active here on 2026-07-17** — `git log`
  + scan `out/` mtimes before touching (settled coordination rule; never revert
  their workspace).
- **First action:** confirm the co-train seam consumes F0's trustworthy scorer as
  its verifier signal (a flywheel built on a bad ruler amplifies error). Then run
  the two-channel-agreement ACCEPT gate on a small grammar-coverage batch and
  measure pseudo-label precision against BB GT.
- **Success criterion:** pseudo-labels whose precision (verified against held-out
  GT) clears the label-model threshold, on a batch selected by DJ-move grammar
  coverage — not popularity.

---

## Self-review

- **Coverage:** F0 fixes the ruler (the session's actual finding); F1 is the
  binding-wall lever; F2 the generalization unlock; F3 the scale path. The four
  map to the four gaps in §0.
- **No placeholders:** F0 has concrete files/steps/commands; F1–F3 are explicitly
  directives with a first action + success criterion (not fake TDD).
- **Consistency:** every metric defers to `docs/alignment_status.md`; every "do
  not" traces to a settled decision above.

## Provenance / start-here pointers

- Living record: [docs/alignment_state_of_record.md](../../alignment_state_of_record.md)
  (run `/align-checkpoint` after any front moves).
- Numbers SSOT: [docs/alignment_status.md](../../alignment_status.md).
- Dead ends: `workspaces/alignment_prototype/attic/EXPERIMENTS.md`.
- This session's banked deliverables: cohort guard + fiber-fair placement metric
  in `experiments/bb_baselines.py`; coherent BB12 result; BB11 regenerated on Vast.
