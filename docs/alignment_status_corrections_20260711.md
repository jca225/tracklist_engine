# Alignment docs — corrections ledger (2026-07-11)

Companion to [alignment_status.md](alignment_status.md) and the overhaul handoff
[agent_handoff_docs_overhaul_20260711.md](agent_handoff_docs_overhaul_20260711.md).
Every drift found while building the single-source-of-truth doc, the corrected
value, and *why it went stale*. Numbers first regenerated at commit `bd44417`,
**re-regenerated and re-verified unchanged at commit `eb21a5e`** (2026-07-11) via
the §3 command block in the handoff.

## C1 — Set-id ↔ BB label SWAPPED in the handoff (blocking)

- **Where:** overhaul handoff §3 (and by inheritance the §8 seed labels).
- **Claimed:** `BB11 = 1fsnxchk`, `BB12 = 2nvzlh2k`.
- **Correct:** **`BB11 = 2nvzlh2k`** (Episode 11), **`BB12 = 1fsnxchk`** (Volume 12).
- **Evidence (3 independent):** `dj_sets.title` (`1fsnxchk` = "Big Bootie Mix
  Volume 12"); the `~/aligning/` directory names; the `set_id:` inside
  `labeling/fixtures/bb11_ground_truth.yaml` (`2nvzlh2k`) and
  `bb12_ground_truth.yaml` (`1fsnxchk`).
- **Why it mattered:** a cold agent running `--set-id 1fsnxchk` believing it is
  BB11 would have stamped BB12's numbers under "BB11" — injecting *fresh* drift
  into the doc meant to end drift. The handoff's §8 *magnitudes* happened to be
  labelled correctly (BB12≈44, BB11≈40), so only the §3 mapping line was wrong.
- **Fix:** corrected §3 in the handoff; annotated §8 as pre-correction/unverified.

## C2 — Loss attribution drifted (decode≫placement → decode≈placement)

- **Where:** `workspaces/alignment_prototype/CLAUDE.md` "State (2026-07-08 …)".
- **Claimed:** "decode-residual 45% > placement ~31% > identity 6%".
- **Regenerated (`make scorecard`, `_lt`, 2026-07-11):** decode-residual **38%**
  ≈ placement **37%** > mis-route 9% > identity 6% > tempo/octave 4% >
  instance-ambiguity 4% > loop-instance 2%.
- **Why:** the 2026-07-08 snapshot predates looptrace-v2 + co-train work; the
  pipeline moved, the hand-typed prose did not. Placement is now co-equal with
  decode-residual as the binding wall, not a distant second.

## C3 — Acappella strict trajectory: "21%" was cross-metric / cross-timeline

- **Where:** `workspaces/alignment_prototype/CLAUDE.md` ("acappella trajectory 21%").
- **Regenerated (`_lt`):** acappella **strict** traj-acc = **10% (BB12) / 12%
  (BB11)**; the scorecard binary success-rate is **11%**.
- **Why:** the "21%" conflated (a) a *binary success rate* vs the *mean
  within-2s traj-acc*, and (b) different timeline variants. Two different
  metrics were being quoted as one number. The canonical doc states the metric
  (mean fraction of span-seconds decoded within ±2s) and the timeline (`_lt`)
  once, up front.

## C4 — Fiber-aware headline is timeline-dependent (45 on base, 38 on `_lt`)

- **Where:** overhaul handoff §8; race board in
  `docs/agent_handoff_fibers_20260710.md`.
- **Finding:** fiber-aware multiseg+loop headline is **BB12 45 / BB11 40 on the
  base classical timeline** (race board) but **BB12 38 / BB11 37 on the `_lt`
  looptrace timeline** (`make scorecard`, the module's declared source of truth).
- **Reconciliation:** looptrace `_lt` does **not** uniformly beat base classical
  on fiber-aware trajectory — on BB12 it *regresses* the fiber headline (45→38).
  The **robust, timeline-invariant finding is the strict→fiber-aware LIFT**
  (+19–22pp across both timelines and all drivers), not any single absolute.
- **⚠ Flagged for John (not fixed — docs-only pass):** the scorecard defaults to
  `_lt`, but base classical scores higher on the fiber headline. Which
  composition is "current best" for fiber-aware trajectory is unresolved and is
  a *metric/composition* question, not a doc bug. Noted, not acted on.

## C5 — Fibers under-credited as a "scoring util" (the original trigger)

- **Where:** the status synthesis John flagged; residually in module CLAUDE.md's
  "Fibers … precise-but-low-recall (SALAMI P .88 / R .06)" framing read as a
  limitation.
- **Corrected framing:** fiber-aware scoring is a **named contribution worth
  ≈+20pp** (the "which-instance" residual — right chorus content, wrong
  occurrence), externally precision-validated (SALAMI P .88), v4 fixed the
  acappella recall hole (vocal coverage 0.06–0.28 → 0.33–0.73, ear-validated
  2026-07-09), and the phase-cancel clone certificate is wired. SALAMI's low
  recall is precision-first-by-design on a jam-band pessimistic floor, not a
  verdict against fibers.

## C6 — Set-id ↔ BB label SWAPPED in cotrain table + flywheel handoff

- **Where (two files):**
  1. `workspaces/alignment_prototype/cotrain_loso_findings.md` LOSO table
     (rows 17–18): the parenthetical set_ids were inverted relative to the
     bb-labels *and* the doc's own preamble (line 5, which is correct). Row read
     `bb11 (1fsnxchk) … 150 spans … 18.6 s`.
  2. `docs/archive/agent_handoff_flywheel_select_20260711.md` line 54:
     `bb11 = 1fsnxchk, bb12 = 2nvzlh2k` (both swapped); line 84's DoD example
     called `1fsnxchk` "bb11's" spans (internally inconsistent).
- **Correct:** **`bb11 = 2nvzlh2k`** (Episode 11), **`bb12 = 1fsnxchk`** (Volume 12).
- **Disambiguation (independent of prose):** the held-out LOSO span count `150`
  matches the actual GT-row count of **`2nvzlh2k` = BB11** (`score_timeline_vs_gt`
  reports `124/150` for `2nvzlh2k`), and the cotrain prose (bb11→18.6 s rescued,
  bb12→1436 s, bb12 has more cues yet places worse) is self-consistent with the
  bb-labels — so **only the parenthetical set_ids were swapped**, not the numbers
  or the analysis.
- **Fix:** flipped the two parenthetical set_ids in the cotrain table (numbers
  and prose untouched); corrected the flywheel handoff line 54 mapping and made
  line 84's DoD example internally consistent (`1fsnxchk` → "bb12's" spans).
- **Why it mattered:** same failure class as C1 — an agent trusting the swapped
  table/handoff would attribute BB11's 18.6 s (good) placement to BB12 and vice
  versa, exactly inverting the "which set is stable" reading. The verified
  mapping (`labeling/fixtures/bb1{1,2}_ground_truth.yaml` `set_id:`, `dj_sets.title`,
  `~/aligning/` dir names) is the anchor; the memory entry `project_gt_set_status.md`
  already has it right.

## Provenance of each regenerated number

| Number block | Command | Timeline | Status |
|---|---|---|---|
| attribution, per-axis, identity, set_start, ref-offset MAE | `make scorecard` | `_lt` | regenerated 2026-07-11 |
| per-set strict vs fiber-aware traj-acc | `score_timeline_vs_gt --set-id <id> --timeline out/<id>_predicted_timeline_lt.json [--fibers] --decompose` | `_lt` | regenerated 2026-07-11 |
| oracle placement ceiling (acappella) | `path_decode --eval --feature hubert --stems acappella --fibers` | oracle | regenerated 2026-07-11 |
| driver race board (classical/agentic/ml) | `make race` | base classical | ⚠ **carried** from `agent_handoff_fibers_20260710.md` (1 day old) — a fresh race re-runs `infer`, out-of-scope for a docs pass and would mutate timeline files |
