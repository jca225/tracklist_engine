# Worktree/Branch Census + Reconcile Plan — Operation Crush Phase-0 (issue #49)

Date: 2026-07-21 · Baseline: `origin/main` = `60cee52` · Repo: `/Users/johnnycabrahams/Desktop/tracklist_engine`
Read-only analysis; no branch was checked out, merged, or deleted.

> **Status (2026-07-21):** Safety net executed — all 45 local branch tips tagged
> `archive/<branch>` (Step 0), so every branch is recoverable regardless of later
> deletion. No branches deleted, no worktrees removed, no PRs merged/closed yet —
> those await approval + the 5 decisions in the final section.

## Method note (important — supersedes the naive test)

Every local branch is 25–773 commits **behind** origin/main, so the prescribed
`git diff --stat origin/main <branch>` two-dot test is non-empty for ~every branch
(main's newer content shows as "deletions") and proves nothing. The decisive test
used instead, per branch:

```
tree=$(git merge-tree --write-tree origin/main <branch>)   # in-memory merge
git diff origin/main $tree                                  # net effect of landing the branch
```

- Net diff **empty** → branch adds nothing → DISCARD.
- Net diff only **conflict-marker hunks** on shared register files (EXPERIMENTS.md,
  guardrails_ratchet.json, alignment_state_of_record.md, Makefile, …) where main's
  side is the newer superset → effectively landed → DISCARD (each verified by
  inspecting the branch side of the conflict and/or naming the merged PR).
- **Clean (marker-free) net-new files** → genuinely unlanded content → LAND/PARK.

## Summary

| Metric | Count |
|---|---|
| Local branches | 45 |
| Worktrees | 29 (28 branch-attached + 1 detached) |
| Branches showing "+N ahead" | 35 |
| **Headline: "ahead" branches that are actually fully landed/superseded (squash illusion)** | **22 of 35** |
| DISCARD | 31 |
| LAND (6 already in-flight as open PRs) | 11 |
| PARK-WITH-NOTE | 3 |
| Worktrees safe to remove now | 19 |
| Worktrees removable after a 1–2-file save | 2 |
| Worktrees to preserve | 8 |

Two structural findings beyond the census:

1. **PRs #11 and #12 were merged into `cotrain-grammar-coverage`, not `main`**
   (verified via `gh pr view … baseRefName`). Consequence: the F0
   timeline-provenance guard + BB12 regression pin
   (`timeline_provenance.py`, `test_timeline_provenance.py`,
   `test_scorecard_regression.py`, the 9,480-line
   `tests/fixtures/1fsnxchk_agentic_timeline.json`, ~140 lines of `path_decode.py`
   fiber-consistency) **never reached main** — `git log origin/main -- <those files>`
   is empty. `fiber_assignment.py` did land separately (via #19). Carrier of the
   full reviewed content: `reconcile-handoff-doc` (== `origin/cotrain-grammar-coverage`,
   diff empty). → PARK, see register.
2. **The entire pws_aligner phase-1b LF suite (~5.7k lines, 44 files, fully tested)
   is unlanded.** Main has the cotrain/corpus-harvest side of `pws_aligner/`
   but none of `continuous_model.py`, the A1–A5/B1–B4 labeling functions, policy,
   stem_routing, verifier, or their 20 test files. `align-pwsv4-transitions`
   (v4 σ-shrinkage, Gate v4: "calibration collapse CURED, placement win on BB12,
   BB11 validation deferred") is a strict superset of `pws-phase1b-continuous`.

## Classification table

Verdict basis abbreviations: **net=∅** — merge-tree net diff empty; **noise** — net diff is only conflict-marker hunks on register files, main side newer; **clean+N** — N marker-free net-new files.

| Branch | Worktree | A/B vs origin/main | Net-new after merge-sim | Verdict | Rationale / landed-by |
|---|---|---|---|---|---|
| main (local) | `.claude/worktrees/alignment-data-integrity` | +0/−80 | — | DISCARD (reset) | Stale local main; fast-forward to origin/main after freeing the worktree |
| chore/daw-env-hard-nogo | `.claude/worktrees/ableton-react-harness` | +0/−25 | — | DISCARD | Landed by PR #36; daw_env NO-GO is in EXPERIMENTS.md ledger |
| crush/augmented-master-plan | `.claude/worktrees/crush-master-plan` | +0/−1 | ∅ (even 2-dot diff empty) | DISCARD | Landed by PR #54 |
| feat/ableton-react-harness | — | +0/−27 | — | DISCARD | Landed by PR #33 |
| feat/adaptive-fp-fibers | `.claude/worktrees/adaptive-fp-fibers` | +0/−30 | — | DISCARD | Landed by PR #32 |
| fix/bb12-inventory-audio-repair | `.claude/worktrees/bb12-inventory-repair` (dirty: 1 M + out/) | +0/−11 | — | DISCARD | Landed by PR #39; save `.superpowers/sdd/task-3-report.md` first if wanted |
| fix/eda-source-audio-resolve | `.claude/worktrees/eda-source-resolve` (dirty: 2 M) | +0/−23 | — | DISCARD | Landed by PR #38; 2 uncommitted post-merge edits — inspect before removal |
| fp-hit-decoder-clean | `.claude/worktrees/fp-hit-decoder-clean` (untracked caches only) | +0/−33 | — | DISCARD | Landed by PR #31 |
| race/post-gt-regen | `.claude/worktrees/gt-transactional-writeback` | +0/−77 | — | DISCARD | No unique commits; GT-regen chain landed via PRs #27–#29 |
| worktree-soundcloud-datalake | `.claude/worktrees/soundcloud-datalake` | +0/−91 | — | DISCARD | Landed by PR #17 |
| fix/transactional-gt-writeback | — | +1/−78 | **net=∅** (clean merge, NO-NET-CHANGE) | DISCARD | Landed by PR #28; its 1 ahead commit (status regen) landed as PR #29 |
| instance-separability | — | +10/−105 | **net=∅** | DISCARD | Landed by PR #18 (squash) |
| worktree-acap-oracle-ladder | `.claude/worktrees/acap-oracle-ladder` | +8/−105 | **net=∅** | DISCARD | Landed by PR #16 |
| acquisition-data-engine | — | +11/−105 | noise (1 file, +3 markers) | DISCARD | Landed by PR #15 |
| cotrain-corpus-harvest | `.claude/worktrees/cotrain-accept-precision` | +76/−85 | noise (3/4 files markers; 1 ledger-echo) | DISCARD | Landed by PR #14 |
| worktree-cotrain-accept-precision | — | +48/−117 | noise (8/9) | DISCARD | All commits contained in cotrain-corpus-harvest → PR #14 |
| cotrain-grammar-coverage | — | +38/−117 | noise (6/7) | DISCARD | Integration base of #11/#12; content carried by reconcile-handoff-doc (diff to origin/cotrain-grammar-coverage = empty) |
| cursor/cloud-agent-1780150002017-tawdd | — | +1/−773 | noise (7/7; main superset of every hunk) | DISCARD | Stale 2026-05-30 cloud-agent snapshot |
| docs-consolidation-phase0 | `.claude/worktrees/docs-consolidation-phase0` | +29/−105 | noise (13/13) | DISCARD | Landed by PRs #20 + #22 (and #19/#24/#25 chain) |
| e1-bb10-unblock | `.claude/worktrees/e1-bb10-unblock` | +2/−74 | noise (1/1; branch side = n=16 ledger note) | DISCARD | Fix landed by PR #30; the n=16 noise-floor note is superseded by PR #35's n=24/n=33 records (fold a line into #35 if desired) |
| feat/e1-flywheel-leftovers | `.claude/worktrees/e1-leftovers` | +8/−83 | noise (5/5) | DISCARD | Landed by PR #25 |
| feat/ridge-diagnostic-leftovers | `.claude/worktrees/ridge-leftovers` | +6/−83 | noise (2/2) | DISCARD | Landed by PR #24 |
| feat/track-audio-id-index | `.claude/worktrees/track-audio-id-index` | +1/−89 | noise (3/3; branch side is **older**: INDEX_VERSION 1 vs main 2) | DISCARD | Landed by PR #23, main evolved past it |
| fix/regenerate-ableton-gt | `.claude/worktrees/regenerate-ableton-gt` | +1/−80 | noise (1/1; branch side = stale slot labels `024` vs canonical `007w1`) | DISCARD | Landed by PR #27 |
| fix/trm-ablation-merge | `.claude/worktrees/trm-pr-check` | +31/−84 | noise (8/8) | DISCARD | Content landed via PR #19 + #24/#25 chain |
| worktree-agent-a95ba81b51954e0ac | `.claude/worktrees/agent-a95ba81b51954e0ac` | +19/−90 | noise (8/8) | DISCARD | Content landed via PR #19 chain |
| fp-hit-decoder-wall | `.claude/worktrees/fp-hit-decoder-wall` (untracked caches only) | +32/−105 | noise (13/13) | DISCARD | Its 2 unique commits landed via the #31 chain: main has `9b1fbb4` "prefer competitive fp cluster" + `MODE_AUDIT.md` |
| vast-box-skill | — | +2/−197 | noise (4/4; main's vast_box.py has race/quarantine superset) | DISCARD | Superseded on main (--race etc. landed via later branches); gpubox is now the preferred path anyway |
| align-f0-scorer | — | +54/−117 | clean+7, but superseded | DISCARD | Pre-review draft; reconcile-handoff-doc carries the reviewed successor (cross-diff shows reconcile is newer). Two review-dropped files (`looptrace/NOTES.md`, `results/bb_baselines_placement.json`) noted in Park register |
| align-f0-scorer-clean | — | +57/−117 | clean+6, but superseded | DISCARD | = PR #11 content; fully contained in reconcile-handoff-doc (PR #12 fold) |
| pws-phase1b-continuous | — | +39/−141 | clean+43, but superseded | DISCARD | Strict subset of align-pwsv4-transitions (every commit present there) |
| e1-flywheel | `.claude/worktrees/e1-flywheel` (**dirty: 11 M + handoff doc**) | +32/−105 | committed side: noise (13/14; the 1 "clean" train.py hunk is a duplicate-def artifact of a function main already has via PR #25) | **PARK** | Committed content landed (PR #25 / #30 chain); park is for the uncommitted work only |
| reconcile-handoff-doc | — | +53/−117 | clean+7: `timeline_provenance.py`, `test_timeline_provenance.py`, `test_scorecard_regression.py`, 9,480-line BB12 timeline fixture, `path_decode.py` fiber-consistency, instance-arbiter design doc | **PARK** | Reviewed twice (#11/#12) but never reached main (PRs based on wrong branch). Scorer has since evolved (WS0 deinflation, A2 timebase refactor) so the BB12 pin figure is stale — re-land requires re-validation, not a mechanical merge |
| work-grouping-proposal | — | +2/−184 | clean+3: proposal doc, `scripts/propose_work_grouping.py`, `work_map.json` (+scorer hunk) | **PARK** | Explicit DRY-RUN proposal; overlaps the Crush path/identity root-cause plan — user decision whether it still stands |
| crush/plan-corrections | `.claude/worktrees/crush-plan-corrections` | +2/−0 | clean merge, +378/−47 | **LAND** | In-flight as PR #55 (MERGEABLE) |
| crush/content-identity | `.claude/worktrees/crush-content-identity` | +2/−0 | clean merge, +298 | **LAND** | In-flight as PR #56 (draft, MERGEABLE) |
| feat/als-audio-roundtrip | `.claude/worktrees/als-audio-roundtrip` | +3/−29 | clean merge, +1781/−55 | **LAND** | In-flight as PR #37 (MERGEABLE); superset of #34 |
| feat/gt-release-gate | `.claude/worktrees/gt-release-gate` | +2/−29 | clean merge, +1024/−43 | **LAND** | In-flight as PR #34 (MERGEABLE); both commits contained in #37 — user decision: merge #34 first (smaller review) or close in favor of #37 |
| e1-hubert-corroborate | `.claude/worktrees/e1-hubert-corroborate` (**dirty: 5 M + 3 untracked**) | +7/−29 | 6 clean files + 1 conflict (EXPERIMENTS.md only) | **LAND** | In-flight as PR #35 (**CONFLICTING** — single-file ledger conflict, easy rebase). Dirty worktree extras (persist_e1_pool.py, e1_session_state.md, lyrics/mert cache edits) must be committed or parked first |
| earliest-instance-tiebreak | `.claude/worktrees/acap-instance-separability` (**dirty: 1 M + new test**) | +13/−105 | clean+5: `path_decode.py` + `looptrace/{run,segments}.py` tie-break, 2 new test files, `joint_ref_decode.py` | **LAND** | Implements the winning verdict of PR #18 (earliest-instance positional prior). Local-only branch (no remote). New PR; fold in the uncommitted `test_joint_ref_decode_earliest.py` |
| align-pwsv4-transitions | — | +71/−117 | clean+44: full pws_aligner LF suite + continuous model + v4 | **LAND** | ~5.7k lines of tested workspace code main lacks; memory says continuous/cotrain still live. One PR. Caveat: v4 promotion deferred pending BB11 validation — land as workspace code, keep claims out of the SSOT status doc |
| synthetic-warp-wiring | — | +1/−260 | clean+1 big: `lab/corpus_empirics/bb_mashup_grammar.py` (896 lines) + findings.md/eda-CLAUDE hunks | **LAND** | The mashup-grammar-prior finding is in memory but the code/findings never landed. Lab backfill PR |
| eda/mashup-pir-infodyn | — | +2/−202 | clean+3: `lab/information_dynamics/{FINDINGS.md, bb_mashup_pir_v1.py, markov_infodyn.py}` | **LAND** | P1 HuBERT-token PIR work (distinct from the refuted chroma critic); local-only branch. Same lab backfill PR |
| identity-miss-decomposition | — | +1/−105 | clean merge, +372: decomposition doc + `identity_miss_decompose.py` + structural-levers plan | **LAND** | Finding is cited in memory; artifacts unlanded. Small docs/eda PR |
| trm-ablation-framework | **root repo** (dirty: 1 M `bb12_ground_truth.yaml`) | +61/−105 | **124 clean net-new files, ≈ +23k lines** | **LAND** | Current working branch. Of its +61 commits, the bottom ~24 re-landed via PRs #15/#19/#24; everything from `d700e94` up (2026-07-19 → 21) is genuinely unlanded. Split into 4–5 focused PRs (below) |

## Consolidation plan — proposed PRs for the LAND set

Ordering matters because five branches touch `labeling/` (`export_als_to_gt.py`, `write_back_ground_truth.py`, `labeling/als/*`).

**Wave 1 — merge the open PRs (in this order):**
1. **PR #55** `crush/plan-corrections` — docs + `audit_gt_recording_ids.py`. No code conflicts.
2. **PR #34 or #37** (user decision): #34 (`feat/gt-release-gate`) is a reviewed subset of #37 (`feat/als-audio-roundtrip`). Either merge #34 then rebase-shrink #37, or close #34 and merge #37 directly. Do not merge both as-is (duplicate content).
3. **PR #35** `e1-hubert-corroborate` — resolve the single EXPERIMENTS.md ledger conflict; optionally fold the n=16 note from `e1-bb10-unblock` while there. Commit/park the worktree's uncommitted extras first.
4. **PR #56** `crush/content-identity` (draft → ready) — content-addressed clip identity. Rebase over whatever #34/#37 did to `labeling/`.

**Wave 2 — new small PRs from standalone branches:**
5. **PR: earliest-instance tie-break** (`earliest-instance-tiebreak`) — scope: `path_decode.py`, `looptrace/run.py`, `looptrace/segments.py`, `joint_ref_decode.py` + 3 test files (incl. the uncommitted one). Push branch to origin first (local-only).
6. **PR: lab backfill** (`synthetic-warp-wiring` + `eda/mashup-pir-infodyn`) — scope: `lab/corpus_empirics/bb_mashup_grammar.py` + findings.md section, `lab/information_dynamics/{bb_mashup_pir_v1.py, markov_infodyn.py, FINDINGS.md}`. Pure lab/, no DAG entanglement.
7. **PR: identity-miss decomposition artifacts** (`identity-miss-decomposition`) — scope: `eda/alignment/failure_analysis/{IDENTITY_MISS_DECOMPOSITION.md, identity_miss_decompose.py}` + plan doc. Clean merge today.
8. **PR: pws_aligner phase-1b + v4** (`align-pwsv4-transitions`) — scope: 44 files under `pws_aligner/` + `scripts/backfill_track_fingerprints.py` hardening. Land as experimental workspace; keep v4 promotion claims out of `docs/alignment_status.md` (BB11 validation still pending). Supersedes and retires `pws-phase1b-continuous`.

**Wave 3 — split the current branch (`trm-ablation-framework`, 124 unlanded files):**
9. **PR-A: labeling/als toolkit + manifest reconciliation** — `labeling/als/*` (cst, raw_cst, read/write, roundtrip, validate, tags, identity, gap_classify), `labeling/ground_truth/schema.py`, `export_als_to_gt.py`, `write_back_ground_truth.py`, `labeling/fixtures/id_maps/*`, `scripts/{resolve_manifest_recording_ids,diagnose_manifest_als_paths,build_work_map,reconcile_works}.py`, `gt_review_ui.py`. Must go after #56 (both touch `labeling/als/`). Include the currently-uncommitted `bb12_ground_truth.yaml` edit (commit it first).
10. **PR-B: learned identity verifier (Phase 8)** — `analysis/identity_{data,learned,verify}.py`, `scripts/{train,eval}_identity_verifier.py`, `reports/identity_verifier_eval_*.json`, status-doc line.
11. **PR-C: acquisition cascade (Phase 5)** — `ingest/{cascade_adapters,stem_cascade}.py`, `scripts/run_cascade.py`, `core` refuse-to-guess gate, `labeling` open-case-at-gap-detection.
12. **PR-D: ops/docs batch** — crush/handoff docs, gpubox scripts (`scripts/gpubox_*`, `_gpubox_relaunch_agentic`, `github_deploy`), vast docs/skills, `eda/alignment/spectrogram_review/*`, remaining plans/specs.

## Worktree cleanup list

**Safe to `git worktree remove` now (19)** — branch is DISCARD and tree clean (or untracked caches/outputs only):

| Worktree path (under `.claude/worktrees/` unless noted) | Branch |
|---|---|
| `ableton-react-harness` | chore/daw-env-hard-nogo |
| `acap-oracle-ladder` | worktree-acap-oracle-ladder |
| `adaptive-fp-fibers` | feat/adaptive-fp-fibers |
| `agent-a95ba81b51954e0ac` | worktree-agent-a95ba81b51954e0ac |
| `alignment-data-integrity` | main — remove, then `git fetch . origin/main:main` to un-stale local main |
| `cotrain-accept-precision` | cotrain-corpus-harvest |
| `crush-master-plan` | crush/augmented-master-plan |
| `docs-consolidation-phase0` | docs-consolidation-phase0 |
| `e1-bb10-unblock` | e1-bb10-unblock |
| `e1-leftovers` | feat/e1-flywheel-leftovers |
| `fail-closed-audio-resolvers` | (detached `fa3d4cc`; PR #21 landed) |
| `fp-hit-decoder-clean` | fp-hit-decoder-clean (untracked caches only) |
| `fp-hit-decoder-wall` | fp-hit-decoder-wall (untracked caches/out only) |
| `gt-transactional-writeback` | race/post-gt-regen |
| `regenerate-ableton-gt` | fix/regenerate-ableton-gt |
| `ridge-leftovers` | feat/ridge-diagnostic-leftovers |
| `soundcloud-datalake` | worktree-soundcloud-datalake |
| `track-audio-id-index` | feat/track-audio-id-index |
| `trm-pr-check` | fix/trm-ablation-merge |

**Removable after a small save (2):**
- `bb12-inventory-repair` — save/discard the modified `.superpowers/sdd/task-3-report.md` (low value, likely discardable) and the untracked `alignment/out`.
- `eda-source-resolve` — inspect 2 modified files (`eda/alignment/spectrogram_review/source_audio.py`, its test): post-PR-#38 tweaks; keep as patch if non-trivial.

**Preserve (8):** root repo (trm-ablation-framework), `acap-instance-separability`, `als-audio-roundtrip`, `crush-content-identity`, `crush-plan-corrections`, `e1-flywheel`, `e1-hubert-corroborate`, `gt-release-gate`.

## Park register

1. **`reconcile-handoff-doc`** (branch kept; = `origin/cotrain-grammar-coverage`). Preserve: the F0 timeline-provenance guard + BB12 regression pin (`timeline_provenance.py`, 2 test files, the 9,480-line `1fsnxchk_agentic_timeline.json` fixture, `path_decode.py` fiber-consistency hunks, `2026-07-17-instance-selection-arbiter-design.md`). Why parked, not landed: reviewed in PRs #11/#12 but those merged into a side branch; main's scorer has since been changed by WS0 deinflation + the A2 `core.timebase.Trajectory` refactor, so the pinned BB12 figure and the path_decode hunks need re-validation before landing. Decision needed: re-land the provenance-guard concept against today's scorer, or record as closed. Also holds (via `align-f0-scorer`) two review-dropped files: `looptrace/NOTES.md`, `results/bb_baselines_placement.json` — keep the branch tips tagged and nothing is lost.
2. **`work-grouping-proposal`** (branch kept, remote exists). Preserve: `docs/work_grouping_proposal.md`, `scripts/propose_work_grouping.py`, `labeling/fixtures/work_map.json`. DRY-RUN version-sibling grouping; potentially subsumed by the Crush path/identity root-cause plan (which builds work maps via `scripts/build_work_map.py` on trm-ablation-framework). Decision needed: reconcile the two work-map approaches, then land one or close both.
3. **`e1-flywheel` worktree (dirty, 11 modified + 1 untracked)**. Preserve before any cleanup: `docs/agent_handoff_e1_flywheel_20260719.md` (untracked) and uncommitted edits to `agentic/{belief,live_runners,loop}.py`, `infer.py`, `lyrics_align.py`, `trajectory/pseudo_materialize.py` + tests, EXPERIMENTS.md. This is live E1 session state that never became commits. Suggested: commit to the branch as a WIP checkpoint (or `git stash store` + note), then decide against PR #35's final state.
4. **`e1-hubert-corroborate` worktree extras** (branch is LAND via PR #35, but the worktree carries uncommitted work): untracked `scripts/persist_e1_pool.py`, `alignment/docs/e1_session_state.md`, plus edits to `lyrics_align.py`, `mert_store.py`, `set_mert_backfill_loop.py`, a test. Commit onto the PR branch or park explicitly before rebasing #35.
5. **Root repo**: uncommitted `labeling/fixtures/bb12_ground_truth.yaml` modification — commit onto trm-ablation-framework before the Wave-3 split (goes with PR-A).

## Recommended execution order

0. **Safety net first (cheap, reversible):** before deleting anything, tag every branch tip: `for b in $(git for-each-ref --format='%(refname:short)' refs/heads/); do git tag "archive/$b" "$b"; done`. Deletion then only removes names, never content.
1. Commit-or-park the uncommitted state in the 4 dirty worktrees that matter (root, e1-flywheel, e1-hubert-corroborate, acap-instance-separability); grab the 2 small saves (bb12-inventory-repair, eda-source-resolve).
2. Merge Wave 1 PRs in order: #55 → (#34/#37 decision) → #35 (fix its one-file conflict) → #56.
3. Open + merge Wave 2 PRs (tie-break, lab backfill, identity-miss artifacts, pws_aligner).
4. Split trm-ablation-framework into PR-A…PR-D (Wave 3); PR-A after #56.
5. Delete the 31 DISCARD branches; remove the 19+2 worktrees; free `alignment-data-integrity` and fast-forward local `main` to `origin/main` last.
6. Also delete the stale remote copies of retired branches (`origin/pws-phase1b-continuous`, `origin/align-f0-scorer-clean`, `origin/cotrain-grammar-coverage` once reconcile-handoff-doc's fate is decided).

**User decisions required:** (a) #34 vs #37 merge strategy; (b) land vs park `align-pwsv4-transitions` given v4 promotion is BB11-gated (recommendation: land as workspace code); (c) fate of the parked provenance-guard suite; (d) work-grouping proposal vs Crush work-map path; (e) whether e1-flywheel's uncommitted session state is still wanted post-#35.
