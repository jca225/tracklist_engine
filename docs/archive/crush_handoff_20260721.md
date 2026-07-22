# Operation Crush — session handoff (2026-07-21)

State of the #49 reconcile after this session. For the full plan see
[reconcile_plan_20260721.md](reconcile_plan_20260721.md). **Safety net: all 45
pre-reconcile branch tips are preserved as `archive/<branch>` tags** — nothing
deleted below is unrecoverable (`git show archive/<name>` / `git branch x archive/<name>`).

## Done this session (all on `main`)

- **Wave 1 (partial):** #55 merged.
- **Wave 2 (complete):** #57 (identity-miss), #58 (pws-aligner phase-1b), #59
  (lab-backfill), #60 (earliest-instance tie-break) — all built, Fable-reviewed,
  merged.
- **Root-cause fix (bigger than the plan):** GT yaml silently drifted from the
  Ableton `.als` source of truth (the `013w3` bug, the #37 block, the bb12
  divergence were all this one class). Fixed by a **GT-derives-from-`.als` gate**:
  - **#61** — commit BB11 `.als` + manifest-free slot-label gate
    (`labeling/gt_als_gate.py`, wired into `scripts/guardrails.py` + pre-commit) +
    BB11 `013w3→013w1` fix.
  - **#62** — extend export to derive `RefSegment.mix_end_s` + `tempo_ratio`;
    regenerate BB12 from its canonical `.als` (`bb12_align.als`, folds in the
    prior hand-enrichment as *derived* output + adopts the correct slot scheme);
    gate `1fsnxchk`; re-baseline the 3 bb12 acquisition tests.
  - Follow-up filed: **issue #63** — `track_id` `2p25k23p` shared by two songs
    (identity-by-string class).
- **#37 + #34 CLOSED** — #37's audio round-trip "ruler" is broken (m4a decode,
  never passed); its purpose is served by the #61/#62 gate. Both archive-tagged.
- **Branch/worktree sweep:** local branches 49→9, remote ~48→14, ~23 worktrees
  removed. All deletions were landed/closed + archive-tagged.

## What remains (for the next session)

### 1. Wave 3 — split `trm-ablation-framework` (biggest; not started)
The current working branch carries **~124 genuinely-unlanded files (~23k lines)**
of the +61 commits above old main; the bottom ~24 already re-landed. Split into
focused PRs (see reconcile plan §Wave 3): **PR-A** labeling/als toolkit +
manifest reconciliation, **PR-B** learned identity verifier, **PR-C** acquisition
cascade, **PR-D** ops/docs. Do PR-A after the content-identity work (#56) since
both touch `labeling/als/`.
**Caveat:** `trm-ablation-framework` still has an OLD-scheme committed
`bb12_ground_truth.yaml`; it will CONFLICT with main's #62-derived bb12 on rebase
— take main's (the derived/canonical one).

### 2. Open PRs left deliberately (real in-progress work — decide, don't auto-close)
- **#56 `crush/content-identity`** (draft) — Operation Crush content-addressed
  identity (kill the path/slot guess-ladder). Substantive; the user's core work.
- **#35 `e1-hubert-corroborate`** — HuBERT↔lyrics corroboration; CONFLICTING +
  has uncommitted worktree WIP (`.claude/worktrees/e1-hubert-corroborate`).

### 3. Parked branches (unlanded, kept visible)
- `reconcile-handoff-doc` — the F0 timeline-provenance guard + BB12 regression
  pin; reviewed in #11/#12 but merged to a side branch, never to main. Needs
  re-validation against today's scorer before landing (scorer moved: WS0
  deinflation, A2 timebase). See memory `looptrace`/`ws0_scorer_deinflation`.
- `work-grouping-proposal` — DRY-RUN version-sibling grouping; overlaps the Crush
  path/identity root-cause plan. Decide vs #56's approach.

### 4. Dirty worktrees parked (uncommitted work — preserve before cleanup)
- `.claude/worktrees/e1-flywheel` (10 modified + handoff doc) — live E1 session
  state that never became commits.
- `.claude/worktrees/bb12-inventory-repair`, `.claude/worktrees/eda-source-resolve`
  — small post-merge tweaks; inspect then discard.

### 5. Loose ends
- 6 older remote-only branches left untouched (NOT archive-tagged, pre-session):
  `als-codec-extraction`, `discord-retry-hardening`, `eda-warp-prior`,
  `fix-pi-locale-ingest`, `pws-alignment-reframe`, `stem-ingest-autoaccept`.
  Assess before deleting.
- Local `main` is stale (`b612b08`); `origin/main` is authoritative. Fast-forward
  local main when convenient.
- The user's prior bb12 hand-WIP is preserved at tag `wip/bb12-enrichment-backup`.

## Fastest path to a fully clean main
Wave 3 (split trm-ablation-framework) is the last big chunk. Everything else is
decisions (#35/#56, the 2 parks) + minor loose ends. `make check` +
`scripts/guardrails.py` (now including the `.als` gate) enforce integrity on push.
