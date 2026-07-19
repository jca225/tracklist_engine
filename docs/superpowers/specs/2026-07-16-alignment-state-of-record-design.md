# Design — Alignment State-of-Record (durable agent orientation)

> Spec written 2026-07-16 @ `8efc2ee` on branch `pws-alignment-reframe`.
> Brainstormed from: "we are drifting as the codebase grows — build a durable
> means to keep coding agents aligned, correct, and fast."

## Problem

The repo is as much a **research log** as a codebase. Alignment knowledge
accumulates faster than it is consolidated, so every new agent session orients
against a slightly-wrong picture of the world. The user's own words:

> "misalignment happens heavily w.r.t. the alignment algorithm — we do not stay
> up to date with previous sessions' work that says what is currently the best
> solution and what we have currently settled on."

Measured drift signals (2026-07-16):

- `docs/` — 91 files / ~19.7k lines. No single doc is authoritative for
  *current state*.
- **~14 dated snapshots** carry "current state" between them:
  11 `agent_handoff_*_YYYYMMDD.md`, `alignment_bearings_20260712.md`,
  `alignment_status_corrections_20260711.md`, `handoff_pws_cotrain_20260716.md`.
  Each is stale the moment it is committed; each session spawns a new one.
- **No decision record exists.** There is nowhere that states "what is the
  aligner at its best right now" or "what we have settled on — build on this,
  don't re-litigate."
- 161 memory files; `MEMORY.md` index (~130 lines) loaded every session, some
  entries name files that may no longer exist.
- Cruft polluting search: stray duplicate worktree
  `.claude/worktrees/agent-a2950f06412c3c49f/` (duplicates ~half the tree);
  vendored `workspaces/msst_webui/ComfyUI` (99k-line file).

Existing partial antidotes (to generalize, not replace):

- `docs/alignment_status.md` — SSOT for **numbers**. Excellent stamp convention:
  dated + commit SHA, "this doc owns every headline number; others cite, do not
  restate; if stale, re-run — do not hand-edit."
- `workspaces/alignment_prototype/attic/EXPERIMENTS.md` — SSOT for **dead ends**.
- `docs/alignment_recharacterization.md` — the interpretive **frame** (3 axes:
  identity / placement / structure).

The missing piece is the SSOT for **current best solution + settled decisions**.

## Root cause

Three symptoms (wrong/stale info · slow/can't-find · scope drift) share one root
cause: **there is no compact, current, trusted map an agent reads to orient.**
This spec builds that map for the alignment algorithm — the north-star gate —
and the minimal machinery to keep it true.

## Goals / non-goals

**Goals**
- A single living record of the aligner's current best solution + settled
  decisions that a new session reads *first* and updates *last*.
- Stop the dated-handoff pile from growing.
- Keep it true durably via read-first placement, a write-last ritual, and
  mechanical staleness/existence warnings.
- Remove zero-risk search-polluting cruft.

**Non-goals (YAGNI — explicitly out of scope)**
- Doc-frontmatter lifecycle system.
- Memory reaper / automated memory consolidation.
- Per-module status dashboards.
- Top-level-directory allowlist / scope-drift linter.
- Any change to the aligner code itself.

## The artifact — `docs/alignment_state_of_record.md`

Undated filename **on purpose**: it is living, not a snapshot. Five sections.

1. **Stamp** — `as of <date> @ <SHA>`, same convention as `alignment_status.md`.
2. **Current best solution** — always *rewritten* (never appended). What the
   aligner is at its best right now, structured by the 3 axes (identity /
   placement / structure) from `recharacterization.md`. This is the thing every
   new session builds ON.
3. **Settled decisions — append-only log** — dated entries, each:
   *"settled on X because Y"* with status `SETTLED` | `SUPERSEDED-BY-#N`.
   Append-only preserves history; the status field makes "current" a filter.
   This is the "don't re-litigate" spine.
4. **Open fronts** — what is actively in flight / undecided. Replaces the
   "live:" section that today spawns a fresh handoff doc each session.
5. **Pointers, not restatements** — links to the SSOTs: numbers →
   `alignment_status.md`, dead ends → `attic/EXPERIMENTS.md`, frame →
   `recharacterization.md`. Explicitly forbids restating numbers/dead-ends here,
   so the record cannot contradict them.

**Interfaces (the contract):**
- *A new session reads:* §1 stamp (freshness), §2 current-best (what to build
  on), §4 open-fronts (what's live). ~2 minutes to full orientation.
- *A finishing session writes:* rewrite §2, append §3 if a decision was made,
  refresh §4, re-stamp §1.

## Durability mechanisms

### D1 — Read-first (placement)
The **first** pointer in the alignment section of root `CLAUDE.md` becomes:
"Before any alignment work, read `docs/alignment_state_of_record.md` — current
best solution + settled decisions." It precedes the existing status/
recharacterization pointers.

### D2 — Write-last (ritual) — `/align-checkpoint` skill
A small skill whose entire job at session end is the four writes above
(rewrite §2, append §3, refresh §4, re-stamp §1). **This replaces spawning a new
dated handoff doc** — the pile stops growing at the source. Chosen over a
documented-convention-only approach because agents are fanned out and a skill
makes "update the record" a reliable, single-command closing move.

### D3 — Can't silently rot (enforcement) — `guardrails.py` / `make check`
- **Staleness warning** (not a hard gate): if HEAD is more than N commits ahead
  of the record's stamp *counting only commits that touch aligner paths*, print
  a warning. Warning-not-gate is deliberate — a hard gate on a fast-moving
  research repo gets bypassed. (Threshold N to be set in the plan; start ~10.)
- **File-existence check**: every repo-relative path referenced in the record
  and in `MEMORY.md` must exist, else warn. Directly kills the top stale-recall
  failure (memory naming files that are gone).

## One-time migration

1. **Seed** `alignment_state_of_record.md`:
   - §2 current-best + §3 initial settled-decisions from
     `alignment_bearings_20260712.md`.
   - §4 open-fronts from `handoff_pws_cotrain_20260716.md`.
   - §5 pointers to status / EXPERIMENTS / recharacterization.
2. **Archive** into `docs/archive/` (moved, not deleted — history preserved):
   the 11 `agent_handoff_*` docs, `alignment_bearings_20260712.md`,
   `alignment_status_corrections_20260711.md`, `handoff_pws_cotrain_20260716.md`.
   Decision: **archive all ~14** dated snapshots, since nothing is destroyed.
3. `MEMORY.md` gets a single pointer to the record as the alignment orientation
   entrypoint.

## Free wins (independent, zero-risk)

- Delete the stray duplicate worktree `.claude/worktrees/agent-a2950f06412c3c49f/`.
- `.gitignore` the vendored `workspaces/msst_webui/ComfyUI` blob out of search
  (and confirm it is not otherwise imported by chain code).

## Testing / verification

- **Existence check** self-tests: seed a known-bad path, confirm the guardrail
  warns; remove it, confirm clean.
- **Staleness check**: simulate HEAD ahead of stamp, confirm warning fires.
- **Orientation smoke test**: a fresh agent, given only the record, can state
  the current best pipeline + top open front without reading other docs.
- `make check` stays green (warnings do not fail the build).

## Risks / costs

- **Opportunity cost vs. the Aug-1 aligner gate.** Mitigation: the artifact +
  migration is hours, not days; the enforcement is small; option-3 scope is cut.
- **The record itself rots if the write-last ritual is skipped.** Mitigation:
  D3 staleness warning is the backstop that surfaces neglect at push time.
- **Archived docs lose discoverability.** Mitigation: `docs/archive/` is moved
  not deleted, and remains greppable; the record's §3 log carries forward the
  load-bearing conclusions.

## Rollout order

1. Free wins (cruft deletion) — no dependencies.
2. Seed `alignment_state_of_record.md` + archive dated docs.
3. Wire D1 (CLAUDE.md pointer) + `MEMORY.md` pointer.
4. Build D2 (`/align-checkpoint` skill).
5. Add D3 checks to `guardrails.py`; verify `make check`.
