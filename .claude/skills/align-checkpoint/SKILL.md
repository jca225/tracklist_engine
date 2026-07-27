---
name: align-checkpoint
description: Update docs/alignment_state_of_record.md at the end of an alignment work session — the write-last ritual that keeps the aligner's current-best + settled-decisions current, replacing the old dated-handoff pile. Use at the end of any session that changed the aligner's best approach, settled a decision, or moved an open front; when the user says "checkpoint the aligner", "update the state of record", "close out this alignment session", or before handing off alignment work. Do NOT use for numbers (those go to alignment_status.md) or dead ends (those go to the EXPERIMENTS ledger).
---

# Align Checkpoint — the write-last ritual

Keeps [docs/alignment_state_of_record.md](../../docs/alignment_state_of_record.md)
current so the next session orients from truth instead of a stale snapshot.
**This replaces writing a new `docs/*handoff*_YYYYMMDD.md`** — do not spawn one.

## When to run
At the end of an alignment session that did any of:
- changed the **current best approach** on an axis (identity / placement / structure),
- **settled** a decision (or superseded a prior one),
- opened / closed / advanced an **open front**.

If none of those happened, you don't need to checkpoint (a pure read/debug session
that changed nothing leaves the record alone).

## The four writes (do all that apply, in order)

1. **§1 Current best solution — REWRITE in place** (never append). State the
   aligner at its best *right now*, per axis. Qualitative pipeline shape only —
   **no accuracy numbers** (those live in `alignment_status.md`; cite it).
2. **§2 Settled decisions — APPEND at the top** if a decision was reached. New
   entry: `#N — <one-line> · <date> · SETTLED`. If it overrides an earlier one,
   flip that earlier entry's status to `SUPERSEDED-BY-#N` (don't delete it).
3. **§3 Open fronts — REFRESH.** Remove what's now closed/settled, add what's now
   live, update what moved. This is where the next session looks for "what's in
   flight" — it replaces the handoff "LIVE STATE" section.
4. **§0 stamp — RE-STAMP** the header: `As of <YYYY-MM-DD> @ <short-SHA>` using
   today's date and current `git rev-parse --short HEAD`.

## Boundaries (what does NOT go in the record)
- **Numbers** → `docs/alignment_status.md` (regen from scorers; the record cites it).
- **Dead ends** → `alignment/attic/EXPERIMENTS.md`.
- **Live cluster/box ops** (teardown SQL, tmux recovery) → these are operational,
  not strategic state; keep them in an ops handoff if needed, not the record.

## Verify
- `make check` stays green (the guardrail warns, never blocks, on stamp drift or
  dead paths — fix dead paths the record references before committing).
- Commit the record update with the aligner change it describes (or on its own):
  `docs(state): checkpoint alignment state-of-record`.
