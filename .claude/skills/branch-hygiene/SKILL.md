---
name: branch-hygiene
description: Forecast branch collisions and keep landings cheap in a repo with many parallel agent worktrees. Check what will conflict before starting or landing work, find files multiple branches are fighting over, pick a landing order, and verify a merge did not produce broken code. Use before opening a new worktree, before landing a branch, when a rebase is painful, or when deciding what to merge next. Triggers on "what will conflict", "can I land this", "merge order", "rebase is a mess", "is this branch too big", "start a new worktree", "after merging".
---

# Branch Hygiene

This repo routinely has 8–12 branches and up to 9 worktrees open at once, almost
all agent-authored. Merge cost here does **not** track calendar age — it tracks
**commits since divergence**. A branch can be four days old and already
expensive.

Measured on this repo: branches under ~10 commits ahead conflicted in 0–2 files;
branches over ~15 conflicted in 9–28.

Tool: `braid` (installed via `uv tool install --editable ~/workspace/braid`).
Everything below is read-only — safe to run with dirty worktrees mid-flight.

## Before starting work in a new worktree

```bash
make collide
```

Look at the **contention hotspots** section. If the file you are about to change
is already contested by two or more branches, say so before starting, and either:

- pick a different slice of work, or
- land one of the competing branches first, or
- flag that the file needs splitting (see below).

## Before landing a branch

```bash
make land-budget          # is this branch already expensive?
make collide              # what exactly will it fight with?
```

**Land smallest first.** Every commit that reaches the target branch makes every
other open branch worse. A branch that would conflict in one file today
conflicts in twenty-eight after it sits for a week of agent commits.

If a branch is over budget (>15 commits ahead), do not just push through the
rebase. Split it: land the mechanical or additive part first, keep the
behavioural part on the branch. This is the same phase-split rule as
`refactor-safety`, applied to landing rather than refactoring.

## After any merge or rebase

```bash
make land-verify
```

Structured merge drivers (`mergiraf`, `weave`) resolve conflicts automatically
that git would have asked a human about. They are usually right. They are
sometimes silently wrong: measured across five open-source repos, weave 0.3.6
produced clean merges whose output **does not parse** in ~2% of cases on Python.

`braid verify` parse-checks the changed files. It catches the case where a merge
succeeded and the code is broken. It is cheap; run it every time.

This is already automatic for `git merge` and `git pull` via
[.githooks/post-merge](../../../.githooks/post-merge) (advisory, never blocks).
**git has no post-rebase hook**, so after a rebase you must run
`make land-verify` yourself — which is the case that matters most, since this
repo rebases far more than it merges.

## Finishing beats merging (the actual bottleneck)

Measured on this repo: branches that land, land in a **median of 2 hours at 7
commits**, and CI takes **2 minutes**. Nothing in the toolchain is slow. But at
last check 14 of 15 open branches had not been touched in days.

Fat branches are not caused by hard merges. They are caused by branches sitting
while `main` moves. `reconcile-handoff-doc` reached +53 commits by being left
for eight days, not by being difficult.

So the rule is: **finish things before starting things.** A branch is worth
finishing or worth parking; leaving it is the only option that is strictly
negative.

```bash
braid stale --days 3      # what nobody has touched
braid park <branch>       # keep the commits as a tag, drop the branch
braid parked              # what is parked
braid unpark <branch>     # bring it back
```

Parking turns a branch into an annotated tag and deletes the branch. The
commits are preserved exactly; the tag never needs rebasing and can never
conflict. It refuses to touch a branch that is checked out in a worktree.

**Order matters.** Before parking or removing anything, commit the dirty work in
that worktree — uncommitted changes are not captured by the tag, and idle
worktrees here routinely hold uncommitted edits. Then `git push origin
parked/<branch>` if you want it backed up; until you do it is local only.

Also: an idle worktree holds its branch hostage (it cannot be parked) and costs
real disk — several here are 4–15 GB. `git worktree remove <path>` when done.

## Reading the hotspot list

A file contested by three or more branches is a **design signal, not a merge
problem**. Multiple independent work streams keep needing to touch it, which
usually means it is doing too much.

Known standing hotspot in this repo:
`alignment/score_timeline_vs_gt.py` — contested by four
branches simultaneously as of 2026-07-25.

Do not "fix" a hotspot by resolving its conflicts faster. Raise it as a
refactoring candidate.

## What this does not do

- It does not merge anything, and it will not make real conflicts disappear.
  When two agents genuinely rewrote the same function, someone has to decide.
- `braid verify` checks that code **parses**, not that it works. `make check` is
  still the gate for behavior.
- `braid status` only sees local branches. Remote-only branches are invisible.
