#!/usr/bin/env python3
"""Branch/worktree garbage collection — the finish half of the git protocol.

The measured failure mode in this repo is not hard merges; it is branches that
get started and never finished. At the time this was written there were 36 local
branches, 20 of them **zero commits ahead of `main`** — fully landed work whose
branch nobody deleted — plus worktrees idle for 9 days holding a branch hostage
(a checked-out branch cannot be parked) and costing 4-15 GB of disk each.

Nothing here is clever. It is the sweep nobody owns, given a name:

    make gc-branches         # report only
    make gc-branches-apply   # delete the already-merged branches

**Safety.** The only destructive action is `git branch -d` (never `-D`) on
branches classified MERGED, which requires `git rev-list --count main..<branch>`
to be exactly 0 — i.e. every commit on the branch is already reachable from
`main`. Unlanded work is reported, never touched. Branches checked out in a
worktree are never deleted (git would refuse anyway); they are reported as HELD
so you know to `git worktree remove` first.

The classification is a pure function (`classify`) so the rules are testable
without a git fixture — see tests/test_gc_branches.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Every git call runs from this file's own directory. git resolves the enclosing
# repo (or worktree) itself, so there is no hardcoded parents[N] depth to rot
# when this script moves — and, unlike a fixed repo root, it stays correct when
# the script is invoked from inside one of the .claude/worktrees/ checkouts.
_SCRIPT_DIR = Path(__file__).resolve().parent

# Branches that are never candidates for anything, whatever their state.
PROTECTED = frozenset({"main", "master", "HEAD"})

# A branch untouched for this long is a park candidate. Age is only ever used to
# *suggest* — never to delete. (Unlike the WIP fence, which deliberately ignores
# age because a pre-push hook has no trustworthy branch-creation timestamp, here
# we have real commit dates from the branch tip.)
STALE_DAYS = 3

MERGED = "MERGED"  # every commit already in main -> safe to delete
HELD = "HELD"  # merged, but checked out in a worktree -> remove the worktree
STALE = "STALE"  # unlanded and idle -> land it or park it
ACTIVE = "ACTIVE"  # unlanded and recent -> leave it alone
PROTECTED_STATE = "PROTECTED"


@dataclass(frozen=True)
class BranchState:
    """Everything the classifier needs, so it can be built in a test."""

    name: str
    ahead: int  # commits on this branch not on main
    checked_out: bool  # occupies a worktree
    days_idle: int  # since the tip commit


def classify(state: BranchState, stale_days: int = STALE_DAYS) -> str:
    """Bucket one branch. Pure — no git, no filesystem.

    Order matters: protection beats everything, and "checked out" beats "merged"
    because a checked-out branch cannot be deleted (or parked) until its
    worktree is gone.
    """
    if state.name in PROTECTED:
        return PROTECTED_STATE
    if state.ahead == 0:
        return HELD if state.checked_out else MERGED
    if state.days_idle >= stale_days:
        return STALE
    return ACTIVE


def _git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=_SCRIPT_DIR,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def _checked_out_branches() -> set[str]:
    """Branches occupying a worktree (including the primary checkout)."""
    out: set[str] = set()
    for line in _git("worktree", "list", "--porcelain").splitlines():
        if line.startswith("branch "):
            out.add(line.split(" ", 1)[1].strip().removeprefix("refs/heads/"))
    return out


def collect_states(base: str = "main") -> list[BranchState]:
    """Read every local branch's state from git."""
    checked_out = _checked_out_branches()
    fmt = "%(refname:short)%09%(committerdate:unix)"
    rows = [
        r
        for r in _git("for-each-ref", "--format", fmt, "refs/heads/").splitlines()
        if r
    ]
    now = int(time.time())

    states: list[BranchState] = []
    for row in rows:
        name, _, ts = row.partition("\t")
        try:
            ahead = int(_git("rev-list", "--count", f"{base}..{name}").strip())
        except RuntimeError:
            continue  # unreachable base (fresh clone) -> skip rather than crash
        days = max(0, (now - int(ts)) // 86400) if ts else 0
        states.append(
            BranchState(
                name=name,
                ahead=ahead,
                checked_out=name in checked_out,
                days_idle=days,
            )
        )
    return states


def idle_worktrees() -> list[tuple[str, str]]:
    """(path, branch) for worktrees whose branch has nothing unlanded.

    These are pure cost: the work is in `main`, the disk is not.
    """
    entries: list[tuple[str, str]] = []
    path = ""
    for line in _git("worktree", "list", "--porcelain").splitlines():
        if line.startswith("worktree "):
            path = line.split(" ", 1)[1].strip()
        elif line.startswith("branch "):
            branch = line.split(" ", 1)[1].strip().removeprefix("refs/heads/")
            entries.append((path, branch))
    primary = _git("rev-parse", "--show-toplevel").strip()
    out = []
    for wt_path, branch in entries:
        if wt_path == primary:
            continue
        try:
            ahead = int(_git("rev-list", "--count", f"main..{branch}").strip())
        except RuntimeError:
            continue
        if ahead == 0:
            out.append((wt_path, branch))
    return out


def _report(states: list[BranchState], stale_days: int) -> dict[str, list[BranchState]]:
    buckets: dict[str, list[BranchState]] = {
        MERGED: [],
        HELD: [],
        STALE: [],
        ACTIVE: [],
        PROTECTED_STATE: [],
    }
    for s in states:
        buckets[classify(s, stale_days)].append(s)
    return buckets


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--apply",
        action="store_true",
        help="delete MERGED branches (git branch -d; never -D, never unlanded work)",
    )
    ap.add_argument("--stale-days", type=int, default=STALE_DAYS)
    args = ap.parse_args(argv)

    try:
        states = collect_states()
    except RuntimeError as exc:
        print(f"gc-branches: {exc}", file=sys.stderr)
        return 1

    buckets = _report(states, args.stale_days)

    merged = sorted(buckets[MERGED], key=lambda s: s.name)
    held = sorted(buckets[HELD], key=lambda s: s.name)
    stale = sorted(buckets[STALE], key=lambda s: -s.days_idle)
    active = sorted(buckets[ACTIVE], key=lambda s: -s.ahead)

    if merged:
        verb = "deleting" if args.apply else "deletable"
        print(f"\n{verb} — already merged into main ({len(merged)}):")
        for s in merged:
            print(f"  {s.name}")
    if held:
        print(f"\nheld by a worktree — merged but not deletable ({len(held)}):")
        for s in held:
            print(f"  {s.name}   (git worktree remove <path> first)")
    if stale:
        print(
            f"\nunlanded and idle ≥{args.stale_days}d — land or `braid park` ({len(stale)}):"
        )
        for s in stale:
            print(f"  {s.name}   +{s.ahead}  {s.days_idle}d idle")
    if active:
        print(f"\nunlanded, recent ({len(active)}):")
        for s in active:
            print(f"  {s.name}   +{s.ahead}  {s.days_idle}d idle")

    for wt_path, branch in idle_worktrees():
        print(f"\nidle worktree (nothing unlanded): {wt_path}  [{branch}]")
        print("  git worktree remove " + wt_path)

    deleted = 0
    if args.apply and merged:
        for s in merged:
            try:
                _git("branch", "-d", s.name)
                deleted += 1
            except RuntimeError as exc:
                print(f"  ! kept {s.name}: {exc}", file=sys.stderr)
        print(f"\ndeleted {deleted} merged branch(es).")
    elif merged:
        print("\nrun `make gc-branches-apply` to delete the merged ones.")

    unlanded = len(stale) + len(active)
    if unlanded > 1:
        print(
            f"\n{unlanded} unlanded branches open. AGENTS.md §2b: one open PR per "
            f"agent — finish before starting, land smallest first."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
