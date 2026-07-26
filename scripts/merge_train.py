#!/usr/bin/env python3
"""Merge train — the missing flow controller for the branch fleet.

The repo has a strong *gate* (branch protection + CI) but no *flow control*:
branches pile up, diverge, and turn into merge pain (three agents once re-did the
same RT1 measurement because nobody could see the others). braid forecasts
collisions but does not act. This is the serialized integrator that decides what
lands next and in what order.

v1 is deliberately **read-only** (``--plan``): it fetches, asks braid for the
landable set + conflict order, and prints the one branch it would land next and
the exact steps — it mutates nothing. ``--land-next`` exists as the interface but
is **gated**: landing to ``main`` fast-pulls to production on both pis within
~10 min (tracklist-autopull.timer), so the actual lander is withheld pending the
deploy-routing decision (a deploy-sensitive label / manual confirm). Run it and
you get a plan, not a surprise deploy.

Order = braid's conflict-size ascending (fewest conflicting files first), READY
before NOT-READY. Dependency-first ordering (e.g. a base before its dependents)
is the operator's call in v1 — pass ``--first <branch>`` to force one to the top.

Usage:
    venvs/audio/bin/python scripts/merge_train.py --plan
    venvs/audio/bin/python scripts/merge_train.py --plan --first fix/foo
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass

# braid + git are local; still bound every call (timeout + text) so a hung ssh/
# lock can't wedge the train and output is decoded, not bytes.
_TIMEOUT = 30
_TARGET = "main"
# Never offer these as train candidates.
_SKIP_PREFIXES = ("parked/",)
_SKIP_EXACT = {_TARGET}


@dataclass(frozen=True)
class Candidate:
    branch: str
    ready: bool
    ahead: int  # commits ahead of origin/main
    conflict_files: int  # files conflicting with main (0 if none / unknown)
    behind: int  # commits behind main (0 if none / unknown)
    note: str  # first braid reason line, for the operator


def _run(args: list[str]) -> tuple[int, str]:
    """Run a local command read-only; return (exit_code, stdout+stderr)."""
    try:
        p = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return 127, str(e)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def _have_braid() -> bool:
    code, _ = _run(["braid", "--help"])
    return code == 0


def candidate_branches() -> list[str]:
    code, out = _run(
        ["git", "for-each-ref", "--format=%(refname:short)", "refs/heads/"]
    )
    if code != 0:
        return []
    heads = [b.strip() for b in out.splitlines() if b.strip()]
    return [
        b for b in heads if b not in _SKIP_EXACT and not b.startswith(_SKIP_PREFIXES)
    ]


def _ahead(branch: str) -> int:
    code, out = _run(["git", "rev-list", "--count", f"origin/{_TARGET}..{branch}"])
    try:
        return int(out.strip()) if code == 0 else 0
    except ValueError:
        return 0


def inspect_branch(branch: str) -> Candidate:
    """braid's landable verdict + parsed conflict/behind counts for one branch."""
    code, out = _run(["braid", "ready", branch])
    ready = code == 0 and "NOT READY" not in out
    conflict_files = 0
    behind = 0
    note = ""
    m = re.search(r"conflicts with \w+ in (\d+) file", out)
    if m:
        conflict_files = int(m.group(1))
    m = re.search(r"\((\d+) behind\)", out)
    if m:
        behind = int(m.group(1))
    for ln in out.splitlines():
        s = ln.strip()
        if s.startswith(("x ", "! ")) or s.startswith(("x\t", "!\t")):
            note = re.sub(r"\s+", " ", s.lstrip("x! \t"))
            break
    return Candidate(
        branch=branch,
        ready=ready,
        ahead=_ahead(branch),
        conflict_files=conflict_files,
        behind=behind,
        note=note,
    )


def plan_order(
    candidates: list[Candidate], first: str | None = None
) -> list[Candidate]:
    """Land order (pure): forced ``first`` (if present), then READY before
    NOT-READY, then fewest conflicting files, then fewest commits ahead (smaller
    blast radius), then name for stability."""

    def key(c: Candidate) -> tuple:
        return (
            0 if (first and c.branch == first) else 1,
            0 if c.ready else 1,
            c.conflict_files,
            c.ahead,
            c.branch,
        )

    return sorted(candidates, key=key)


def format_plan(ordered: list[Candidate], read_only: bool = True) -> str:
    lines: list[str] = []
    mode = "plan (read-only)" if read_only else "land-next (gated)"
    lines.append(f"merge train — {mode} · target={_TARGET}")
    lines.append("")
    if not ordered:
        lines.append("  no candidate branches (fleet empty or all parked).")
        return "\n".join(lines)

    lines.append(
        f"  {'':2} {'branch':40} {'ahead':>5} {'conf':>4} {'behind':>6}  verdict"
    )
    for i, c in enumerate(ordered):
        mark = "->" if i == 0 and c.ready else "  "
        verdict = "READY" if c.ready else "blocked"
        lines.append(
            f"  {mark} {c.branch:40} {c.ahead:>5} {c.conflict_files:>4} "
            f"{c.behind:>6}  {verdict}"
        )
        if c.note and not c.ready:
            lines.append(f"       └ {c.note[:88]}")

    nxt = next((c for c in ordered if c.ready), None)
    lines.append("")
    if nxt is None:
        lines.append(
            "  next: NOTHING landable — every branch needs a rebase/resolve first."
        )
        lines.append("  (rebase the smallest-conflict branch onto main, then re-run.)")
    else:
        lines.append(f"  next to land: {nxt.branch}")
        lines.append("  would run (NOT executed in --plan):")
        lines.append(f"    1. git checkout {nxt.branch}")
        lines.append(f"    2. git rebase origin/{_TARGET}")
        lines.append("    3. make check-fast")
        lines.append("    4. gh pr create --base main   (if no PR exists)")
        lines.append("    5. gh pr merge --squash --auto   (lands when CI is green)")
        lines.append("    6. STOP + report (land-one-and-stop until proven)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--plan",
        action="store_true",
        help="read-only: print what would land next and in what order (default)",
    )
    mode.add_argument(
        "--land-next",
        action="store_true",
        help="GATED: the real lander is withheld pending deploy-routing (main "
        "auto-deploys within ~10 min). Prints the plan and refuses.",
    )
    p.add_argument(
        "--first", default=None, help="force a branch to the top of the order"
    )
    p.add_argument("--no-fetch", action="store_true", help="skip git fetch")
    args = p.parse_args(argv)

    if not _have_braid():
        print(
            "merge_train: braid not found on PATH — it is the order/verdict source. "
            "Install braid (~/workspace/braid) or add it to PATH.",
            file=sys.stderr,
        )
        return 2

    if not args.no_fetch:
        _run(["git", "fetch", "origin", _TARGET])

    # --plan is the default read-only mode; --land-next is the gated opt-out.
    read_only = args.plan or not args.land_next

    inspected = [inspect_branch(b) for b in candidate_branches()]
    # A branch with 0 commits ahead of main has nothing to land — not a train
    # candidate. Keep the count so the fleet size stays honest.
    candidates = [c for c in inspected if c.ahead > 0]
    empty = len(inspected) - len(candidates)
    ordered = plan_order(candidates, first=args.first)
    print(format_plan(ordered, read_only=read_only))
    if empty:
        print(f"\n  ({empty} branch(es) with nothing to land omitted)")

    if args.land_next:
        print("")
        print(
            "--land-next is GATED: landing to main fast-pulls to production on both "
            "pis within ~10 min. The autonomous lander is withheld until deploy-"
            "sensitive routing (a manual-confirm label) is decided. Land the branch "
            "above by hand for now, or wire the routing first.",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
