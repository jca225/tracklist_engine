#!/usr/bin/env bash
# SessionStart hook — collision forecast (braid) + where this agent is standing.
#
# The braid half was already here. The posture half exists because the collision
# report told agents which *files* were contested but never that they were about
# to work in the shared primary checkout, or that fourteen other branches were
# already open and unlanded. Once per session, not once per edit — a warning on
# every write is noise, and noise is how the last set of rules got ignored.
set -uo pipefail

command -v braid >/dev/null 2>&1 && braid status 2>/dev/null || true

root="${CLAUDE_PROJECT_DIR:-$PWD}"
branch=$(git -C "$root" rev-parse --abbrev-ref HEAD 2>/dev/null) || exit 0
gitdir=$(git -C "$root" rev-parse --path-format=absolute --git-dir 2>/dev/null)
common=$(git -C "$root" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)

echo
echo "git posture:"
if [ "$branch" = "main" ] || [ "$branch" = "master" ]; then
	echo "  ! HEAD is '$branch' — file edits are BLOCKED by the guard-worktree hook."
	echo "    Start ritual (CLAUDE.md \"Git workflow\"):"
	echo "      git worktree add -b <type>/<slug> .claude/worktrees/<slug> origin/main"
elif [ "$gitdir" = "$common" ]; then
	echo "  ! primary checkout, on '$branch' — substantive work belongs in a worktree"
	echo "    (AGENTS.md §1): git worktree add -b <type>/<slug> .claude/worktrees/<slug> origin/main"
else
	echo "  ok worktree on '$branch'"
fi

unlanded=0
while IFS= read -r b; do
	[ -n "$b" ] || continue
	n=$(git -C "$root" rev-list --count "main..$b" 2>/dev/null || echo 0)
	[ "$n" -gt 0 ] 2>/dev/null && unlanded=$((unlanded + 1))
done < <(git -C "$root" for-each-ref --format='%(refname:short)' refs/heads/ 2>/dev/null)

if [ "$unlanded" -gt 1 ]; then
	echo "  $unlanded unlanded local branches — AGENTS.md §2b: one open PR per agent,"
	echo "    finish before starting, land smallest first. Sweep: make gc-branches"
fi
exit 0
