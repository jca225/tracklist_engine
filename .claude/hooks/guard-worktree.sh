#!/usr/bin/env bash
# PreToolUse guard — refuse file mutations while HEAD is on `main`.
#
# AGENTS.md §1 says "one agent, one worktree", but nothing enforced it, and the
# measured result was agents editing the shared primary checkout and sweeping
# each other's staged work into their commits. This is the one hard stop: you may
# not modify a tracked file while the branch you would commit it to is `main`.
# Everything else in the protocol stays advisory.
#
# Scope: only this repo (matched by git common-dir, so all its worktrees count
# and other projects on the machine do not). Non-git paths pass through.
# Escape hatch: TRACKLIST_ALLOW_MAIN_EDIT=1 — for the rare deliberate hotfix.
#
# Exit 0 = allow, exit 2 = block (stderr is fed back to the agent).
set -uo pipefail

payload=$(cat)

target=$(
	printf '%s' "$payload" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
ti = d.get("tool_input") or {}
print(ti.get("file_path") or ti.get("notebook_path") or d.get("cwd") or "")
' 2>/dev/null
)
[ -n "$target" ] || exit 0

# A Write can name a file that does not exist yet — walk up to a real directory.
dir="$target"
[ -d "$dir" ] || dir=$(dirname "$dir")
while [ ! -d "$dir" ] && [ "$dir" != "/" ] && [ "$dir" != "." ]; do
	dir=$(dirname "$dir")
done
[ -d "$dir" ] || exit 0

branch=$(git -C "$dir" rev-parse --abbrev-ref HEAD 2>/dev/null) || exit 0
case "$branch" in
main | master) ;;
*) exit 0 ;;
esac

# Same repository? Worktrees share a git common-dir with the primary checkout.
root="${CLAUDE_PROJECT_DIR:-$(dirname "$(dirname "$(dirname "$0")")")}"
ours=$(git -C "$root" rev-parse --path-format=absolute --git-common-dir 2>/dev/null) || exit 0
theirs=$(git -C "$dir" rev-parse --path-format=absolute --git-common-dir 2>/dev/null) || exit 0
[ "$ours" = "$theirs" ] || exit 0

[ "${TRACKLIST_ALLOW_MAIN_EDIT:-}" = "1" ] && exit 0

cat >&2 <<EOF
BLOCKED — '$target' is in a checkout whose HEAD is '$branch'.

Work never lands on $branch directly here (AGENTS.md §1, CLAUDE.md "Git workflow").
Start the work properly first, from the main checkout root:

  git worktree add -b <type>/<slug> .claude/worktrees/<slug> origin/main
  ln -s "\$PWD/venvs" .claude/worktrees/<slug>/venvs
  make collide

then make this edit inside that worktree. If you only need a branch (no
worktree): git switch -c <type>/<slug>.

Deliberate exception: re-run with TRACKLIST_ALLOW_MAIN_EDIT=1.
EOF
exit 2
