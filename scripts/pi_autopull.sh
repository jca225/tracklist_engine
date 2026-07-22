#!/usr/bin/env bash
#
# pi_autopull.sh — keep a cluster pi's checkout in sync with origin/main WITHOUT
# clobbering local work. Driven by the tracklist-autopull.timer systemd unit.
#
# This is the mechanical fix for issue #73 ("Deploy discipline: treat pi-storage
# as immutable canonical infra"). The recurring failure it prevents: a pi drifts
# N commits behind origin/main (because `make deploy` is manual and nobody runs
# it), and/or accumulates stranded WIP that makes the next `git pull` abort.
#
# Design invariants:
#   - NEVER clobber. Refuse to pull if the tree has uncommitted *tracked* changes
#     (parallel WIP) or if HEAD has diverged from origin (local commits). Log and
#     exit 0 so the timer stays green; a human resolves the divergence.
#   - Untracked files (?? — e.g. .wt/ worktrees, scratch outputs) do NOT block a
#     sync by themselves. If one would collide with an incoming tracked file the
#     ff-merge fails loudly (exit 1) rather than silently — that's the signal to
#     clean it up by hand, exactly as we did on 2026-07-22.
#   - Fast-forward only. No merge commits, no rebases on the canonical box.
#   - Restart ONLY tracklist-jobqueue (safe, ~2s), and only when code actually
#     changed and the unit exists+is active on this host. Never auto-touch the
#     ajax-retry drain or the scraper — those are disruptive (see cluster-deploy
#     skill / CLAUDE.md).
#
set -uo pipefail

REPO="${TRACKLIST_REPO:-$HOME/tracklist_engine}"
BRANCH="${TRACKLIST_BRANCH:-main}"
PIP="$REPO/venvs/web_crawler/bin/pip"

cd "$REPO" || { echo "[autopull] FATAL: $REPO not found"; exit 1; }

# %z (not GNU-only `-Is`) so the timestamp works on BSD/macOS workers too.
log() { echo "[autopull $(date +%Y-%m-%dT%H:%M:%S%z)] $*"; }

# --- No-clobber guard: refuse on uncommitted TRACKED changes ------------------
if ! git diff --quiet || ! git diff --cached --quiet; then
  log "SKIP: uncommitted tracked changes present — refusing to pull (issue #73)"
  git status --porcelain=v1 | grep -vE '^\?\?' | sed 's/^/[autopull]   /'
  exit 0
fi

git fetch -q origin "$BRANCH" || { log "WARN: fetch failed"; exit 0; }
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse "origin/$BRANCH")

if [ "$LOCAL" = "$REMOTE" ]; then
  log "up-to-date at ${LOCAL:0:9}"
  exit 0
fi

# --- Diverged guard: only fast-forward -----------------------------------------
if ! git merge-base --is-ancestor "$LOCAL" "$REMOTE"; then
  log "SKIP: HEAD ${LOCAL:0:9} is not an ancestor of origin/$BRANCH — diverged, refusing (issue #73)"
  exit 0
fi

# --- Migration/schema guard: HOLD (don't pull) to keep schema+code coordinated -
# A change to a DB migration or the canonical schema requires a deliberate,
# human-run migrate+deploy (migrations are not idempotent — see scripts/CLAUDE.md).
# Auto-fast-forwarding here would land new code on the canonical box before the
# schema it expects is applied. So we stop the line and wait for a coordinated
# deploy (issue #73 / Crush Phase B GATE) rather than sync blindly.
MIGRATION_CHANGED=$(git diff --name-only "$LOCAL" "$REMOTE" -- \
  scripts/migrations web_crawler/database/schema.sql)
if [ -n "$MIGRATION_CHANGED" ]; then
  log "MIGRATION PENDING — holding at ${LOCAL:0:9}; run a coordinated migrate+deploy by hand:"
  printf '%s\n' "$MIGRATION_CHANGED" | sed 's/^/[autopull]   /'
  exit 0
fi

log "syncing ${LOCAL:0:9} -> ${REMOTE:0:9}"
# Inspect the incoming range BEFORE moving HEAD.
REQ_CHANGED=$(git diff --name-only "$LOCAL" "$REMOTE" -- requirements.txt | head -1)
CODE_CHANGED=$(git diff --name-only "$LOCAL" "$REMOTE" -- '*.py' | head -1)

if ! git merge --ff-only "origin/$BRANCH"; then
  log "ERROR: fast-forward failed (untracked-file collision or unexpected state) — needs a human"
  exit 1
fi

if [ -n "$REQ_CHANGED" ]; then
  log "requirements.txt changed — pip install"
  "$PIP" install -q -r requirements.txt || log "WARN: pip install returned non-zero"
fi

# --- Restart jobqueue only (safe), only if it runs here and code changed --------
if [ -n "$CODE_CHANGED" ] && systemctl cat tracklist-jobqueue.service &>/dev/null; then
  if systemctl is-active --quiet tracklist-jobqueue.service; then
    log "code changed — restarting tracklist-jobqueue"
    if sudo -n systemctl restart tracklist-jobqueue.service; then
      log "jobqueue restarted"
    else
      log "WARN: jobqueue restart failed (passwordless sudo not available?)"
    fi
  fi
fi

log "done at $(git rev-parse --short HEAD)"
