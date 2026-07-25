#!/bin/bash
# Strict static-type gate (Phase-2 Tier-2 "types as lightweight proofs").
#
# Runs `mypy --strict` over the entire core/ package — the substrate the whole
# pipeline composes on. This is a RATCHET: core/ is clean and must stay clean.
# Extend coverage to the chain modules (ingest/, tokenizer/, labeling/) by adding
# their paths here once each passes --strict.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# Resolve a python with mypy. Prefer this checkout's venv; in a linked worktree
# $ROOT has no venvs/, so fall back to the MAIN worktree's venv (via the shared
# git common dir); then bare python3 (the CI path, where mypy is pip-installed).
# Honor an inherited $PYTHON (e.g. from the pre-commit hook) if it is executable.
if [[ -z "${PYTHON:-}" || ! -x "${PYTHON:-}" ]]; then
  COMMON_DIR="$(git rev-parse --git-common-dir)"
  case "$COMMON_DIR" in
    /*) ;;
    *) COMMON_DIR="$ROOT/$COMMON_DIR" ;;
  esac
  MAIN_ROOT="$(cd "$(dirname "$COMMON_DIR")" && pwd)"
  PYTHON=python3
  for cand in "${ROOT}/venvs/audio/bin/python" "${MAIN_ROOT}/venvs/audio/bin/python"; do
    if [[ -x "$cand" ]]; then
      PYTHON="$cand"
      break
    fi
  done
fi

exec "$PYTHON" -m mypy --strict core/
