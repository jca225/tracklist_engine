#!/bin/bash
# Lint gate (ruff). Config lives in pyproject.toml [tool.ruff].
#
# Runs ruff over core/ only. This is a RATCHET, deliberately seeded narrow and
# matching scripts/typecheck.sh: core/ is the substrate everything composes on,
# so it is the one tree that must stay clean. Extend LINT_PATHS as each module
# is cleaned.
#
# Why so narrow: at the time this gate was added the repo had 659 ruff findings
# on the default rule set (alignment 220, pws_aligner 135, eda 109, scripts 60,
# labeling 31 …). Gating all of that at once would just mean `--no-verify`
# everywhere. Seventeen of those are F821 undefined-name — real NameErrors, not
# style — and are worth fixing on their own branch, not smuggled in here.
set -euo pipefail

LINT_PATHS=(core/)

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# Resolve a python with ruff. Same resolution ladder as scripts/typecheck.sh —
# duplicated rather than factored out on purpose: both files are gate entry
# points invoked by CI and .githooks/pre-commit, and a shared sourced helper is
# one more thing that can break the gate for every worktree at once. If a third
# gate script appears, factor it then.
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

# Prefer ruff in the resolved interpreter (venv locally, pip-installed in CI),
# then a ruff on PATH, then uvx with the version the editor extension ships.
# Never silently skip: a lint gate that no-ops when the tool is missing reads as
# green and gates nothing.
if "$PYTHON" -m ruff --version >/dev/null 2>&1; then
  exec "$PYTHON" -m ruff check "${LINT_PATHS[@]}"
elif command -v ruff >/dev/null 2>&1; then
  exec ruff check "${LINT_PATHS[@]}"
elif command -v uvx >/dev/null 2>&1; then
  exec uvx ruff@0.15.19 check "${LINT_PATHS[@]}"
else
  echo "lint.sh: ruff not found." >&2
  echo "  install it into the venv:  $PYTHON -m pip install ruff" >&2
  echo "  (CI gets it from requirements-ci.txt)" >&2
  exit 1
fi
