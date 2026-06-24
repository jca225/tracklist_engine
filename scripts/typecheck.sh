#!/bin/bash
# Strict static-type gate (Phase-2 Tier-2 "types as lightweight proofs").
#
# Runs `mypy --strict` over the type-clean core subset. This is a RATCHET: a file
# is added here only once it passes clean, and never removed. Expand the list as
# the remaining core modules are brought to --strict — currently NOT yet clean:
#   core/db.py, core/slot_inventory.py, core/acquisition_case.py
# (23 errors total as of this gate; tracked as a Phase-2 follow-up).
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

PYTHON="${ROOT}/venvs/audio/bin/python"
[[ -x "$PYTHON" ]] || PYTHON=python3

exec "$PYTHON" -m mypy --strict \
  core/identity.py \
  core/result.py \
  core/audio_resolve.py \
  core/errors.py \
  core/models.py
