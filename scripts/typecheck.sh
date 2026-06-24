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

PYTHON="${ROOT}/venvs/audio/bin/python"
[[ -x "$PYTHON" ]] || PYTHON=python3

exec "$PYTHON" -m mypy --strict core/
