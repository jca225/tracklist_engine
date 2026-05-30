#!/bin/bash
# Cursor afterFileEdit — run guardrails when risky Python paths change.
set -euo pipefail

input=$(cat)
file_path=$(echo "$input" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('file_path',''))" 2>/dev/null || true)

if [[ -z "$file_path" ]]; then
  exit 0
fi

case "$file_path" in
  */analysis/adapters/*.py|*/ingest/adapters/*.py|*/tokenizer/*.py|*/ingest/*.py|*/labeling/*.py|*/scripts/guardrails.py)
    ;;
  *)
    exit 0
    ;;
esac

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PYTHON="${ROOT}/venvs/audio/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi

if output=$("$PYTHON" scripts/guardrails.py 2>&1); then
  exit 0
fi

python3 -c "
import json, sys
print(json.dumps({'additional_context': 'Guardrails failed after edit:\n' + sys.argv[1]}))
" "$output"
exit 0
