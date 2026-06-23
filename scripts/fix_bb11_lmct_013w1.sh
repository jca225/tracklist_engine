#!/usr/bin/env bash
# BB11 slot 013w1 — register acapella candidate + refresh aligning folder.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SET_DIR="$HOME/aligning/2nvzlh2k__Two Friends - Big Bootie Mix Episode 11"
STEM='013w1__DJ Kool - Let Me Clear My Throat [103bpm 10A]'
CAND_DIR="$SET_DIR/stems/$STEM/candidates"
CAND="$CAND_DIR/vocals/cand1__DJ Kool - Let Me Clear My Throat (Acappella).__Sergio Iván Ríos Muñiz.m4a"
PY="$REPO/venvs/audio/bin/python"

if [[ ! -f "$CAND" ]]; then
  echo "Missing candidate: $CAND" >&2
  exit 1
fi

cat > "$CAND_DIR/WINNER.txt" <<EOF
vocals/cand1__DJ Kool - Let Me Clear My Throat (Acappella).__Sergio Iván Ríos Muñiz.m4a
1qrzf9p
acappella
EOF

echo "==> ingest acapella sibling on pi-storage"
"$PY" "$REPO/scripts/ingest_stem_url.py" \
  --file "$CAND" \
  --track-id 1qrzf9p \
  --role acappella \
  --set-id 2nvzlh2k \
  --position 013w1 \
  --promote \
  --reason 'quality:human_pick|identity:acapella|note:BB11 013w1 LMCT payload' \
  --pull

echo "==> re-check inventory"
make -C "$REPO" check-inventory SET=2nvzlh2k
