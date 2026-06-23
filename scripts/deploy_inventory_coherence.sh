#!/usr/bin/env bash
# Deploy slot-inventory coherence to pi-storage (run from Mac repo root).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PI_DB="/mnt/storage/data/db/music_database.db"

echo "==> deploy (requires committed+pushed main on pi)"
make -C "$REPO" deploy-storage

echo "==> migrations"
ssh pi-storage "sqlite3 $PI_DB" < "$REPO/scripts/migrate_layer_role.sql"
ssh pi-storage "sqlite3 $PI_DB" < "$REPO/scripts/migrate_slot_satisfaction_view.sql"

echo "==> backfill layer_role"
ssh pi-storage "cd ~/tracklist_engine && venvs/web_crawler/bin/python scripts/backfill_layer_role.py --db $PI_DB"

echo "==> inventory check BB11"
make -C "$REPO" check-inventory SET=2nvzlh2k

echo "Done. Run scripts/fix_bb11_lmct_013w1.sh to ingest acapella winner."
