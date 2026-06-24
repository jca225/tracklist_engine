#!/usr/bin/env bash
# Wait for pwgrrb1 likes + playlists enrichment, then run analysis.
set -euo pipefail
cd "$(dirname "$0")/.."

DB="data/taste/taste_warehouse.db"
MIX="pwgrrb1"
TOTAL=2845
PY="venvs/audio/bin/python"
LOG="logs/murph_taste_post_enrich.log"
mkdir -p logs

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

likes_done() {
  sqlite3 "$DB" "SELECT json_array_length(checkpoint_json,'\$.completed_sc_user_ids') FROM scrape_checkpoints WHERE mix_id='$MIX' AND phase='enrich_likes';"
}

playlists_done() {
  sqlite3 "$DB" "SELECT COALESCE(json_array_length(checkpoint_json,'\$.completed_sc_user_ids'),0) FROM scrape_checkpoints WHERE mix_id='$MIX' AND phase='enrich_playlists';"
}

log "watching $MIX enrichment (target $TOTAL users)"

while true; do
  lk=$(likes_done)
  log "likes: ${lk:-0}/$TOTAL"
  if [ "${lk:-0}" -ge "$TOTAL" ]; then
    log "likes enrichment complete"
    break
  fi
  sleep 120
done

while true; do
  pl=$(playlists_done)
  log "playlists: ${pl:-0}/$TOTAL"
  if [ "${pl:-0}" -ge "$TOTAL" ]; then
    log "playlists enrichment complete"
    break
  fi
  sleep 120
done

log "running score-bots"
$PY -m personalization.main score-bots --mix "$MIX" 2>&1 | tee -a "$LOG"

log "running cluster"
$PY -m personalization.main cluster --mix "$MIX" 2>&1 | tee -a "$LOG"

log "cross-cohort overlap summary"
sqlite3 "$DB" <<'SQL' | tee -a "$LOG"
.headers on
.mode column
SELECT 'shared_sc_users' AS metric, COUNT(*) AS n FROM (
  SELECT sc_user_id FROM listeners WHERE mix_id='pwgrrb1' AND sc_user_id IS NOT NULL
  INTERSECT
  SELECT sc_user_id FROM listeners WHERE mix_id='2nvzlh2k' AND sc_user_id IS NOT NULL
);
SELECT 'murph_users_also_in_bb11' AS metric, COUNT(*) FROM (
  SELECT sc_user_id FROM listeners WHERE mix_id='pwgrrb1'
  INTERSECT SELECT sc_user_id FROM listeners WHERE mix_id='2nvzlh2k'
);
SELECT 'murph_users_also_in_bb12' AS metric, COUNT(*) FROM (
  SELECT sc_user_id FROM listeners WHERE mix_id='pwgrrb1'
  INTERSECT SELECT sc_user_id FROM listeners WHERE mix_id='1fsnxchk'
);
SELECT 'shared_liked_tracks_murph_bb11' AS metric, COUNT(*) FROM (
  SELECT track_id FROM sc_likes WHERE mix_id='pwgrrb1'
  INTERSECT SELECT track_id FROM sc_likes WHERE mix_id='2nvzlh2k'
);
SELECT 'shared_liked_tracks_murph_bb12' AS metric, COUNT(*) FROM (
  SELECT track_id FROM sc_likes WHERE mix_id='pwgrrb1'
  INTERSECT SELECT track_id FROM sc_likes WHERE mix_id='1fsnxchk'
);
SQL

log "done — review $LOG and update personalization/findings.md"
