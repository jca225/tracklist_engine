#!/usr/bin/env bash
# Resumable likes + playlists enrichment for pwgrrb1 (survives terminal/session exit).
set -euo pipefail
cd "$(dirname "$0")/.."

MIX="pwgrrb1"
DB="data/taste/taste_warehouse.db"
PY="venvs/audio/bin/python"
LOG="logs/murph_taste_enrich.log"
PIDFILE="logs/murph_taste_enrich.pid"
export TASTE_SC_RPM="${TASTE_SC_RPM:-30}"
BATCH="${BATCH:-10}"

mkdir -p logs

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

likes_complete() {
  local out
  out=$($PY -m personalization.main enrich --mix "$MIX" --batch "$BATCH" 2>&1) || {
    log "likes tick FAILED: $out"
    return 1
  }
  echo "$out" >>"$LOG"
  echo "$out" | tail -1
  echo "$out" | grep -q "enrich complete"
}

playlists_tick() {
  local out
  out=$($PY -m personalization.main enrich-playlists --mix "$MIX" --batch "$BATCH" 2>&1) || {
    log "playlists tick FAILED: $out"
    return 1
  }
  echo "$out" >>"$LOG"
  echo "$out" | tail -1
}

likes_done_n() {
  sqlite3 "$DB" "SELECT COALESCE(json_array_length(checkpoint_json,'\$.completed_sc_user_ids'),0) FROM scrape_checkpoints WHERE mix_id='$MIX' AND phase='enrich_likes';"
}

playlists_done_n() {
  sqlite3 "$DB" "SELECT COALESCE(json_array_length(checkpoint_json,'\$.completed_sc_user_ids'),0) FROM scrape_checkpoints WHERE mix_id='$MIX' AND phase='enrich_playlists';"
}

total_listeners() {
  sqlite3 "$DB" "SELECT COUNT(*) FROM listeners WHERE mix_id='$MIX';"
}

if [[ "${1:-}" == "--detach" ]]; then
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "already running pid=$(cat "$PIDFILE") — tail -f $LOG"
    exit 0
  fi
  # -dims: prevent idle/display/disk sleep while this long batch runs (Mac).
  nohup caffeinate -dims "$0" >>"$LOG" 2>&1 &
  echo $! >"$PIDFILE"
  echo "started pid=$(cat "$PIDFILE") (caffeinate) — tail -f $LOG"
  exit 0
fi

trap 'log "driver exiting (signal or error)"; rm -f "$PIDFILE"' EXIT
echo $$ >"$PIDFILE"
TOTAL=$(total_listeners)
log "driver start pid=$$ mix=$MIX total=$TOTAL rpm=$TASTE_SC_RPM batch=$BATCH"

# Phase 1: likes
i=0
while :; do
  i=$((i + 1))
  done_n=$(likes_done_n)
  if [[ "${done_n:-0}" -ge "$TOTAL" ]]; then
    log "likes already complete ($done_n/$TOTAL)"
    break
  fi
  log "likes tick $i starting ($done_n/$TOTAL)"
  if likes_complete; then
    log "=== LIKES DONE after $i ticks ==="
    break
  fi
  if [[ $i -ge 400 ]]; then
    log "=== LIKES SAFETY STOP at $i ticks ($done_n/$TOTAL) ==="
    exit 1
  fi
done

# Phase 2: playlists
j=0
while :; do
  j=$((j + 1))
  done_n=$(playlists_done_n)
  log "playlists tick $j ($done_n/$TOTAL)"
  if [[ "${done_n:-0}" -ge "$TOTAL" ]]; then
    log "=== PLAYLISTS DONE ($done_n/$TOTAL) ==="
    break
  fi
  playlists_tick || sleep 60
  if [[ $j -ge 400 ]]; then
    log "=== PLAYLISTS SAFETY STOP at $j ticks ($done_n/$TOTAL) ==="
    exit 1
  fi
done

log "=== ENRICHMENT COMPLETE ==="
