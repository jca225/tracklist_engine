#!/usr/bin/env bash
# Info-dynamics batch for BB sets: beat grid (pi CPU) → roformer stems (Mac MPS)
# → MERT embed + run_set (Mac). Resumable: skips sets that already have
# set_analysis / set_stems / cross_set_summary rows.
#
#   nohup scripts/info_dynamics_bb_batch.sh >> /tmp/info_dynamics_bb_batch.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
PY=venvs/audio/bin/python
LOG=${LOG:-/tmp/info_dynamics_bb_batch.log}
exec >>"$LOG" 2>&1

echo "=== info_dynamics_bb_batch start $(date) ==="

# set_audio_id:set_id pairs — mix-ready, not yet fully tested (BB11/12 excluded)
PAIRS=(
  "324:1n81jy3k"
  "7:w1mgcjt"
  "4:qj4v0wt"
  "3:1yl70ql1"
  "2:237tdqmk"
  "93:zwf3n2t"
  "551:9l2wdv1"
  "552:z0mhsf1"
  "553:x5yyn4k"
  "554:21khc009"
  "555:2svckg31"
  "556:1mpqt5wk"
  "557:2cxndfmk"
  "36:2vpur281"
)

for pair in "${PAIRS[@]}"; do
  sid="${pair%%:*}"
  set_id="${pair##*:}"
  echo "--- [$set_id] set_audio_id=$sid $(date) ---"

  if ! grep -q "^${set_id}" data/analysis/info_dynamics_grid/cross_set_summary.tsv 2>/dev/null; then
    if ! ssh -o ConnectTimeout=15 pi-storage \
      "sqlite3 /mnt/storage/data/db/music_database.db \"SELECT 1 FROM set_analysis WHERE set_audio_id=$sid LIMIT 1\"" \
      | grep -q 1; then
      echo "  beat grid on pi…"
      ssh -o ConnectTimeout=15 pi-storage \
        "cd ~/tracklist_engine && venvs/audio/bin/python scripts/pi_analyze_set_beats.py --set-audio-id $sid"
    else
      echo "  beat grid: already present"
    fi

    if ! ssh -o ConnectTimeout=15 pi-storage \
      "test -f /mnt/storage/stems/set/$sid/vocals.flac && test -f /mnt/storage/stems/set/$sid/instrumental.flac"; then
      echo "  roformer stems on Mac…"
      caffeinate -i "$PY" scripts/render_set_stems.py \
        --set-audio-id "$sid" --separator roformer --device mps
    else
      echo "  stems: already on pi"
    fi

    echo "  MERT + run_set…"
    "$PY" scripts/cache_tracklist_boundaries.py --set-ids "$set_id"
    "$PY" scripts/info_dynamics_embed_set.py --set-id "$set_id"
  else
    echo "  skip: already in cross_set_summary.tsv"
  fi
done

echo "=== info_dynamics_bb_batch done $(date) ==="
