#!/usr/bin/env bash
# Wait for the GPU to free (parallel alignment jobs to exit), then launch the
# enhanced-vocals A/B contention-free. My own driver was already killed, so any
# match below is another agent's job. Requires 2 consecutive idle checks so it
# doesn't fire in a between-stem gap.
set -u
cd "$(dirname "$0")/../.."   # repo root
PAT='alignment_prototype\.(lyrics_align|verify_vocal|infer|infer_fused)'
echo "WAIT START $(date) — watching for: $PAT"
idle=0
while true; do
  if pgrep -f "$PAT" >/dev/null 2>&1; then
    idle=0
  else
    idle=$((idle + 1))
    echo "gpu idle $idle/2 at $(date)"
    [ "$idle" -ge 2 ] && break
  fi
  sleep 60
done
echo "GPU FREE $(date) — launching enhanced A/B"
exec bash alignment/run_enhance_ab.sh
