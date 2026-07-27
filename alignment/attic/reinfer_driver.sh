#!/usr/bin/env bash
# Parallel post-fix re-infer driver (2026-07-09). bash-3.2 safe (macOS /bin/bash):
# no associative arrays. Decodes each set as soon as its infer PID exits, with
# --workers 8 on the looptrace decode, then scores both.
# Args: $1 = BB12 infer PID (or "" if already done), $2 = BB11 infer PID.
set -u
cd "$(dirname "$0")/.."
PY=./venvs/audio/bin/python

decode() {  # $1 set_id   $2 infer_pid (may be empty = already done)
  sid="$1"; pid="$2"
  if [ -n "$pid" ]; then
    echo "### wait infer $sid (pid $pid) $(date)"
    while kill -0 "$pid" 2>/dev/null; do sleep 30; done
  fi
  echo "### joint_ref_decode looptrace $sid --workers 8 $(date)"
  $PY -m alignment.joint_ref_decode \
      --set-id "$sid" --decoder looptrace --workers 8 \
      --out "alignment/out/${sid}_predicted_timeline_lt_v2.json" \
      > "logs/jointdecode_${sid}.log" 2>&1
  echo "### joint_ref_decode $sid rc=$? $(date)"
}

echo "DRIVER START $(date) (BB12pid='${1:-}' BB11pid='${2:-}')"
decode 1fsnxchk "${1:-}"     # BB12 infer already done -> decode immediately
decode 2nvzlh2k "${2:-}"     # waits for BB11 infer, then decodes

echo "### scorecard $(date)"
for sid in 1fsnxchk 2nvzlh2k; do
  echo "=================== $sid ==================="
  $PY -m alignment.score_timeline_vs_gt \
      --set-id "$sid" --fibers \
      --timeline "alignment/out/${sid}_predicted_timeline_lt_v2.json" \
      2>&1 | tail -45
done
echo "DRIVER DONE $(date)"
