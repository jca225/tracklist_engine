#!/usr/bin/env bash
# Overnight A/B: does VoiceFixer restoration of candidate acappella stems before
# Whisper ASR improve the lyrics-alignment channel (candidate coverage +
# set_start / ref_start error)? Mix vocal stays raw both runs (isolates the
# candidate-enhancement effect). Logs to out/enhance_ab/.
set -u
cd "$(dirname "$0")/../.."   # repo root
PY=./venvs/audio/bin/python
OUT=alignment/out/enhance_ab
mkdir -p "$OUT"

run_set () {
  local name="$1" gt="$2" dir="$3"
  echo "############ $name ############"
  echo "--- [$name] BASELINE (raw candidates, full transcribe) ---"
  $PY -m alignment.lyrics_align \
      --set-dir "$dir" --gt "$gt" --transcribe --eval \
      > "$OUT/baseline_${name}.log" 2>&1
  echo "baseline rc=$? -> $OUT/baseline_${name}.log"
  echo "--- [$name] ENHANCED (VoiceFixer candidates, full transcribe) ---"
  $PY -m alignment.lyrics_align \
      --set-dir "$dir" --gt "$gt" --transcribe --enhance-vocals --eval \
      > "$OUT/enhanced_${name}.log" 2>&1
  echo "enhanced rc=$? -> $OUT/enhanced_${name}.log"
}

echo "START $(date)"
# BB11 first: thinner raw cache, acappella-weak transfer set — most to gain.
run_set BB11 labeling/fixtures/bb11_ground_truth.yaml "$HOME/aligning/2nvzlh2k__Two Friends - Big Bootie Mix Episode 11"
run_set BB12 labeling/fixtures/bb12_ground_truth.yaml "$HOME/aligning/1fsnxchk__Two Friends - Big Bootie Mix Volume 12"

echo ""
echo "================= A/B SUMMARY ================="
for name in BB11 BB12; do
  echo "----- $name -----"
  for kind in baseline enhanced; do
    f="$OUT/${kind}_${name}.log"
    echo "### $kind"
    grep -E "acappella spans with candidates|MONOTONIC \+ position|oracle ceiling|abstained" "$f" 2>/dev/null
  done
done
echo "DONE $(date)"
