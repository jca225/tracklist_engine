#!/usr/bin/env bash
# backup_als.sh — snapshot every Ableton .als (+ manifests) before any relocation
# or GT re-export. Operation Crush Phase-1 pre-gate (docs/operation_crush_master_plan.md §4).
#
# An .als stores RELATIVE sample paths at a fixed folder depth, not the audio.
# Moving an .als to a new depth, or moving/renaming its audio, orphans every
# reference at once. The .als files are tiny and irreplaceable (hand labeling);
# the audio is large and reproducible. So we snapshot the .als + manifests, both
# in-tree and off-tree (survives an ~/aligning wipe), and verify integrity.
#
# Usage: scripts/backup_als.sh [ALIGNING_DIR]   (default: ~/aligning)
set -euo pipefail

ALIGNING="${1:-$HOME/aligning}"
TS="$(date +%Y%m%d_%H%M%S)"
IN_TREE="$ALIGNING/_backups/als_snapshot_$TS"
OFF_TREE="$HOME/als_snapshots/als_snapshot_$TS"

[ -d "$ALIGNING" ] || { echo "no aligning dir at $ALIGNING" >&2; exit 1; }

echo "snapshotting .als + manifests from $ALIGNING ..."
mkdir -p "$IN_TREE"
# preserve structure so depth/relative-refs are documented; skip the backups tree
rsync -aR --prune-empty-dirs \
  --include='*/' --include='*.als' --include='manifest.json' --include='manifest.json.bak*' \
  --exclude='_backups/***' --exclude='*' \
  "$ALIGNING/./" "$IN_TREE/"

mkdir -p "$OFF_TREE"
rsync -a "$IN_TREE/" "$OFF_TREE/"

n_in=$(find "$IN_TREE" -name '*.als' | wc -l | tr -d ' ')
n_off=$(find "$OFF_TREE" -name '*.als' | wc -l | tr -d ' ')
[ "$n_in" = "$n_off" ] || { echo "MIRROR MISMATCH: in=$n_in off=$n_off" >&2; exit 1; }

# integrity: .als files are gzip; a zero-byte or non-gzip file means a bad copy
bad=0
while IFS= read -r f; do
  file -b "$f" | grep -q gzip || { echo "NOT gzip (suspect): $f" >&2; bad=1; }
done < <(find "$IN_TREE" -name '*.als')
[ "$bad" = 0 ] || { echo "integrity check FAILED" >&2; exit 1; }

echo "OK — $n_in .als backed up + verified"
echo "  in-tree:  $IN_TREE"
echo "  off-tree: $OFF_TREE"
