#!/usr/bin/env bash
# reconcile_pass1_manual.sh — human-reviewed dup-cluster + safe coexist cleanup
# on pi-storage BEFORE `reconcile_orphans.py --apply`.
#
# Resolves the 24 REVIEW rows from the first dry-run (see conversation 2026-05-30).
# Does NOT touch the DB. After this + `git pull` (≥3a39d98), re-dry-run then --apply.
#
# Usage (on pi-storage, from repo root):
#   bash scripts/reconcile_pass1_manual.sh              # print actions only
#   bash scripts/reconcile_pass1_manual.sh --execute      # actually delete files
#
# Another agent may be editing the repo concurrently — this script is intentionally
# standalone (no Python imports beyond bash).

set -euo pipefail

AUDIO_ROOT="${TRACKLIST_AUDIO_ROOT:-/mnt/storage}"
OBJECTS="${AUDIO_ROOT}/objects"
EXECUTE=0
if [[ "${1:-}" == "--execute" ]]; then
  EXECUTE=1
fi

rmf() {
  # rmf <path> ...
  for p in "$@"; do
    if [[ ! -e "$p" ]]; then
      echo "  skip (missing): $p"
      continue
    fi
    if [[ "$EXECUTE" -eq 1 ]]; then
      rm -rf -- "$p"
      echo "  deleted: $p"
    else
      echo "  would delete: $p"
    fi
  done
}

echo "=== reconcile pass 1 (execute=$EXECUTE) objects=$OBJECTS ==="

# ── Cluster 1: 3Qffe2hLRvo (650s wrong hit — not either Flash remix) ──
echo "--- cluster 1: Green Velvet Flash (3Qffe2hLRvo) ---"
rmf \
  "${OBJECTS}/11ud9chf/11ud9chf__youtube_music__3Qffe2hLRvo.m4a" \
  "${OBJECTS}/11ud9chf/11ud9chf__youtube_music__3Qffe2hLRvo.webm" \
  "${OBJECTS}/4dtxu75/4dtxu75__youtube_music__3Qffe2hLRvo.m4a"
# KEEP: 11ud9chf__youtube__uhw1tY0c99k.m4a (registered Eats Everything remix)

# ── Cluster 2: Jb6gcoR266U (429s ATW full — home is gnsjmf only) ──
echo "--- cluster 2: Daft Punk Around The World (Jb6gcoR266U) ---"
rmf \
  "${OBJECTS}/9hp84x/9hp84x__youtube_music__Jb6gcoR266U.m4a" \
  "${OBJECTS}/9hp84x/9hp84x__youtube_music__Jb6gcoR266U.webm" \
  "${OBJECTS}/19bg4m9p/19bg4m9p__youtube_music__Jb6gcoR266U.m4a" \
  "${OBJECTS}/19bg4m9p/19bg4m9p__youtube_music__Jb6gcoR266U.webm" \
  "${OBJECTS}/gnsjmf/gnsjmf__youtube_music__Jb6gcoR266U.webm"
# KEEP: gnsjmf__youtube_music__Jb6gcoR266U.m4a → REGISTER on pass-2 --apply
# KEEP: 9hp84x__youtube__L2yarMfIwwY.m4a (107s acappella ref)

# ── Cluster 3: naNj8bXnySE (4500s — playlist, not a track) ──
echo "--- cluster 3: GTA Heavy Thunder / KURA Thunder (naNj8bXnySE) ---"
rmf \
  "${OBJECTS}/1fw35fxp/1fw35fxp__youtube_music__naNj8bXnySE.m4a" \
  "${OBJECTS}/1fw35fxp/1fw35fxp__youtube_music__naNj8bXnySE.webm" \
  "${OBJECTS}/1mlz2hg5/1mlz2hg5__youtube_music__naNj8bXnySE.m4a" \
  "${OBJECTS}/1mlz2hg5/1mlz2hg5__youtube_music__naNj8bXnySE.webm"

# ── Cluster 4: AT0e3LGteoA (633s Lane 8 length — wrong folder) ──
echo "--- cluster 4: Strobe remixes (AT0e3LGteoA) ---"
rmf \
  "${OBJECTS}/1uz8820p/1uz8820p__youtube_music__AT0e3LGteoA.m4a" \
  "${OBJECTS}/1uz8820p/1uz8820p__youtube_music__AT0e3LGteoA.webm" \
  "${OBJECTS}/h57964g5/h57964g5__youtube_music__AT0e3LGteoA.m4a" \
  "${OBJECTS}/h57964g5/h57964g5__youtube_music__AT0e3LGteoA.webm"
# KEEP: h57964g5__youtube__uJV80_9yNno.m4a (285s Layton Giordani ref)

# ── Cluster 5: aBq83ksKmUo (4514s year-mix scale) ──
echo "--- cluster 5: Boundaries / ASOT year mix (aBq83ksKmUo) ---"
rmf \
  "${OBJECTS}/1wws7mtf/1wws7mtf__youtube_music__aBq83ksKmUo.m4a" \
  "${OBJECTS}/1wws7mtf/1wws7mtf__youtube_music__aBq83ksKmUo.webm" \
  "${OBJECTS}/hm0pvnp/hm0pvnp__youtube_music__aBq83ksKmUo.m4a" \
  "${OBJECTS}/hm0pvnp/hm0pvnp__youtube_music__aBq83ksKmUo.webm"

# ── Safe coexist dupes (orphan ≈ ref duration, bare-name duplicates) ──
echo "--- safe coexist: delete orphan duplicate only ---"
rmf \
  "${OBJECTS}/1s79kvxp/Vice, Jon Bellion - Obsession (feat. Jon Bellion).m4a" \
  "${OBJECTS}/2dn2373p/Chumbawamba - Tubthumping.m4a" \
  "${OBJECTS}/2lbnf3jx/Bryce Vine - Drew Barrymore.m4a" \
  "${OBJECTS}/2ms4pz25/JoJo - Too Little Too Late.m4a" \
  "${OBJECTS}/57cy895/Tim Berg, Avicii - Bromance - Avicii's Radio Edit.m4a" \
  "${OBJECTS}/lxmls9f/Fitz and The Tantrums - HandClap.m4a" \
  "${OBJECTS}/9q1p5f/9q1p5f__youtube_music__m-Y6ZGz5PzA.m4a" \
  "${OBJECTS}/x87nspf/x87nspf__youtube_music__kRy7IOrs-2Q.m4a"

echo ""
if [[ "$EXECUTE" -eq 0 ]]; then
  echo "Dry-run complete. Re-run with:  bash scripts/reconcile_pass1_manual.sh --execute"
  echo ""
  echo "Then on pi-storage:"
  echo "  cd ~/tracklist_engine && git pull   # need ≥3a39d98 (insert_audio_or_reap)"
  echo "  venvs/audio/bin/python -c \"from core import db; assert hasattr(db,'insert_audio_or_reap')\""
  echo "  venvs/audio/bin/python scripts/reconcile_orphans.py --review-tsv /tmp/reconcile_pass2.tsv"
  echo "  venvs/audio/bin/python scripts/reconcile_orphans.py --apply --review-tsv /tmp/reconcile_pass2.tsv"
  echo "  # promotions later:  ... --apply --apply-promotions"
else
  echo "Pass-1 deletes done. Re-dry-run reconcile_orphans before --apply."
fi
