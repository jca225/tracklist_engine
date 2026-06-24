#!/usr/bin/env bash
# Mac → Vast orchestrator for personalization tail-track MERT embedding.
#
# Isolated from analysis/vast_worker — downloads from SoundCloud, writes
# data/taste/tail_track_embeds.pkl only (never touches music_database.db).
# See docs/vast_coordination.md — label your Vast instance `taste-embed`.
#
# Prereqs (one-time per box):
#   1. Rent a GPU box on cloud.vast.ai (PyTorch template, 4090/PRO 4000 class).
#   2. Label it `taste-embed` in the Vast dashboard.
#   3. Update ~/.ssh/config:
#        Host vast-taste
#            HostName <ip>
#            Port <port>
#            User root
#
# Usage:
#   scripts/vast_taste_embed.sh                  # full batch (min-likers 50, limit 1000)
#   scripts/vast_taste_embed.sh --limit 50     # smoke
#   scripts/vast_taste_embed.sh --pull-only    # fetch pickle from running box
#   scripts/vast_taste_embed.sh --destroy      # pull results + print destroy reminder
#
# Env:
#   VAST_SSH_ALIAS=vast-taste   default SSH Host alias
#   TMUX_SESSION=taste-embed
set -euo pipefail

VAST_SSH_ALIAS="${VAST_SSH_ALIAS:-vast-taste}"
TMUX_SESSION="${TMUX_SESSION:-taste-embed}"
REPO_LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
REPO_REMOTE="/workspace/tracklist_engine"
PY="/venv/main/bin/python"
MIN_LIKERS=50
LIMIT=1000
MAX_DURATION_S=600          # skip full DJ mixes >10 min (slow + wrong granularity)
PULL_ONLY=0
DESTROY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --min-likers) MIN_LIKERS="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --max-duration-s) MAX_DURATION_S="$2"; shift 2 ;;
        --pull-only) PULL_ONLY=1; shift ;;
        --destroy) DESTROY=1; shift ;;
        --ssh-alias) VAST_SSH_ALIAS="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,35p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

say() { printf "==> %s\n" "$*"; }

_pull_results() {
    say "pulling tail_track_embeds.pkl from ${VAST_SSH_ALIAS}"
    mkdir -p "${REPO_LOCAL}/data/taste"
    rsync -avz --progress \
        "${VAST_SSH_ALIAS}:${REPO_REMOTE}/data/taste/tail_track_embeds.pkl" \
        "${REPO_LOCAL}/data/taste/" 2>/dev/null \
        || rsync -avz "${VAST_SSH_ALIAS}:${REPO_REMOTE}/data/taste/tail_track_embeds.pkl" \
            "${REPO_LOCAL}/data/taste/"
    local n
    n=$("${REPO_LOCAL}/venvs/audio/bin/python" -c \
        "import pickle; print(len(pickle.load(open('${REPO_LOCAL}/data/taste/tail_track_embeds.pkl','rb'))))" \
        2>/dev/null || echo "?")
    say "local cache now has ${n} embedded tracks"
}

if [ "$PULL_ONLY" = "1" ]; then
    _pull_results
    exit 0
fi

# ---------- 0. list-before-create reminder ----------
say "coordination: ensure your box is labeled taste-embed and is YOURS"
say "  (never stop/destroy a box another agent is using — see docs/vast_coordination.md)"

# ---------- 1. SSH ----------
say "checking SSH to ${VAST_SSH_ALIAS}"
ssh -o ConnectTimeout=15 -o BatchMode=yes "${VAST_SSH_ALIAS}" 'echo ok' >/dev/null \
    || { echo "ssh ${VAST_SSH_ALIAS} failed — rent a box and update ~/.ssh/config" >&2; exit 1; }

# ---------- 2. Bootstrap (no pi-storage / tailscale needed) ----------
say "bootstrapping vast (clone + pip — taste-embed needs no pi-storage mount)"
ssh "${VAST_SSH_ALIAS}" "bash -s" <<'BOOT'
set -euo pipefail
REPO_DIR=/workspace/tracklist_engine
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone --depth 1 https://github.com/jca225/tracklist_engine.git "$REPO_DIR"
fi
apt-get update -qq
apt-get install -y -qq ffmpeg nodejs libsndfile1 git rsync curl ca-certificates sqlite3 >/dev/null
/venv/main/bin/pip install -q \
    "yt-dlp>=2026.5.0" "librosa>=0.11" "soundfile>=0.13" \
    "transformers>=4.57" "timm>=1.0"
/venv/main/bin/python - <<'PY'
import torch
print(f"cuda: {torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
assert torch.cuda.is_available()
PY
echo "bootstrap ok"
BOOT

# ---------- 3. Sync taste data + code snapshot ----------
say "syncing taste warehouse + resume cache to vast"
ssh "${VAST_SSH_ALIAS}" "mkdir -p ${REPO_REMOTE}/data/taste ${REPO_REMOTE}/personalization"
rsync -az "${REPO_LOCAL}/data/taste/taste_warehouse.db" \
    "${VAST_SSH_ALIAS}:${REPO_REMOTE}/data/taste/"
if [ -f "${REPO_LOCAL}/data/taste/tail_track_embeds.pkl" ]; then
    rsync -az "${REPO_LOCAL}/data/taste/tail_track_embeds.pkl" \
        "${VAST_SSH_ALIAS}:${REPO_REMOTE}/data/taste/"
fi
# Push local code (may be ahead of origin/main — don't rely on git pull alone)
rsync -az --exclude '.git' --exclude 'venvs' --exclude 'data' --exclude 'workspaces' \
    "${REPO_LOCAL}/personalization/" "${VAST_SSH_ALIAS}:${REPO_REMOTE}/personalization/"
rsync -az "${REPO_LOCAL}/analysis/adapters/" "${VAST_SSH_ALIAS}:${REPO_REMOTE}/analysis/adapters/"
rsync -az "${REPO_LOCAL}/core/" "${VAST_SSH_ALIAS}:${REPO_REMOTE}/core/"
touch "${REPO_LOCAL}/analysis/__init__.py" 2>/dev/null || true
rsync -az "${REPO_LOCAL}/analysis/__init__.py" "${VAST_SSH_ALIAS}:${REPO_REMOTE}/analysis/" 2>/dev/null || \
    ssh "${VAST_SSH_ALIAS}" "mkdir -p ${REPO_REMOTE}/analysis && touch ${REPO_REMOTE}/analysis/__init__.py"

# ---------- 4. Start embed in tmux ----------
EMBED_CMD="cd ${REPO_REMOTE} && PYTHONPATH=${REPO_REMOTE} ${PY} -m personalization.embed_tail \
    --min-likers ${MIN_LIKERS} --limit ${LIMIT} --device cuda \
    --max-duration-s ${MAX_DURATION_S} \
    2>&1 | tee ${REPO_REMOTE}/data/taste/embed_tail.log"

say "starting embed_tail in tmux session '${TMUX_SESSION}'"
ssh "${VAST_SSH_ALIAS}" "
    tmux kill-session -t ${TMUX_SESSION} 2>/dev/null || true
    tmux new -d -s ${TMUX_SESSION} bash -lc $(printf %q "${EMBED_CMD}")
    sleep 2
    tmux ls
    tail -5 ${REPO_REMOTE}/data/taste/embed_tail.log 2>/dev/null || true
"

say "running on vast — tail log:"
echo "    ssh ${VAST_SSH_ALIAS} 'tail -f ${REPO_REMOTE}/data/taste/embed_tail.log'"
echo "    ssh ${VAST_SSH_ALIAS} 'tmux attach -t ${TMUX_SESSION}'"
echo
say "when done (log shows 'done: embedded ...'):"
echo "    scripts/vast_taste_embed.sh --pull-only --ssh-alias ${VAST_SSH_ALIAS}"
if [ "$DESTROY" = "0" ]; then
    echo "    then destroy YOUR taste-embed box in the Vast dashboard (or: vastai destroy instance <id>)"
fi

if [ "$DESTROY" = "1" ]; then
    say "waiting for job to finish before pull..."
    ssh "${VAST_SSH_ALIAS}" "bash -s" <<WAIT
set -euo pipefail
LOG=${REPO_REMOTE}/data/taste/embed_tail.log
for i in \$(seq 1 720); do
    if grep -q '^done: embedded' "\$LOG" 2>/dev/null; then exit 0; fi
    sleep 30
done
echo "timeout waiting for embed — pull manually" >&2
exit 1
WAIT
    _pull_results
    say "destroy YOUR taste-embed instance in the Vast dashboard now"
fi
