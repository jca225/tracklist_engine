#!/usr/bin/env bash
# Mac → Vast orchestrator for one analysis session.
#
# Workflow:
#   1. You rent a PRO 4000 (or similar) on cloud.vast.ai
#   2. You update ~/.ssh/config 'Host vast' with the new HostName/Port
#   3. You run this script. It:
#        a. Verifies SSH to vast
#        b. Bootstraps Vast (clone repo, install deps, download essentia models)
#        c. On first run: generates an SSH key on Vast and pauses for you to
#           authorize it on pi-storage (one-time setup; pubkey persists)
#        d. Mounts pi-storage's /mnt/storage on Vast at /mnt/pi-storage via sshfs
#        e. Starts vast_worker --loop in a tmux session on Vast
#        f. Tails the log so you can watch progress (Ctrl-C to detach without
#           killing the worker — tmux keeps it alive)
#   4. When the queue drains (or you hit --max-tracks), tear down via the
#      Vast.ai dashboard.
#
# Examples:
#   scripts/vast_run.sh --bb-only           # analyze just BB10-15 (~700 tracks)
#   scripts/vast_run.sh --max-tracks 100    # smoke run
#   scripts/vast_run.sh                     # drain everything pi-storage has
#
# Env overrides:
#   VAST_SSH_ALIAS=vast            # ~/.ssh/config Host alias
#   PI_SSH_ALIAS=pi-storage        # what Vast SSHes to (must resolve from Vast,
#                                    so use the Tailscale magicDNS hostname)
#   PI_USER=johncabrahams          # user account on pi-storage
set -euo pipefail

VAST_SSH_ALIAS="${VAST_SSH_ALIAS:-vast}"
PI_SSH_HOST="${PI_SSH_HOST:-pi-storage.tail116c2d.ts.net}"
PI_USER="${PI_USER:-johncabrahams}"
TMUX_SESSION="${TMUX_SESSION:-analyze}"

WORKER_ARGS=()
PASSTHROUGH=()
while [ $# -gt 0 ]; do
    case "$1" in
        --bb-only|--max-tracks|--set-ids|--device|--log-level|--separator)
            PASSTHROUGH+=("$1")
            if [ "$1" != "--bb-only" ]; then
                shift
                PASSTHROUGH+=("$1")
            fi
            ;;
        --help|-h)
            sed -n '2,32p' "$0"
            exit 0
            ;;
        *)
            echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

say() { printf "==> %s\n" "$*"; }

# ---------- 1. SSH reachability ----------
say "checking SSH to ${VAST_SSH_ALIAS}"
ssh -o ConnectTimeout=10 -o BatchMode=yes "${VAST_SSH_ALIAS}" 'echo ok' >/dev/null \
    || { echo "ssh ${VAST_SSH_ALIAS} failed — update ~/.ssh/config with the rented box's HostName/Port" >&2; exit 1; }

# ---------- 2. Bootstrap Vast (idempotent) ----------
say "bootstrapping vast (clone + pip + tailscale + essentia models)"
# Use SKIP_CLONE=0 so the bootstrap pulls from public origin
ssh "${VAST_SSH_ALIAS}" 'bash <(curl -fsSL https://raw.githubusercontent.com/jca225/tracklist_engine/main/scripts/vast_bootstrap.sh)' \
    | tail -20

# ---------- 3. SSH key (one-time pi-storage trust) ----------
say "ensuring vast has an SSH key"
VAST_PUBKEY=$(ssh "${VAST_SSH_ALIAS}" 'test -f ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 >/dev/null; cat ~/.ssh/id_ed25519.pub')

# Check whether pi-storage already trusts Vast's key
if ! ssh "${VAST_SSH_ALIAS}" "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 ${PI_USER}@${PI_SSH_HOST} echo ok" >/dev/null 2>&1; then
    echo
    echo "Vast can't yet SSH to pi-storage. One-time setup needed."
    echo "Copy this pubkey:"
    echo
    echo "    ${VAST_PUBKEY}"
    echo
    echo "Then on your Mac (where you have ssh access to pi-storage), run:"
    echo "    ssh ${PI_SSH_HOST} \"echo '${VAST_PUBKEY}' >> ~/.ssh/authorized_keys\""
    echo
    read -rp "Hit Enter once you've done that (or Ctrl-C to abort)... "
fi

# Verify it actually works now
say "verifying vast → pi-storage SSH"
ssh "${VAST_SSH_ALIAS}" "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10 ${PI_USER}@${PI_SSH_HOST} hostname" \
    || { echo "vast → pi-storage SSH still failing — check pi-storage authorized_keys" >&2; exit 1; }

# ---------- 4. SSHFS mount of /mnt/storage ----------
say "mounting pi-storage's /mnt/storage at /mnt/pi-storage on vast"
ssh "${VAST_SSH_ALIAS}" "
    apt-get install -y -qq sshfs >/dev/null 2>&1 || true
    mkdir -p /mnt/pi-storage
    if mountpoint -q /mnt/pi-storage; then
        echo '  already mounted'
    else
        sshfs ${PI_USER}@${PI_SSH_HOST}:/mnt/storage /mnt/pi-storage \
            -o StrictHostKeyChecking=no \
            -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 \
            -o allow_other
        echo '  mounted'
    fi
    ls /mnt/pi-storage | head -3
"

# ---------- 5. Start the worker in tmux ----------
WORKER_ARGS=(
    --loop
    --db /mnt/pi-storage/data/db/music_database.db
    --audio-root /mnt/pi-storage
    --stems-dir /mnt/pi-storage/stems
)
WORKER_ARGS+=("${PASSTHROUGH[@]}")

say "starting vast_worker in tmux session '${TMUX_SESSION}'"
say "  args: ${WORKER_ARGS[*]}"

ssh "${VAST_SSH_ALIAS}" "
    tmux kill-session -t ${TMUX_SESSION} 2>/dev/null || true
    tmux new -d -s ${TMUX_SESSION} \"PYTHONPATH=/workspace/tracklist_engine /venv/main/bin/python -m analysis.vast_worker ${WORKER_ARGS[*]} 2>&1 | tee /workspace/vast_worker.log\"
    sleep 3
    tmux ls
"

# ---------- 6. Tail the log ----------
say "tailing /workspace/vast_worker.log (Ctrl-C to detach; worker keeps running)"
echo
ssh "${VAST_SSH_ALIAS}" 'tail -f /workspace/vast_worker.log'
