#!/usr/bin/env bash
# Mac → Vast orchestrator for info-dynamics sets (RoFormer stems + MERT on CUDA).
#
# Follows the same proven pattern as scripts/vast_taste_embed.sh and
# scripts/vast_run.sh — do NOT rent via raw API; use the UI + this script.
#
# Prereqs (one-time per box):
#   1. Rent on cloud.vast.ai — **PyTorch template**, RTX 4090 (or PRO 4000).
#      Label the instance `info-dynamics`.
#   2. Wait for dashboard status "running" / SSH port live.
#   3. Update ~/.ssh/config:
#        Host vast
#            HostName ssh<N>.vast.ai
#            Port <port>
#            User root
#   4. First box only: bootstrap runs `tailscale up` on Vast if pi-storage
#      mount fails — click the auth URL once (see vast_bootstrap.sh).
#
# GPU policy: RoFormer + MERT run on Vast CUDA only. Beat grid runs on Vast
# CPU (beat_this) reading mix files via sshfs — not on Mac MPS.
#
# Usage:
#   scripts/vast_info_dynamics.sh --set-audio-id 324 --set-id 1n81jy3k
#   scripts/vast_info_dynamics.sh --bb9-pending          # all mix-ready BB gaps
#   scripts/vast_info_dynamics.sh --pull-only --set-id 1n81jy3k
#
# Env:
#   VAST_SSH_ALIAS=vast
#   PI_SSH_HOST=pi-storage.tail116c2d.ts.net
#   PI_USER=johncabrahams
set -euo pipefail

VAST_SSH_ALIAS="${VAST_SSH_ALIAS:-vast}"
PI_SSH_HOST="${PI_SSH_HOST:-pi-storage.tail116c2d.ts.net}"
PI_USER="${PI_USER:-johncabrahams}"
TMUX_SESSION="${TMUX_SESSION:-info-dynamics}"
REPO_LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
REPO_REMOTE="/workspace/tracklist_engine"
PY="/venv/main/bin/python"
PI_MOUNT="/mnt/pi-storage"
CANONICAL_DB="${PI_MOUNT}/data/db/music_database.db"
SET_AUDIO_ID=""
SET_ID=""
BB9_PENDING=0
PULL_ONLY=0
SKIP_BEATS=0

while [ $# -gt 0 ]; do
    case "$1" in
        --set-audio-id) SET_AUDIO_ID="$2"; shift 2 ;;
        --set-id) SET_ID="$2"; shift 2 ;;
        --bb9-pending) BB9_PENDING=1; shift ;;
        --pull-only) PULL_ONLY=1; shift ;;
        --skip-beats) SKIP_BEATS=1; shift ;;
        --ssh-alias) VAST_SSH_ALIAS="$2"; shift 2 ;;
        -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

say() { printf "==> %s\n" "$*"; }

_pull_artifacts() {
    local sid="$1"
    say "pulling MERT npz + measure times for ${sid} from ${VAST_SSH_ALIAS}"
    mkdir -p "${REPO_LOCAL}/data/analysis"
    rsync -az "${VAST_SSH_ALIAS}:${REPO_REMOTE}/data/analysis/${sid}_"*.npz \
        "${REPO_LOCAL}/data/analysis/" 2>/dev/null || true
    rsync -az "${VAST_SSH_ALIAS}:${REPO_REMOTE}/data/analysis/${sid}_measure_times.json" \
        "${REPO_LOCAL}/data/analysis/" 2>/dev/null || true
    "${REPO_LOCAL}/venvs/audio/bin/python" "${REPO_LOCAL}/scripts/cache_tracklist_boundaries.py" \
        --set-ids "${sid}" 2>/dev/null || true
    if [ -f "${REPO_LOCAL}/data/analysis/${sid}_mix_mert.npz" ]; then
        "${REPO_LOCAL}/venvs/audio/bin/python" -m eda.alignment.info_dynamics.run_set --set-id "${sid}"
    else
        echo "missing ${sid}_mix_mert.npz after pull" >&2
        return 1
    fi
}

_ensure_pi_trust() {
    local pubkey
    pubkey=$(ssh "${VAST_SSH_ALIAS}" \
        'test -f ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 -q; cat ~/.ssh/id_ed25519.pub')
    ssh pi-storage \
        "grep -F '${pubkey}' ~/.ssh/authorized_keys >/dev/null 2>&1 || echo '${pubkey}' >> ~/.ssh/authorized_keys"
}

_mount_pi_storage() {
    ssh "${VAST_SSH_ALIAS}" "bash -s" <<MOUNT
set -euo pipefail
apt-get install -y -qq sshfs sqlite3 >/dev/null 2>&1 || true
mkdir -p ${PI_MOUNT}
if mountpoint -q ${PI_MOUNT}; then
    echo "  pi-storage already mounted at ${PI_MOUNT}"
else
    sshfs ${PI_USER}@${PI_SSH_HOST}:/mnt/storage ${PI_MOUNT} \\
        -o StrictHostKeyChecking=no \\
        -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 \\
        -o allow_other
    echo "  mounted ${PI_MOUNT}"
fi
test -f ${CANONICAL_DB}
ls ${PI_MOUNT}/sets | head -2
MOUNT
}

_run_one_set() {
    local audio_id="$1"
    local sid="$2"
    say "queueing ${sid} (set_audio_id=${audio_id}) on vast"
    ssh "${VAST_SSH_ALIAS}" "bash -s" <<REMOTE
set -euo pipefail
export PYTHONPATH=${REPO_REMOTE}
cd ${REPO_REMOTE}
DB=${CANONICAL_DB}
AID=${audio_id}
SID=${sid}
SKIP_BEATS=${SKIP_BEATS}

mix=\$(sqlite3 -noheader "\$DB" "SELECT path FROM set_audio WHERE set_audio_id=\$AID")
if [ -z "\$mix" ]; then echo "no set_audio \$AID"; exit 1; fi
# paths in DB are /mnt/storage/... — remap to sshfs mount
mix="\${mix//\\/mnt\\/storage/${PI_MOUNT}}"

has_analysis=\$(sqlite3 "\$DB" "SELECT COUNT(*) FROM set_analysis WHERE set_audio_id=\$AID")
if [ "\$SKIP_BEATS" = "0" ] && [ "\$has_analysis" = "0" ]; then
    echo "== beats (CPU) on \$mix"
    ${PY} scripts/pi_analyze_set_beats.py --set-audio-id "\$AID" --mix-path "\$mix" --db "\$DB"
fi

stem_dir="${PI_MOUNT}/stems/set/\$AID"
if [ ! -f "\$stem_dir/vocals.flac" ] || [ ! -f "\$stem_dir/instrumental.flac" ]; then
    echo "== roformer stems (CUDA)"
    ${PY} scripts/render_set_stems.py --set-audio-id "\$AID" --separator roformer --device cuda \\
        --mix "\$mix" --no-push
    mkdir -p "\$stem_dir"
    cp _mac_scratch/set_stems/set/\$AID/vocals.flac "\$stem_dir/"
    cp _mac_scratch/set_stems/set/\$AID/instrumental.flac "\$stem_dir/"
    sqlite3 "\$DB" "BEGIN; DELETE FROM set_stems WHERE set_audio_id=\$AID;
      INSERT INTO set_stems VALUES (\$AID,'vocals','\$stem_dir/vocals.flac','flac');
      INSERT INTO set_stems VALUES (\$AID,'instrumental','\$stem_dir/instrumental.flac','flac');
      COMMIT;"
fi

measures=\$(sqlite3 -noheader "\$DB" "SELECT measure_times_json FROM set_analysis WHERE set_audio_id=\$AID")
mkdir -p data/analysis
echo "\$measures" > "data/analysis/\${SID}_measure_times.json"

voc="\$stem_dir/vocals.flac"
inst="\$stem_dir/instrumental.flac"
for spec in "mix:\$mix:data/analysis/\${SID}_mix_mert.npz" \\
              "vocals:\$voc:data/analysis/\${SID}_mix_vocals_mert.npz" \\
              "instrumental:\$inst:data/analysis/\${SID}_mix_instrumental_mert.npz"; do
    label="\${spec%%:*}"; rest="\${spec#*:}"; audio="\${rest%%:*}"; out="\${rest##*:}"
    echo "== MERT (\$label) -> \$out"
    ${PY} -m eda.alignment.prepare_mix_artifact --set-id "\$SID" --audio "\$audio" \\
        --measure-times-json "data/analysis/\${SID}_measure_times.json" --out "\$out"
done
echo "done ${sid} on vast"
REMOTE
    _pull_artifacts "${sid}"
}

if [ "$PULL_ONLY" = "1" ]; then
    [ -n "$SET_ID" ] || { echo "need --set-id for --pull-only" >&2; exit 2; }
    _pull_artifacts "$SET_ID"
    exit 0
fi

say "coordination: box must be labeled info-dynamics and YOURS (see docs/vast_coordination.md)"
say "checking SSH + CUDA on ${VAST_SSH_ALIAS}"
ssh -o ConnectTimeout=15 -o BatchMode=yes "${VAST_SSH_ALIAS}" 'echo ok' >/dev/null \
    || { echo "ssh failed — rent 4090 PyTorch template in UI, update ~/.ssh/config" >&2; exit 1; }
ssh "${VAST_SSH_ALIAS}" "${PY} -c \"import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))\"" \
    || { echo "CUDA unavailable on this box — destroy it and rent a 4090 PyTorch template in the UI" >&2; exit 1; }

say "bootstrapping (clone + deps + roformer — taste-embed pattern)"
ssh "${VAST_SSH_ALIAS}" "bash -s" <<'BOOT'
set -euo pipefail
REPO_DIR=/workspace/tracklist_engine
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone --depth 1 https://github.com/jca225/tracklist_engine.git "$REPO_DIR"
fi
apt-get update -qq
apt-get install -y -qq ffmpeg nodejs libsndfile1 build-essential git rsync curl ca-certificates sqlite3 >/dev/null
/venv/main/bin/pip install -q \
    "yt-dlp>=2026.5.0" "librosa>=0.11" "soundfile>=0.13" "pyloudnorm>=0.2" \
    "beat-this>=1.1" "transformers>=4.57" "timm>=1.0"
/venv/main/bin/pip install -q --no-deps "demucs>=4.0.1" || true
/venv/main/bin/pip install -q "dora-search" "julius>=0.2.3" "lameenc>=1.2" "openunmix" || true
/venv/main/bin/python -c "import torch; assert torch.cuda.is_available()"
echo bootstrap ok
BOOT

say "syncing repo snapshot (local may be ahead of origin)"
rsync -az --exclude '.git' --exclude 'venvs' --exclude 'data' --exclude '_mac_scratch' \
    --exclude 'workspaces/msst_webui/pretrain' --exclude 'workspaces/msst_webui/logs' \
    "${REPO_LOCAL}/" "${VAST_SSH_ALIAS}:${REPO_REMOTE}/"

say "roformer backend"
ssh "${VAST_SSH_ALIAS}" "cd ${REPO_REMOTE} && bash scripts/setup_roformer_separation.sh" 2>&1 | tail -8

say "pi-storage trust + sshfs"
_ensure_pi_trust
if ! _mount_pi_storage; then
    echo
    echo "sshfs mount failed. On the vast box run once:"
    echo "  ssh ${VAST_SSH_ALIAS} 'tailscale up --hostname=vast-info-dynamics'"
    echo "Then re-run this script."
    exit 1
fi

if [ "$BB9_PENDING" = "1" ]; then
    PAIRS=(
        "324:1n81jy3k" "7:w1mgcjt" "4:qj4v0wt" "3:1yl70ql1" "2:237tdqmk"
        "93:zwf3n2t" "551:9l2wdv1" "552:z0mhsf1" "553:x5yyn4k" "554:21khc009"
        "555:2svckg31" "556:1mpqt5wk" "557:2cxndfmk" "36:2vpur281"
    )
    for pair in "${PAIRS[@]}"; do
        aid="${pair%%:*}"; sid="${pair##*:}"
        if [ -f "${REPO_LOCAL}/data/analysis/info_dynamics_grid/cross_set_summary.tsv" ] \
           && grep -q "^${sid}" "${REPO_LOCAL}/data/analysis/info_dynamics_grid/cross_set_summary.tsv" 2>/dev/null; then
            say "skip ${sid} (already in cross_set_summary)"
            continue
        fi
        _run_one_set "$aid" "$sid"
    done
elif [ -n "$SET_AUDIO_ID" ] && [ -n "$SET_ID" ]; then
    _run_one_set "$SET_AUDIO_ID" "$SET_ID"
else
    echo "pass --set-audio-id + --set-id, or --bb9-pending" >&2
    exit 2
fi

say "all done — destroy YOUR info-dynamics box in the Vast dashboard when finished"
