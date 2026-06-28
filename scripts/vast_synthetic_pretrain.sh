#!/usr/bin/env bash
# Mac → Vast orchestrator for synthetic-mix MERT pretrain + BB12 ablation.
#
# Isolated job: reads/writes data/synthetic_mixes + .cache/pretrain_synthetic_mert.pt
# only. No pi-storage / music_database.db. Label box `synth-pretrain`.
#
# Prereqs:
#   1. Rent 4090 **PyTorch template** on cloud.vast.ai (UI) OR:
#        scripts/vast_synthetic_pretrain.sh --rent
#      (uses template_hash_id — do NOT raw-rent vastai/pytorch image; see docs/vast_coordination.md)
#   2. ~/.ssh/config (match working Host vast pattern):
#        Host vast-synth
#            HostName <ssh_host>
#            Port <ssh_port>
#            User root
#            IdentityFile ~/.ssh/id_ed25519
#            UserKnownHostsFile ~/.ssh/known_hosts.vast
#            StrictHostKeyChecking accept-new
#
# Usage:
#   scripts/vast_synthetic_pretrain.sh --rent              # API rent + wait + run
#   scripts/vast_synthetic_pretrain.sh                     # run on existing box
#   scripts/vast_synthetic_pretrain.sh --generate --n 100  # generate corpus on Vast too
#   scripts/vast_synthetic_pretrain.sh --pull-only
#   scripts/vast_synthetic_pretrain.sh --destroy             # pull + remind destroy
#
# Env:
#   VAST_SSH_ALIAS=vast-synth
#   TMUX_SESSION=synth-pretrain
set -euo pipefail

VAST_SSH_ALIAS="${VAST_SSH_ALIAS:-vast-synth}"
TMUX_SESSION="${TMUX_SESSION:-synth-pretrain}"
REPO_LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
REPO_REMOTE="/workspace/tracklist_engine"
PY="/venv/main/bin/python"
# Corpus + checkpoint are overridable so v2 (BB12-realistic) reuses this script:
#   SYNTH_SUBDIR=synthetic_mixes_v2 CKPT_NAME=pretrain_synthetic_v2_mert.pt scripts/...
SYNTH_SUBDIR="${SYNTH_SUBDIR:-synthetic_mixes}"
CKPT_NAME="${CKPT_NAME:-pretrain_synthetic_mert.pt}"
# Overlay disk size (GB) for the rented box. v2 corpus stores full ref stems
# (~31GB for 100 windows), so the default 32GB template overflows — bump it.
DISK_GB="${DISK_GB:-120}"
# Official Vast PyTorch template (same as docs.vast.ai API examples — NOT raw image tag)
PYTORCH_TEMPLATE_HASH="${PYTORCH_TEMPLATE_HASH:-4e17788f74f075dd9aab7d0d4427968f}"
SSH_OPTS=(-o IdentityFile="${HOME}/.ssh/id_ed25519" -o UserKnownHostsFile="${HOME}/.ssh/known_hosts.vast" -o StrictHostKeyChecking=accept-new)
N_MIXES=100
CURRICULUM=medium
GENERATE=0
PULL_ONLY=0
DESTROY=0
RENT=0
INSTANCE_ID=""
SEED=1

while [ $# -gt 0 ]; do
    case "$1" in
        --n) N_MIXES="$2"; shift 2 ;;
        --curriculum) CURRICULUM="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --generate) GENERATE=1; shift ;;
        --rent) RENT=1; shift ;;
        --instance-id) INSTANCE_ID="$2"; shift 2 ;;
        --pull-only) PULL_ONLY=1; shift ;;
        --destroy) DESTROY=1; shift ;;
        --ssh-alias) VAST_SSH_ALIAS="$2"; shift 2 ;;
        -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

say() { printf "==> %s\n" "$*"; }

_vast_key() { cat "${HOME}/.config/vastai/vast_api_key"; }

_rent_instance() {
    local key offer_id
    key=$(_vast_key)
    say "searching RTX 4090 offers..."
    offer_id=$(curl -s -X POST "https://console.vast.ai/api/v0/bundles/" \
        -H "Authorization: Bearer ${key}" -H "Content-Type: application/json" \
        -d "{\"gpu_name\":{\"in\":[\"RTX 4090\"]},\"num_gpus\":{\"eq\":1},\"reliability\":{\"gte\":0.98},\"verified\":{\"eq\":true},\"rentable\":{\"eq\":true},\"direct_port_count\":{\"gte\":1},\"disk_space\":{\"gte\":$((DISK_GB + 20))},\"type\":\"on-demand\",\"order\":[[\"dph_total\",\"asc\"]],\"limit\":3}" \
        | python3 -c "import json,sys; o=json.load(sys.stdin).get('offers',[]); print(o[0]['id'] if o else '')")
    [ -n "$offer_id" ] || { echo "no 4090 offers" >&2; exit 1; }
    say "creating instance from offer ${offer_id} (label=synth-pretrain)..."
    INSTANCE_ID=$(curl -s -X PUT "https://console.vast.ai/api/v0/asks/${offer_id}/" \
        -H "Authorization: Bearer ${key}" -H "Content-Type: application/json" \
        -d "{\"template_hash_id\":\"${PYTORCH_TEMPLATE_HASH}\",\"disk\":${DISK_GB},\"label\":\"synth-pretrain\",\"runtype\":\"ssh_direct\"}" \
        | python3 -c "import json,sys; r=json.load(sys.stdin); print(r.get('new_contract',''))")
    [ -n "$INSTANCE_ID" ] || { echo "instance create failed" >&2; exit 1; }
    say "instance ${INSTANCE_ID} created — waiting for SSH..."
    local i host port
    for i in $(seq 1 60); do
        read -r host port <<< "$(curl -s "https://console.vast.ai/api/v0/instances/" \
            -H "Authorization: Bearer ${key}" \
            | python3 -c "
import json,sys
for inst in json.load(sys.stdin).get('instances',[]):
    if str(inst.get('id'))=='${INSTANCE_ID}':
        print(inst.get('ssh_host',''), inst.get('ssh_port',22))
        break
")"
        if [ -n "$host" ] && [ "$host" != "None" ]; then
            mkdir -p "${HOME}/.ssh"
            if ! grep -q "Host ${VAST_SSH_ALIAS}$" "${HOME}/.ssh/config" 2>/dev/null; then
                cat >> "${HOME}/.ssh/config" <<EOF

Host ${VAST_SSH_ALIAS}
    HostName ${host}
    Port ${port}
    User root
    IdentityFile ~/.ssh/id_ed25519
    UserKnownHostsFile ~/.ssh/known_hosts.vast
    StrictHostKeyChecking accept-new
EOF
                say "wrote ~/.ssh/config Host ${VAST_SSH_ALIAS}"
            else
                # update in place via temp file
                python3 - <<PY
from pathlib import Path
p = Path("${HOME}/.ssh/config")
text = p.read_text()
lines = text.splitlines()
out, in_block = [], False
for line in lines:
    if line.strip() == "Host ${VAST_SSH_ALIAS}":
        in_block = True
        out.extend(["Host ${VAST_SSH_ALIAS}", "    HostName ${host}", "    Port ${port}", "    User root", "    IdentityFile ~/.ssh/id_ed25519", "    UserKnownHostsFile ~/.ssh/known_hosts.vast", "    StrictHostKeyChecking accept-new"])
        continue
    if in_block and line.startswith("Host ") and line.strip() != "Host ${VAST_SSH_ALIAS}":
        in_block = False
    if not in_block:
        out.append(line)
p.write_text("\n".join(out) + "\n")
PY
            fi
            sleep 5
            if ssh "${SSH_OPTS[@]}" -o ConnectTimeout=15 -o BatchMode=yes "${VAST_SSH_ALIAS}" 'echo ok' >/dev/null 2>&1; then
                say "SSH ready: ${VAST_SSH_ALIAS} (${host}:${port})"
                return 0
            fi
        fi
        sleep 10
    done
    echo "timeout waiting for SSH on instance ${INSTANCE_ID}" >&2
    exit 1
}

_pull_results() {
    say "pulling checkpoint + log from ${VAST_SSH_ALIAS} (ckpt=${CKPT_NAME})"
    mkdir -p "${REPO_LOCAL}/workspaces/alignment_prototype/.cache"
    rsync -avz -e "ssh ${SSH_OPTS[*]}" --progress \
        "${VAST_SSH_ALIAS}:${REPO_REMOTE}/workspaces/alignment_prototype/.cache/${CKPT_NAME}" \
        "${REPO_LOCAL}/workspaces/alignment_prototype/.cache/" 2>/dev/null || true
    rsync -avz -e "ssh ${SSH_OPTS[*]}" \
        "${VAST_SSH_ALIAS}:${REPO_REMOTE}/synth_pretrain.log" \
        "${REPO_LOCAL}/data/${SYNTH_SUBDIR}/" 2>/dev/null || true
    if [ -f "${REPO_LOCAL}/workspaces/alignment_prototype/.cache/${CKPT_NAME}" ]; then
        say "checkpoint pulled OK"
    else
        say "warning: checkpoint not found locally"
    fi
}

if [ "$PULL_ONLY" = "1" ]; then
    _pull_results
    exit 0
fi

if [ "$RENT" = "1" ]; then
    key=$(_vast_key)
    existing=$(curl -s "https://console.vast.ai/api/v0/instances/" -H "Authorization: Bearer ${key}" \
        | python3 -c "import json,sys; print(next((str(i['id']) for i in json.load(sys.stdin).get('instances',[]) if i.get('label')=='synth-pretrain'), ''))")
    if [ -n "$existing" ]; then
        say "reusing existing synth-pretrain instance ${existing}"
        INSTANCE_ID="$existing"
    else
        _rent_instance
    fi
fi

say "coordination: box must be labeled synth-pretrain and YOURS (docs/vast_coordination.md)"
say "checking SSH to ${VAST_SSH_ALIAS}"
ssh "${SSH_OPTS[@]}" -o ConnectTimeout=20 -o BatchMode=yes "${VAST_SSH_ALIAS}" 'echo ok' >/dev/null \
    || { echo "ssh ${VAST_SSH_ALIAS} failed — run with --rent or update ~/.ssh/config" >&2; exit 1; }

# Bootstrap (no pi-storage) — must run BEFORE cuda check (base template has no torch yet)
say "bootstrapping vast (clone + deps — no pi-storage)"
ssh "${SSH_OPTS[@]}" "${VAST_SSH_ALIAS}" "bash -s" <<'BOOT'
set -euo pipefail
REPO_DIR=/workspace/tracklist_engine
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone --depth 1 https://github.com/jca225/tracklist_engine.git "$REPO_DIR"
fi
apt-get update -qq
apt-get install -y -qq ffmpeg libsndfile1 git rsync curl ca-certificates sqlite3 >/dev/null
# Base-image template: torch not preinstalled — use cu128 wheels (driver 570 / CUDA 12.8)
if ! /venv/main/bin/python -c "import torch" 2>/dev/null; then
    echo "installing torch (cu128) into /venv/main..."
    /venv/main/bin/pip install -q torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128
fi
/venv/main/bin/pip install -q \
    "librosa>=0.11" "soundfile>=0.13" "transformers>=4.57" "timm>=1.0" "pyyaml>=6"
/venv/main/bin/python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
echo bootstrap ok
BOOT

say "CUDA check after bootstrap"
ssh "${SSH_OPTS[@]}" "${VAST_SSH_ALIAS}" "${PY} -c \"import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))\""

# Sync local code (ahead of origin) + existing corpus if any
say "syncing minimal code + synthetic corpus to vast (skip data/db, analysis caches)"
ssh "${SSH_OPTS[@]}" "${VAST_SSH_ALIAS}" "mkdir -p ${REPO_REMOTE}/data/${SYNTH_SUBDIR} ${REPO_REMOTE}/workspaces/alignment_prototype/.cache"
RSYNC_EXCLUDES=(
    --exclude '.git' --exclude 'venvs' --exclude 'data/db' --exclude 'data/analysis'
    --exclude 'data/taste' --exclude 'data/mashup_compat/stems' --exclude 'data/raw'
    # NB: per-dir rsync roots make repo-relative patterns useless — these must be
    # leaf names so they match inside workspaces/alignment_prototype/.
    --exclude '.feat_cache'           # 6.6G HuBERT probe caches — pretrain never reads them
    --exclude '.cache/external_features'
    --exclude '.DS_Store' --exclude 'node_modules' --exclude '__pycache__'
)
rsync -az -e "ssh ${SSH_OPTS[*]}" "${RSYNC_EXCLUDES[@]}" \
    "${REPO_LOCAL}/workspaces/alignment_prototype/" "${VAST_SSH_ALIAS}:${REPO_REMOTE}/workspaces/alignment_prototype/"
rsync -az -e "ssh ${SSH_OPTS[*]}" "${RSYNC_EXCLUDES[@]}" \
    "${REPO_LOCAL}/core/" "${VAST_SSH_ALIAS}:${REPO_REMOTE}/core/"
rsync -az -e "ssh ${SSH_OPTS[*]}" "${RSYNC_EXCLUDES[@]}" \
    "${REPO_LOCAL}/labeling/" "${VAST_SSH_ALIAS}:${REPO_REMOTE}/labeling/"
rsync -az -e "ssh ${SSH_OPTS[*]}" "${RSYNC_EXCLUDES[@]}" \
    "${REPO_LOCAL}/analysis/" "${VAST_SSH_ALIAS}:${REPO_REMOTE}/analysis/"
if [ -d "${REPO_LOCAL}/data/${SYNTH_SUBDIR}" ]; then
    rsync -az -e "ssh ${SSH_OPTS[*]}" --delete \
        "${REPO_LOCAL}/data/${SYNTH_SUBDIR}/" "${VAST_SSH_ALIAS}:${REPO_REMOTE}/data/${SYNTH_SUBDIR}/"
fi

GEN_CMD=""
if [ "$GENERATE" = "1" ]; then
    GEN_CMD="cd ${REPO_REMOTE} && PYTHONPATH=${REPO_REMOTE} ${PY} -m workspaces.alignment_prototype.synthetic_mix.generate --n ${N_MIXES} --curriculum ${CURRICULUM} --seed ${SEED} --out data/${SYNTH_SUBDIR} &&"
fi

RUN_CMD="cd ${REPO_REMOTE} && PYTHONPATH=${REPO_REMOTE} ${GEN_CMD} \
${PY} -m workspaces.alignment_prototype.pretrain \
  --synthetic-root data/${SYNTH_SUBDIR} --features mert --epochs 40 --n-heads 3 \
  --out workspaces/alignment_prototype/.cache/${CKPT_NAME} \
  && ${PY} -m workspaces.alignment_prototype.pretrain --ablation \
  --pretrain-checkpoint workspaces/alignment_prototype/.cache/${CKPT_NAME} \
  --epochs 40 --n-heads 3 \
  2>&1 | tee ${REPO_REMOTE}/synth_pretrain.log"

say "starting synth pretrain+ablation in tmux '${TMUX_SESSION}'"
ssh "${SSH_OPTS[@]}" "${VAST_SSH_ALIAS}" "
    tmux kill-session -t ${TMUX_SESSION} 2>/dev/null || true
    tmux new -d -s ${TMUX_SESSION} bash -lc $(printf %q "${RUN_CMD}")
    sleep 3
    tmux ls
    tail -8 ${REPO_REMOTE}/synth_pretrain.log 2>/dev/null || true
"

say "running on vast — monitor:"
echo "    ssh ${VAST_SSH_ALIAS} 'tail -f ${REPO_REMOTE}/synth_pretrain.log'"
echo "    ssh ${VAST_SSH_ALIAS} 'tmux attach -t ${TMUX_SESSION}'"
echo "    scripts/vast_synthetic_pretrain.sh --pull-only"
if [ -n "$INSTANCE_ID" ]; then
    echo "    instance id: ${INSTANCE_ID}"
fi

if [ "$DESTROY" = "1" ]; then
    say "waiting for job (grep 'ablation delta' in log)..."
    ssh "${SSH_OPTS[@]}" "${VAST_SSH_ALIAS}" "bash -s" <<WAIT
set -euo pipefail
LOG=${REPO_REMOTE}/synth_pretrain.log
for i in \$(seq 1 240); do
    if grep -q 'ablation delta' "\$LOG" 2>/dev/null; then exit 0; fi
    if grep -q 'Traceback' "\$LOG" 2>/dev/null; then exit 1; fi
    sleep 30
done
echo timeout >&2; exit 1
WAIT
    _pull_results
    say "destroy instance ${INSTANCE_ID:-YOUR synth-pretrain box} in Vast dashboard when done reviewing"
fi
