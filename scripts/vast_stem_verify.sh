#!/usr/bin/env bash
# Mac → Vast orchestrator for the stem-library audio-verify pass
# (scripts/match_stem_library.py --verify) on a GPU box.
#
# Why: HuBERT-Large verify runs ~17 CPU-min/pair on pi-storage's aarch64 CPU
# (1-2 days for a ~300-acappella shortlist) vs seconds/pair on CUDA.
#
# Coordination (docs/vast_coordination.md): label = stem-verify. List before
# create; destroy ONLY the stem-verify box.
#
# Flow:
#   --rent      search cheapest 1x RTX 4090, create instance (label stem-verify),
#               write ~/.ssh/config Host vast-stem
#   --bootstrap curl the repo bootstrap (deps + repo@main + tailscale userspace),
#               apt fpcalc, then PAUSE: user must click the `tailscale up` URL
#   --mount     authorize the box's SSH key on pi-storage + sshfs /mnt/pi-storage
#   --run       dump work/recording mini-DB from pi, launch the matcher in tmux
#   --pull      copy the finished CSV back to pi staging + ~/Desktop
#   --destroy   destroy the stem-verify instance (and only it)
#
# Typical: --rent && --bootstrap  (click URL)  && --mount && --run
#          ... wait ...           --pull && --destroy
set -euo pipefail

VAST_SSH_ALIAS="${VAST_SSH_ALIAS:-vast-stem}"
LABEL="stem-verify"
TMUX_SESSION="stem-verify"
PYTORCH_TEMPLATE_HASH="${PYTORCH_TEMPLATE_HASH:-4e17788f74f075dd9aab7d0d4427968f}"
DISK_GB="${DISK_GB:-40}"
REPO_REMOTE="/workspace/tracklist_engine"
PY="/venv/main/bin/python"
STAGING="/mnt/pi-storage/staging/discord_stems"
SHORTLIST="${SHORTLIST:-verify_shortlist.txt}"
OUT_CSV="proposed_matches_verified_$(date +%Y%m%d).csv"
SSH_OPTS=(-o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new)

say() { printf "==> %s\n" "$*"; }
_vast_key() { cat "${HOME}/.config/vastai/vast_api_key"; }

list_instances() {
    curl -s "https://console.vast.ai/api/v0/instances/" \
        -H "Authorization: Bearer $(_vast_key)" \
        | python3 -c "
import json, sys
for i in json.load(sys.stdin).get('instances', []):
    print(i.get('id'), i.get('label'), i.get('gpu_name'), i.get('actual_status'))
"
}

rent() {
    say "existing instances (touch only label=${LABEL}):"
    list_instances
    local key offer_id id
    key=$(_vast_key)
    say "searching RTX 4090 offers..."
    offer_id=$(curl -s -X POST "https://console.vast.ai/api/v0/bundles/" \
        -H "Authorization: Bearer ${key}" -H "Content-Type: application/json" \
        -d "{\"gpu_name\":{\"in\":[\"RTX 4090\"]},\"num_gpus\":{\"eq\":1},\"reliability\":{\"gte\":0.98},\"verified\":{\"eq\":true},\"rentable\":{\"eq\":true},\"direct_port_count\":{\"gte\":1},\"disk_space\":{\"gte\":$((DISK_GB + 20))},\"type\":\"on-demand\",\"order\":[[\"dph_total\",\"asc\"]],\"limit\":3}" \
        | python3 -c "import json,sys; o=json.load(sys.stdin).get('offers',[]); print(o[0]['id'] if o else '')")
    [ -n "$offer_id" ] || { echo "no 4090 offers" >&2; exit 1; }
    say "creating instance from offer ${offer_id} (label=${LABEL})..."
    id=$(curl -s -X PUT "https://console.vast.ai/api/v0/asks/${offer_id}/" \
        -H "Authorization: Bearer ${key}" -H "Content-Type: application/json" \
        -d "{\"template_hash_id\":\"${PYTORCH_TEMPLATE_HASH}\",\"disk\":${DISK_GB},\"label\":\"${LABEL}\",\"runtype\":\"ssh_direct\"}" \
        | python3 -c "import json,sys; print(json.load(sys.stdin).get('new_contract',''))")
    [ -n "$id" ] || { echo "instance create failed" >&2; exit 1; }
    say "instance ${id} created — waiting for SSH..."
    local i host port
    for i in $(seq 1 60); do
        read -r host port <<< "$(curl -s "https://console.vast.ai/api/v0/instances/" \
            -H "Authorization: Bearer ${key}" | python3 -c "
import json, sys
for inst in json.load(sys.stdin).get('instances', []):
    if str(inst.get('id')) == '${id}':
        print(inst.get('ssh_host', ''), inst.get('ssh_port', 22))
        break
")"
        if [ -n "${host:-}" ] && [ "$host" != "None" ]; then
            python3 - <<PY
from pathlib import Path
p = Path.home() / ".ssh/config"
text = p.read_text() if p.exists() else ""
block = """
Host ${VAST_SSH_ALIAS}
    HostName ${host}
    Port ${port}
    User root
    IdentityFile ~/.ssh/id_ed25519
    UserKnownHostsFile ~/.ssh/known_hosts.vast
    StrictHostKeyChecking accept-new
"""
lines, out, skipping = text.splitlines(), [], False
for line in lines:
    if line.strip() == "Host ${VAST_SSH_ALIAS}":
        skipping = True
        continue
    if skipping and line.startswith("Host "):
        skipping = False
    if not skipping:
        out.append(line)
p.write_text("\n".join(out).rstrip() + "\n" + block)
print("ssh config: Host ${VAST_SSH_ALIAS} -> ${host}:${port}")
PY
            sleep 5
            if ssh "${SSH_OPTS[@]}" -o BatchMode=yes "${VAST_SSH_ALIAS}" 'echo ok' >/dev/null 2>&1; then
                say "SSH ready: ${VAST_SSH_ALIAS} (${host}:${port})  instance=${id}"
                return 0
            fi
        fi
        sleep 10
    done
    echo "instance never became SSH-reachable" >&2
    exit 1
}

bootstrap() {
    say "bootstrapping (repo@main + deps + tailscale userspace)..."
    ssh "${SSH_OPTS[@]}" "${VAST_SSH_ALIAS}" \
        'curl -fsSL https://raw.githubusercontent.com/jca225/tracklist_engine/main/scripts/vast_bootstrap.sh | bash'
    say "fpcalc (chromaprint) — not in the shared bootstrap"
    ssh "${SSH_OPTS[@]}" "${VAST_SSH_ALIAS}" \
        'apt-get install -y -qq libchromaprint-tools >/dev/null && fpcalc -version'
    say "NOW RUN ON THE BOX AND CLICK THE URL:"
    echo "    ssh ${VAST_SSH_ALIAS} 'tailscale up --hostname=vast-stem-verify'"
}

mount_pi() {
    say "authorizing the box's SSH key on pi-storage + sshfs mount"
    ssh "${SSH_OPTS[@]}" "${VAST_SSH_ALIAS}" \
        '[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 -q; cat ~/.ssh/id_ed25519.pub'
    say "append that key to pi-storage ~/.ssh/authorized_keys (done automatically):"
    ssh "${SSH_OPTS[@]}" "${VAST_SSH_ALIAS}" 'cat ~/.ssh/id_ed25519.pub' \
        | ssh pi-storage 'cat >> ~/.ssh/authorized_keys'
    ssh "${SSH_OPTS[@]}" "${VAST_SSH_ALIAS}" "
        apt-get install -y -qq sshfs netcat-openbsd >/dev/null 2>&1 || true
        mkdir -p /mnt/pi-storage
        mountpoint -q /mnt/pi-storage || sshfs pi-storage:/mnt/storage /mnt/pi-storage \
            -o StrictHostKeyChecking=no \
            -o reconnect,ServerAliveInterval=15,ServerAliveCountMax=3
        ls /mnt/pi-storage | head -3
    "
}

run() {
    say "mini identity DB (work+recording) from pi -> box"
    ssh "${SSH_OPTS[@]}" "${VAST_SSH_ALIAS}" "
        ssh pi-storage 'sqlite3 /mnt/storage/data/db/music_database.db \".dump work recording\"' \
            | sqlite3 /workspace/canonical_identity.db
        sqlite3 /workspace/canonical_identity.db 'SELECT count(*) FROM recording;'
    "
    say "launching matcher in tmux session ${TMUX_SESSION}"
    ssh "${SSH_OPTS[@]}" "${VAST_SSH_ALIAS}" "
        tmux new-session -d -s ${TMUX_SESSION} \
        'cd ${REPO_REMOTE} && ${PY} scripts/match_stem_library.py \
            --src ${STAGING} \
            --files-from ${STAGING}/${SHORTLIST} \
            --db /workspace/canonical_identity.db \
            --verify --audio-root /mnt/pi-storage \
            --out /workspace/${OUT_CSV} \
            2>&1 | tee /workspace/stem_verify.log'
        sleep 20; tail -3 /workspace/stem_verify.log
    "
}

pull() {
    say "pulling ${OUT_CSV} back to pi staging + ~/Desktop"
    ssh "${SSH_OPTS[@]}" "${VAST_SSH_ALIAS}" \
        "scp -q /workspace/${OUT_CSV} pi-storage:/mnt/storage/staging/discord_stems/"
    scp -q "${VAST_SSH_ALIAS}:/workspace/${OUT_CSV}" "${HOME}/Desktop/"
    say "pulled: ~/Desktop/${OUT_CSV} and pi staging"
}

destroy() {
    local key id
    key=$(_vast_key)
    id=$(curl -s "https://console.vast.ai/api/v0/instances/" -H "Authorization: Bearer ${key}" \
        | python3 -c "import json,sys; print(next((str(i['id']) for i in json.load(sys.stdin).get('instances',[]) if i.get('label')=='${LABEL}'), ''))")
    [ -n "$id" ] || { echo "no ${LABEL} instance found" >&2; exit 1; }
    say "destroying instance ${id} (label=${LABEL})"
    curl -s -X DELETE "https://console.vast.ai/api/v0/instances/${id}/" \
        -H "Authorization: Bearer ${key}" | head -c 200; echo
}

case "${1:-}" in
    --list) list_instances ;;
    --rent) rent ;;
    --bootstrap) bootstrap ;;
    --mount) mount_pi ;;
    --run) run ;;
    --pull) pull ;;
    --destroy) destroy ;;
    *) sed -n '2,24p' "$0" ;;
esac
