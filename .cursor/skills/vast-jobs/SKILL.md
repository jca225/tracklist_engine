---
name: vast-jobs
description: >-
  Run GPU jobs on Vast.ai for tracklist_engine. Use when renting a Vast box,
  SSH to vast/vast-synth/vast-taste, bootstrapping CUDA, rsyncing to /workspace,
  launching tmux jobs, pulling results, or destroying instances. Read this BEFORE
  improvising — follow existing orchestrator scripts.
---

# Vast.ai jobs (tracklist_engine)

## First reads (mandatory)

1. **`docs/vast_coordination.md`** — registry, labels, collision rules, curl API
2. **On the instance:** `/etc/vast-agents-guide.md` (or `./AGENTS.md`)
3. **The orchestrator script** for your job — do not rewrite from scratch:
   - `scripts/vast_taste_embed.sh` — tail MERT embeds (`taste-embed`)
   - `scripts/vast_info_dynamics.sh` — RoFormer + MERT sets (`info-dynamics`)
   - `scripts/vast_synthetic_pretrain.sh` — synthetic MERT pretrain (`synth-pretrain`)
   - `scripts/vast_bootstrap.sh` — full bootstrap (Demucs/Essentia/pi-storage)

Copy the pattern from a working script. The repo already solved CUDA, SSH, tmux, and pull.

## Why agents fail here (avoid these)

| Mistake | Fix |
|---------|-----|
| Raw-rent `vastai/pytorch:@vastai-automatic-tag` via API | Use **PyTorch template** `template_hash_id: 4e17788f74f075dd9aab7d0d4427968f` + `runtype: ssh_direct` |
| CUDA check before torch install | Base template has **no torch** — `pip install torch --index-url https://download.pytorch.org/whl/cu128` first, then verify CUDA |
| Full repo rsync to 32 GB overlay | **Minimal sync only** — never `data/db` (~11 GB). See orchestrator excludes |
| SSH without Vast key/host setup | `IdentityFile ~/.ssh/id_ed25519` + `UserKnownHostsFile ~/.ssh/known_hosts.vast` |
| `vastai` pip CLI | Broken on Mac Python 3.14 — use **curl API** with `~/.config/vastai/vast_api_key` |
| Killing long rsync mid-flight | Rsync ~11 GB takes 15–30 min — run in **tmux/nohup**, verify with `find … -name mix.flac \| wc -l` |
| Destroying another agent's box | List instances first; only destroy **your labeled** box |

## SSH config template

```
Host vast-synth
    HostName ssh<N>.vast.ai
    Port <port>
    User root
    IdentityFile ~/.ssh/id_ed25519
    UserKnownHostsFile ~/.ssh/known_hosts.vast
    StrictHostKeyChecking accept-new
```

Get host/port from API:
```bash
KEY=$(cat ~/.config/vastai/vast_api_key)
curl -s https://console.vast.ai/api/v0/instances/ -H "Authorization: Bearer $KEY" \
  | python3 -c "import json,sys; [print(i['id'],i.get('label'),i.get('ssh_host'),i.get('ssh_port')) for i in json.load(sys.stdin).get('instances',[])]"
```

## Preflight checklist

Before launching any job:

```bash
# 1. Instance is yours and labeled
KEY=$(cat ~/.config/vastai/vast_api_key)
curl -s https://console.vast.ai/api/v0/instances/ -H "Authorization: Bearer $KEY" | python3 -m json.tool

# 2. SSH works
ssh -o IdentityFile=~/.ssh/id_ed25519 -o UserKnownHostsFile=~/.ssh/known_hosts.vast vast-synth 'echo ok'

# 3. Disk headroom (32 GB overlay — need corpus + torch + repo)
ssh vast-synth 'df -h /workspace'

# 4. CUDA after bootstrap
ssh vast-synth '/venv/main/bin/python -c "import torch; print(torch.cuda.get_device_name(0))"'
```

Free space if needed: remove `_mac_scratch`, partial `data/db`, never delete another agent's work.

## Launch pattern

**Always use the orchestrator script**, not ad-hoc ssh:

```bash
cd ~/Desktop/tracklist_engine
scripts/vast_synthetic_pretrain.sh --ssh-alias vast-synth   # example
```

Monitor:
```bash
ssh vast-synth 'find /workspace/tracklist_engine/data/synthetic_mixes -name mix.flac | wc -l'  # corpus progress
ssh vast-synth 'tmux ls'
ssh vast-synth 'tail -f /workspace/tracklist_engine/synth_pretrain.log'
```

Success criteria for synth-pretrain:
- 100 mixes with `mix.flac` on remote
- tmux session `synth-pretrain` running
- `synth_pretrain.log` exists and shows epoch/ablation output

## Pull + destroy

```bash
scripts/vast_synthetic_pretrain.sh --pull-only --ssh-alias vast-synth
curl -s -X DELETE "https://console.vast.ai/api/v0/instances/<ID>/" -H "Authorization: Bearer $KEY"
```

Stopped ≠ destroyed. Idle boxes at `gpu_util=0` still bill.

## Mac vs Vast split

See `docs/vast_coordination.md` GPU policy table. Multi-track MERT / transformer pretrain → **Vast only**. Mac OK for smoke tests (`--limit ≤5`).
