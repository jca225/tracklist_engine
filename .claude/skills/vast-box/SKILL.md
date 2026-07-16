---
name: vast-box
description: Full Vast.ai GPU-box lifecycle for tracklist_engine — search offers, rent, wait for SSH (dud-host auto-re-rent), provision via GitHub clone + vast_bootstrap.sh, link to pi-storage over Tailscale, and destroy. Wraps scripts/vast_box.py. Use when the user wants to rent a Vast box, spin up a GPU worker, run MERT/RoFormer/Essentia batches, check or tear down Vast instances, or says "get me a 4090", "spin up vast", "provision the box", "destroy the vast box", "is anything running on vast".
---

# Vast.ai box lifecycle (`scripts/vast_box.py`)

Automates the rent → ssh → provision → link → destroy loop for ephemeral GPU
boxes. Stdlib-only; run from repo root with any python:

```bash
venvs/audio/bin/python scripts/vast_box.py <subcommand> ...
```

The pip `vastai` CLI does **not** install on this Mac (Py3.14 pyexpat) — this
tool and raw `curl` are the only supported API paths. API key:
`~/.config/vastai/vast_api_key`.

## Coordination rules (docs/vast_coordination.md — non-negotiable)

- **List before create.** `rent` prints existing instances first and refuses a
  duplicate label. Never reuse, stop, or destroy a box you didn't create —
  another agent may be mid-job on it.
- **One distinctly-labeled box per agent.** The label is the ownership tag.
- **Destroy only your own.** `destroy` refuses instances not in this tool's
  ledger (`~/.config/vastai/vast_box_ledger.json`). If you find an idle orphan
  (gpu_util≈0, finished job) owned by someone else, surface it to the user —
  don't kill it yourself.
- **Namespace your outputs**; declare them in the registry in
  docs/vast_coordination.md.

## The flow

```bash
# 1. Ranked offers (default filter: 1× RTX 4090, rentable, direct ports,
#    reliability>0.99, inet 500+, disk>40, cheapest first)
venvs/audio/bin/python scripts/vast_box.py search            # --gpu "RTX 5090" --max-dph 0.7

# 2. Rent (template 405071 = "PyTorch (Vast)": /venv/main + sshd baked in)
venvs/audio/bin/python scripts/vast_box.py rent <offer_id> --label <purpose>

# 3. Wait for SSH — polls, TCP-probes direct port 22, rewrites `Host vast`
#    in ~/.ssh/config on success. DUD RULE: a "running" box whose port 22
#    isn't listening after 10 min is destroyed and the next offer is
#    auto-rented (this happens in practice).
venvs/audio/bin/python scripts/vast_box.py wait-ssh <instance_id>

# 2+3 combined — RACE (preferred when duds hurt): rent N boxes at once, keep
#    the first to open port 22, destroy the losers, and quarantine dud
#    machine_ids (~/.config/vastai/vast_quarantine) so they aren't re-rented.
#    Faster + more robust than sequential wait-ssh re-rent. --n 3 is the sweet
#    spot. Emits "Winner instance id: <id>" as its last line.
venvs/audio/bin/python scripts/vast_box.py race --label <purpose> --n 3

# 4. Provision — GitHub clone (never rsync of the working tree) + bootstrap.
#    --roformer adds the 2.5 GB checkpoint fetch in the background.
venvs/audio/bin/python scripts/vast_box.py provision --id <instance_id> [--branch main] [--roformer]

# 5. Link to pi-storage (TWO human steps — see below)
venvs/audio/bin/python scripts/vast_box.py link-pi --id <instance_id>

# 6. Tear down the moment the job is done
venvs/audio/bin/python scripts/vast_box.py destroy <instance_id>
```

After `link-pi` succeeds the box can run `scripts/vast_loop.py` /
`analysis.vast_worker` against pi-storage (prefer `vast_loop.py` — rsync+ssh-SQL,
works on the common no-FUSE hosts where sshfs silently fails).

## The two unavoidable human steps (`link-pi` prints both, then waits)

Agents cannot do these; hand them to the user verbatim:

1. **pi-storage key authorization** — appending to a prod host's
   `authorized_keys` is classifier-blocked for agents. The user runs the
   printed one-liner:
   `ssh pi-storage 'echo "<pubkey> vast-<label>-<id>" >> ~/.ssh/authorized_keys'`
2. **Tailscale auth** — no stored authkey exists. `link-pi` runs
   `tailscale up --hostname=vast-<label>` on the box and prints the
   `https://login.tailscale.com/a/...` URL for the user to click. (Userspace
   tailscaled with socks5 on `localhost:1055` + the pi ProxyCommand ssh config
   are already set up by the bootstrap.)

After both, `link-pi` verifies `ssh pi-storage.tail116c2d.ts.net true` from
the box.

## Provisioning facts baked into `provision`

- Deploy = `git clone --depth 1 --branch <branch>` from GitHub + `SKIP_CLONE=1
  bash scripts/vast_bootstrap.sh`. **Never rsync the working tree** to the box —
  bulk tree uploads are blocked as exfiltration, and the box must run
  committed code anyway (push your branch first).
- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` is exported in the box's `~/.bashrc` —
  torch ≥2.12 `weights_only` default breaks beat_this/MSST checkpoint loads.
- `torch.cuda.is_available()` is asserted; a failure means wrong template/host.
- RoFormer needs `setup_roformer_separation.sh` **after** bootstrap
  (`--roformer`; ~2.5 GB, backgrounded — tail `/workspace/roformer_setup.log`).

## Cost hygiene

- **Destroy ≠ stop.** Stopped boxes still bill for storage. Always `destroy`.
- An idle box (gpu_util≈0, job finished) is a money leak — 4090s run
  ~$0.30–0.45/hr. When your job drains, destroy immediately; re-bootstrapping
  a fresh box takes ~5 min and is idempotent, so there is no reason to keep
  one warm.
- Periodically `rent`'s instance listing (or `curl .../instances/`) is the
  audit: anything running that nobody claims should be surfaced to the user.
- Teardown protocol: `destroy` prints the reminder to remove the box's
  `vast-<label>-<id>` line from pi's `authorized_keys` (user runs it) and to
  drop the node from the tailnet admin console.

## Throughput expectations (sanity checks)

- RTX 4090 full analyze (RoFormer + MERT + beats + cues): ~80–130 s/track.
- If a fresh box is far off that, suspect the host (destroy + re-rent) before
  suspecting the code.
