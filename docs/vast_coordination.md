# Vast.ai / cluster coordination (multi-agent)

Multiple agents run in parallel and share **one Vast.ai account** + the pi-storage
cluster. This is the registry + protocol so no two collide. **Read and update this
file before renting a Vast box or starting a long cluster job.**

## Protocol

1. **List before create.** `vastai show instances` first. **Never reuse, stop, or
   destroy a box you did not create** — another agent may be mid-job on it.
2. **One box per agent, distinctly labeled.** Tag your instance (see the registry
   below) so ownership is unambiguous. Tear down **only your labeled box**.
3. **Namespace your outputs.** Write to a path/DB no other agent writes. Declare it
   in the registry. Never write another agent's namespace.
4. **Budget/instance cap.** Agree a ceiling so two boxes don't blow spend or hit the
   account instance limit. Default: at most 1 box per agent at a time.
5. **Separate IP = separate SoundCloud/YouTube rate-limit** — a dedicated box per
   agent avoids a shared download ban.

## GPU policy (Mac vs Vast)

**Deep-learning GPU work runs on Vast only** — do not start multi-track MERT /
transformer inference on Mac MPS (slow, ties up the laptop, WiFi-sensitive).

| Runs on **Vast** (CUDA) | Runs on **Mac** (CPU / I/O) |
|---|---|
| `personalization.embed_tail` (tail MERT batch) | SoundCloud scrape, SQLite, causal pairs |
| `prior-mert` GPU batches | `taste_model_v0` (ID-CF, no GPU) |
| Corpus MERT re-embed / fine-tune | Smoke: `--limit ≤5 --allow-local-gpu` only |
| `analysis.vast_worker` / `scripts/vast_loop.py` | |

Launch taste embed: `scripts/vast_taste_embed.sh` → label box **`taste-embed`**
→ pull with `--pull-only` → **destroy only your box** when done.

Info-dynamics sets (RoFormer + MERT vs tracklist cues): rent **4090 PyTorch
template** in the UI (same as taste-embed — do not use raw API / cheap 3090s),
label **`info-dynamics`**, update `Host vast` in `~/.ssh/config`, then:

    scripts/vast_info_dynamics.sh --set-audio-id 324 --set-id 1n81jy3k
    scripts/vast_info_dynamics.sh --bb9-pending

## Ownership registry

| Agent / task | Instance label | Reads | Writes (namespace) | Touches canonical DB? | Status |
|---|---|---|---|---|---|
| **taste-embed** | `taste-embed` | SoundCloud (fresh DL) | `data/taste/tail_track_embeds.pkl` (local pickle) | **No** | **running** on Vast 4090 (`40765622`); auto-destroy on completion |
| **info-dynamics / BB9–25** (this session) | `info-dynamics` | pi-storage via sshfs | `data/analysis/*_mix_*_mert.npz`, `info_dynamics_grid/` (local Mac) | **Yes** (beats CPU + GPU stems/MERT on Vast) | **awaiting UI rent** — run `scripts/vast_info_dynamics.sh` after 4090 PyTorch box + `Host vast` in ssh config |
| analysis / track corpus | *(unknown — confirm)* | pi-storage `objects/` | `music_database.db` (`track_mert_measures`, …), pi-storage `stems/` | **Yes** | BB9–25 ingest running on pi |

## Collision surface, and why taste-embed is isolated

The taste-embed job downloads from SoundCloud and writes a **local pickle** — it does
**not** touch `music_database.db`, pi-storage `objects/`/`stems/`, or any shared job
queue. So at the data layer there is **no write contention** with the analysis agent.
The only shared resource is the **Vast account/instances** — covered by the protocol
above (separate labeled box, list-before-create, destroy-only-yours).

> If taste-embed ever persists to a canonical DB, it must use the **taste warehouse**
> (`data/taste/taste_warehouse.db`, table `sc_track_mert`) — NOT `music_database.db` —
> to stay out of the analysis agent's namespace.

## Access from Mac (API key — no dashboard needed)

Vast IS reachable from the Mac via the account API key — **every agent can list,
inspect, and destroy instances programmatically.** Do not assume you need the UI.

- **Key:** `~/.config/vastai/vast_api_key` (also `.env: VAST_API_KEY`).
- **List instances + cost/idle check:**
  ```bash
  KEY=$(cat ~/.config/vastai/vast_api_key)
  curl -s https://console.vast.ai/api/v0/instances/ -H "Authorization: Bearer $KEY" \
    | python -m json.tool
  ```
  Per instance: `id`, `label`, `gpu_util` (0/None ≈ idle), `dph_total` ($/hr),
  `start_date` (elapsed = now − start_date), `ssh_host`:`ssh_port`, `status_msg`.
- **Destroy a box (stops billing immediately):**
  ```bash
  curl -s -X DELETE https://console.vast.ai/api/v0/instances/<ID>/ -H "Authorization: Bearer $KEY"
  ```
- The `vastai` pip CLI **fails to install under this Mac's Python 3.14** (pyexpat /
  libexpat). Use the **curl API above**, not the CLI.

**Cost hygiene (EVERY agent):** when your job finishes, **DESTROY your labeled box**
(stopping ≠ destroying — stopped boxes can still bill for storage). Periodically list
instances; a box at `gpu_util=0` with a finished job is a money leak. Never destroy
another agent's *active* box — but surface idle orphans to the user immediately.

## Status 2026-06-13

- `taste-embed` (`40765622`) — auto-torn-down ✓; 732 tail embeds pulled to `data/taste/tail_track_embeds.pkl`.
- `info-dynamics` (`40770301`, `40770307`) — **destroyed 2026-06-13** (both idle ~0% GPU, 11.5h, ~$10 total — orphans not torn down by the info-dynamics agent). Billing stopped.
- **0 instances running.** If the info-dynamics agent resumes, it must re-rent and re-verify its results were pulled before these were killed.
