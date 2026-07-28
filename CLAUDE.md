# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tracklist Engine is a pipeline for analyzing recorded DJ mixes against the
tracklists scraped for them. The chain is a DAG of stages, each in its own
top-level module:

`core · scrape → ingest → analysis → labeling ⟶ (GT) ⟶ alignment`

with `eda/` a cross-cutting consumer that reads from multiple stages.

**Terminology — "labeling" vs "alignment" (do NOT conflate):**

- **labeling** = *manual* ground-truth production. A human aligns a set's
  stems against the mix in Ableton (`~/aligning/`, the `.als`), producing
  ground-truth labels. Tools: `pull_set_for_alignment.py`,
  `tag_aligning_folder.py`. Note many *existing* names use "align" for this
  (`~/aligning/`, the `set_section_alignment` table) — legacy, not renamed.
- **alignment** = *algorithmic* labeling — a model that learns to align
  automatically from the ground truth that manual labeling produces. There is
  no working aligner yet — the ML model is not built; it will incubate in
  `workspaces/`.

**BEFORE ANY ALIGNMENT WORK, read [docs/alignment_state_of_record.md](docs/alignment_state_of_record.md)** —
the single *living* record of the aligner's current best solution + settled
decisions (what to build ON, what not to re-litigate) + open fronts. A new
session reads it first; a finishing session updates it last via `/align-checkpoint`
instead of writing a new dated handoff.

**Alignment north star (target Aug 1):** the aligner consumes `{tokenized tracklist,
track audios, set audio}` → an Ableton-round-trippable structure, trained on manual
Ableton GT. Stem discovery and version/variant QA are **ingest**, *not* the aligner.
Full spec: [docs/alignment_objective.md](docs/alignment_objective.md); architecture
map + Point A→B phases: [docs/architecture_north_star.md](docs/architecture_north_star.md)
(kernel entrypoint: `make align SET=<id>`). **Current alignment status + every
headline number: [docs/alignment_status.md](docs/alignment_status.md)** (the single
source of truth — regenerated from the scorers, dated+SHA stamped). Status metrics
belong ONLY there; do not hand-type alignment numbers into other docs or memory —
cite the canonical doc. Set ids: BB11 = `2nvzlh2k`, BB12 = `1fsnxchk`. **Interpretive
frame for all alignment work:** [docs/alignment_recharacterization.md](docs/alignment_recharacterization.md)
— alignment as three near-orthogonal axes (identity / placement / structure) with
different synthetic-vs-real difficulty and generalization; read progress as three
curves, never one scalar.

Everything outside this chain is one of:
- An incubating provenance kernel: `engine/` (Rust `dj_kernel` /
  `dj_migrate` + Python sensors). Fixtures-only canary today — **not** the live
  aligner. See [engine/README.md](engine/README.md).
- A vendored dependency: `cue-detr/` (DETR-based cue-point detection model,
  consumed only by `analysis/canonical_cues.py`).
- Exploration / scratch: `eda/` notebooks.
- The algorithmic aligner lives at top-level `alignment/` (promoted from
  `workspaces/alignment_prototype`); weak-supervision fusion at `pws_aligner/`.
  Remaining experimental forks (if any) stay under `workspaces/` until promoted.
- **Deferred / attic'd (not in this checkout):** personalization, SoundCloud
  lake, and the research `lab/` north-north-star live outside the focused DAG
  tree (see `~/tracklist_attic/`). Do not re-entangle them with the aligner.

**Two-tier north star.** *Operative (now):* a SOTA alignment algorithm that
generalizes across ~40,000 DJ sets with as close to 100% accuracy as humanly
possible — the gate. *North-north (deferred):* research lab / personalization (attic). Alignment is the necessary-but-not-sufficient foundation.
Current state: [docs/alignment_state_of_record.md](docs/alignment_state_of_record.md)
(prior stock-take [docs/alignment_bearings_20260712.md](docs/alignment_bearings_20260712.md)
is a superseded 2026-07-12 snapshot).

New features land inside one of the chain modules. New top-level folders
require explicit justification.

Standing design/plan/research docs (not dated snapshots) are indexed in
[docs/design_docs_index.md](docs/design_docs_index.md); dated handoffs age out to
`docs/archive/` via `scripts/docs_gc.py` (`make docs-gc`).

## Track identity (three axes)

Recorded music in this repo is keyed on **three orthogonal axes** (plus optional
remixer name). Do not conflate them with Demucs stem names (`vocals`, `drums`, …)
or with scrape-only field names from older docs.

| Axis | DB / code values | Meaning |
|------|------------------|---------|
| **version** | `original`, `remix`, `rework`, `altversion`, `edit`, `bootleg`, `mashup` | Creative version (remix vs original) — `track_metadata.version` |

**Slot inventory (2026-06):** playable rows carry a derived `layer_role` (bed/payload/constituent/solo). Pre-pull gate: `make check-inventory SET=<set_id>`. Spec: [docs/inventory_coherence_contract.md](docs/inventory_coherence_contract.md).
| **stem** | `regular`, `acappella`, `instrumental` | Vocal/instrumental form — `track_audio.stem` (canonical default: **`regular`**, not `full` / `original`) |
| **variant** | `regular`, `extended` | Edit length — `track_audio.variant` |

**Concatenated lookup key:** `version__stem__variant` (e.g. `remix__acappella__extended`)
via `RecordingAxes.key()` in [core/identity.py](core/identity.py). Remix **artist**
lives on `recording.version_artist`, not in the key.

**Layers (do not merge):**

- **Work / recording** — `work` + `recording` tables; `recording_id` ≈ legacy
  `track_id`. Sibling recordings under one work (not stem-children).
- **Set claim** — `set_track_slots` (`claimed_version`, `claimed_stem`,
  `claimed_variant`) = what the DJ *played*; view `identity_mismatch` flags
  scrape claim vs canonical recording.
- **Download** — `track_audio` row per platform rip; `is_reference` picks the
  analysis/alignment reference.
- **Demucs** — `track_stems.stem_name` = separated components, unrelated to the
  identity `stem` axis.

**Baby rule (labeling + ingest):** one full mix file in `~/aligning/.../tracks/`;
when the tracklist says acappella/instrumental, use Demucs `stems/vocals` or
`stems/instrumental` — do not download a second full file unless it is truly a
different recording.

**Tokenizer scrape field:** `TrackRow.version_tag` remains Title Case (`Remix`,
`Rework`, …) until [tokenizer/materialize.py](tokenizer/materialize.py) writes
lowercase DB columns. Acappella/instrumental are **`claimed_stem`**, not
`version_tag` — see [tokenizer/CLAUDE.md](tokenizer/CLAUDE.md).

**Ground truth YAML:** prefer `claimed_stem:`; legacy `version_tag:` still parses.
Write-back: [labeling/commit/write_back_ground_truth.py](labeling/commit/write_back_ground_truth.py)
→ `set_ground_truth`.

**Pi-storage rollout** (code may be ahead of canonical DB — deploy + migrate together):

1. Backup `music_database.db`
2. `make deploy`
3. `scripts/migrations/migrate_identity_axes.sql` then `scripts/migrations/migrate_phase4_recording.sql`
4. `python -m tokenizer.materialize`
5. `scripts/reconcile_orphans.py` — **dry-run only** if pass-1/apply already ran

Full phase checklist: [docs/identity_and_inventory_plan.md](docs/identity_and_inventory_plan.md).
Recent reconcile state (2026-05-30): [docs/agent_handoff_reconcile_20260530.md](docs/agent_handoff_reconcile_20260530.md) — do not re-run `--apply` without a fresh dry-run.

## Per-module guides

Each module carries its own `CLAUDE.md`, loaded on demand when you touch that
subtree — keep stage-specific detail there, not here. Index:

- **[web_crawler/CLAUDE.md](web_crawler/CLAUDE.md)** — the scraper (scrape
  stage): architecture, `config.yaml`, run command, captcha. *(Pending rename to
  `scrape/`.)*
- **[tokenizer/CLAUDE.md](tokenizer/CLAUDE.md)** — scrape-row → `track_metadata`
  + `set_track_slots`. **Authoring home** for version vs stem vs variant parsing
  (`identity_axes.py`) and the "Rvmor gap" (sided rows with no `data-trackid`).
- **[ingest/CLAUDE.md](ingest/CLAUDE.md)** — audio download topology (yt-dlp main
  / spotdl retry / YT Music rescue), yt-dlp bot-detection + JS-runtime recipe.
- **[analysis/CLAUDE.md](analysis/CLAUDE.md)** — per-track/set MIR; MERT layer-6
  choice + 330M backlog; which dependency runs where (Essentia/Demucs/MERT/
  cue-detr/beat_this); `persistence.py` vs `core/db.py` boundary.
- **[labeling/CLAUDE.md](labeling/CLAUDE.md)** — manual ground-truth production:
  `pull_set_for_alignment.py` into `~/aligning/`, the consistency model
  (`--prune`), the annotator rename convention.
- **[eda/CLAUDE.md](eda/CLAUDE.md)** — exploratory analysis feeding the aligner
  (`eda/alignment/` MERT probes + `failure_analysis/` scorecard, imported by the
  aligner) + `eda/queries/`. *(Corpus-empirics / info-dynamics attic'd with `lab/`.)*

- **[core/CLAUDE.md](core/CLAUDE.md)** — shared substrate; `core/identity.py`
  (three axes); the rule that `core` imports nothing upward.


## Database

SQLite with ~25 tables split into two groups, all with cascade-deleting FKs.

**Scraper tables** (populated by `web_crawler/`):
`dj_sets` (canonical metadata), `dj_set_crawls` (HTML snapshots with ETag dedup),
`dj_set_media_links`, `dj_set_rows`, `dj_set_track_media_links`, `scrape_failures`.

**Audio-pipeline tables** (populated downstream of the scraper):
`work` / `recording` (canonical identity; `recording_id` ≈ `track_id`),
`set_audio` / `set_stems` / `set_measures` (mix-side audio + demucs stems + beat
grid), `track_audio` (`stem`, `variant`, `recording_id`) / `track_stems` /
`track_measures` (ref-track equivalents), `track_metadata` (`version`),
`set_track_slots` (per-set spine + `claimed_*` axes), `set_ground_truth` (manual GT
from labeling write-back), `track_analysis` / `track_identity` /
`track_audio_features` / `track_mert_sections` / `track_sections` (per-ref analysis),
`canonical_track_cue_points` (cue-detr on `stem='regular'` only),
`track_fingerprints` (`recording_id`, `stem`) / `set_fingerprint_hits`,
`track_audio_correction` (replace/add ledger by axis), `set_section_alignment`
(legacy alignment-output table, currently unused), `measure_alignment`,
`set_playback_score`, `set_timeline`, `set_analysis`.

Schema lives in [web_crawler/database/schema.sql](web_crawler/database/schema.sql).

## Storage & cluster

The project runs across four machines (see [Makefile](Makefile) for cluster ops):

- **pi-storage** (Linux aarch64) — canonical state (DB + audio + stems) + scraper services + **CPU-side analysis** (yt-dlp downloads, beat_this, cue-detr, librosa, pyloudnorm). Long-running services live here. Reachable via Tailscale MagicDNS.
- **pi-worker** (Linux aarch64) — AJAX retry drain (`tracklist-ajax-retry.service`). Spare CPU available for batch CPU analysis when idle.
- **gpubox-rented GPU** (ephemeral) — **GPU-bound analysis** (stem separation, MERT) and **Essentia** (no aarch64 wheels — must run on x86_64). Job pulls audio from pi-storage over Tailscale, runs inference, writes results back, terminates. **Always rent + manage GPU boxes with `gpubox`** (`~/workspace/gpubox`) — the only sanctioned path; it rents from Vast.ai under the hood, races healthy offers, and guarantees guarded teardown. Never raw-`curl` the Vast API or ad-hoc rent. Cost target ≤$5–10 for the whole 16k-track corpus on a 3090/4090 spot. *(Separator: **Roformer** is current; **Demucs** is stale/legacy — see [analysis/CLAUDE.md](analysis/CLAUDE.md) "Stem-separation backends".)*
- **Mac** (Apple Silicon) — dev driver *and* a second analysis worker. The full Demucs/beat_this/cue-detr/MERT/Essentia pipeline runs locally on the MPS backend via [scripts/mac_analyze_loop.py](scripts/mac_analyze_loop.py) (the GPU counterpart, [analysis/gpu_worker.py](analysis/gpu_worker.py), runs on a `gpubox`-rented box); pulls audio from pi-storage over Tailscale, writes results back. Expect ~200–250 s/track vs ~85 s on a 4090 — useful when no GPU box is rented or for the long tail. Also drives the **labeling** workflow.

The full **"which dependency runs where"** split (Essentia/Demucs/MERT vs
beat_this/cue-detr/librosa) lives in [analysis/CLAUDE.md](analysis/CLAUDE.md).
The **audio-download topology** (yt-dlp / spotdl / YT Music) lives in
[ingest/CLAUDE.md](ingest/CLAUDE.md).

**Canonical paths on pi-storage:**

| Kind | Path |
|---|---|
| DB | `/mnt/storage/data/db/music_database.db` |
| Track audio | `/mnt/storage/objects/{track_id}/{track_id}__{platform}__{player_id}.{ext}` |
| Demucs stems | `/mnt/storage/stems/{track_audio_id}/{vocals,drums,bass,other,instrumental}.{ext}` |
| Essentia TF model cache | `/mnt/storage/data/essentia_models/*.pb` (synced from a gpubox GPU or fetched on first use) |

The repo's `data/db/music_database.db` is **a stale local copy for development — never the source of truth.** Services on pi-storage write to the canonical DB continuously; the local copy diverges quickly. To inspect canonical state, query pi-storage directly: `ssh pi-storage 'sqlite3 /mnt/storage/data/db/music_database.db "..."'` or via the FastAPI jobqueue.

**Pi-storage venvs:**
- `venvs/web_crawler/` — scraper, materializer, FastAPI jobqueue (BeautifulSoup, lxml, ddddocr, FastAPI).
- `venvs/audio/` — yt-dlp + spotdl for downloads, plus the CPU analysis stack (PyTorch CPU, beat_this, cue-detr, librosa, pyloudnorm). **Does not include Essentia / Demucs / MERT** — those run on a gpubox GPU or the Mac.

**Mac venvs (same names, different role):**
- `venvs/audio/` — same stack as pi-storage but with MPS-backed PyTorch, so beat_this / cue-detr / Demucs / MERT all run on Apple Silicon GPU.
- `venvs/essentia/` — Py3.13 sandbox holding the `essentia-tensorflow` wheel (macOS arm64 wheel exists; pi-storage's aarch64 Linux does not). Invoked as a subprocess from `venvs/audio/` whenever the analysis pipeline needs Essentia.

Use [Makefile](Makefile) for cluster ops (`make deploy`, `make status`, `make ssh-storage`).

> **Deploy caveat:** pi-storage systemd units that ran `python -m audio_pipeline.main` must be repointed to `ingest.main` (renamed out of `audio_pipeline/`) before `make deploy`, or services won't restart.

## How agents work here

Process rules — worktree isolation, branch+PR through the gate, SSOT for numbers,
coordinate before mutating live state — live in **[AGENTS.md](AGENTS.md)** (one page).
Read it once. This section covers only the git specifics.

## Git workflow

The failure mode here is not bad merges — it is branches that are *started* and
never *finished*. So the protocol is two rituals, not a style guide.

**Start ritual — before substantive work, every time:**

```bash
# run from the main checkout root
git worktree add -b <type>/<slug> .claude/worktrees/<slug> origin/main
ln -s "$PWD/venvs" .claude/worktrees/<slug>/venvs   # gitignored; the gate needs it
make collide                                        # who else touches these files?
```

One agent, one worktree — never the primary checkout, never `main` (a PreToolUse
hook blocks edits while HEAD is `main`). Branch names are `<type>/<slug>` with a
conventional type (`feat`/`fix`/`chore`/`docs`/`refactor`). If `make collide`
shows a file you're about to change is already contested by 2+ branches, pick a
different slice or land the competing branch first.

**While working:** commit logical units with conventional prefixes — the default
"only commit when explicitly asked" rule is **overridden for this project**. One
feature per commit, not one-giant-blob. `make check` must pass before every push;
never `--no-verify`. Keep the branch under **10 commits ahead of `origin/main`**
(the guardrails WIP fence trips above that): landing cost tracks
commits-since-divergence, not calendar age.

**Finish ritual — a branch is worth finishing or worth parking; leaving it is the
only strictly-negative option:**

1. Open the PR and land it. **One open PR per agent at a time** — finish before
   starting. Land smallest first; every commit that reaches `main` makes every
   other open branch worse.
2. After a rebase, `make land-verify` (git has no post-rebase hook).
3. `make gc-branches` — deletes merged local branches, lists idle worktrees and
   park candidates. `braid park <branch>` anything you're abandoning, then
   `git worktree remove <path>` (idle worktrees here cost 4–15 GB each and hold
   their branch hostage).

Full rules — isolation, the gate, SSOT for numbers, live-state coordination —
are in [AGENTS.md](AGENTS.md).

## Guardrails

Mechanical checks catch rename drift, stale module names, and wrong adapter path depth:

- **`make check`** — the pre-push gate: [scripts/guardrails.py](scripts/guardrails.py) (stale-name/path/dead-flag/WIP-limit + the [scripts/entropy_audit.py](scripts/entropy_audit.py) AST bug-class fences) + mypy (`scripts/typecheck.sh`) + the **full** pytest suite
- **`make check-fast`** — inner-loop check: guardrails (which rides the entropy_audit fences) + a curated, import-light pytest subset (~5s vs ~40s+). Cheap to run on every save; NOT a substitute for `make check` before a push
- **Git hooks** — one-time per clone: `git config core.hooksPath .githooks`
- **Cursor rules** — [.cursor/rules/](.cursor/rules/) (`identity-axes`, `repo-paths`) load when editing matching files
- **Refactor checklist** — [.claude/skills/refactor-safety/SKILL.md](.claude/skills/refactor-safety/SKILL.md) for module renames and directory splits
- **Branch hygiene** — [.claude/skills/branch-hygiene/SKILL.md](.claude/skills/branch-hygiene/SKILL.md); `make collide` / `make land-budget` before landing, `make land-verify` after a rebase. Merge cost tracks commits-since-divergence, not age
- **CI** — [.github/workflows/guardrails.yml](.github/workflows/guardrails.yml) on push/PR

## Environment

- Python project — no pyproject.toml, uses `requirements.txt` files
- `.env` file (loaded via python-dotenv) holds optional secrets (e.g. the
  email-based captcha fallback — see [web_crawler/CLAUDE.md](web_crawler/CLAUDE.md)).
  The default paths need no secrets.
- Virtual environments in `venvs/` (gitignored)
- `data/`, `profiles/`, `logs/` are gitignored — only `data/djs/*.json` job files are tracked
- Tests/imports run from repo root with `venvs/audio/bin/python`.

