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

Everything outside this chain is one of:
- A vendored dependency: `cue-detr/` (DETR-based cue-point detection model,
  consumed only by `analysis/canonical_cues.py`).
- Exploration / scratch: `eda/` notebooks.
- Experimental forks of chain modules: `workspaces/` (e.g.
  `workspaces/alignment_workbench`). Promote a fork out of `workspaces/`
  when it stabilizes.

New features land inside one of the chain modules. New top-level folders
require explicit justification.

## Per-module guides

Each module carries its own `CLAUDE.md`, loaded on demand when you touch that
subtree — keep stage-specific detail there, not here. Index:

- **[web_crawler/CLAUDE.md](web_crawler/CLAUDE.md)** — the scraper (scrape
  stage): architecture, `config.yaml`, run command, captcha. *(Pending rename to
  `scrape/`.)*
- **[tokenizer/CLAUDE.md](tokenizer/CLAUDE.md)** — scrape-row → `track_metadata`
  materialization. **Authoring home** for the remix/version-qualifier rule and
  the "Rvmor gap" (sided rows with no `data-trackid`).
- **[ingest/CLAUDE.md](ingest/CLAUDE.md)** — audio download topology (yt-dlp main
  / spotdl retry / YT Music rescue), yt-dlp bot-detection + JS-runtime recipe.
- **[analysis/CLAUDE.md](analysis/CLAUDE.md)** — per-track/set MIR; MERT layer-6
  choice + 330M backlog; which dependency runs where (Essentia/Demucs/MERT/
  cue-detr/beat_this); `persistence.py` vs `core/db.py` boundary.
- **[labeling/CLAUDE.md](labeling/CLAUDE.md)** — manual ground-truth production:
  `pull_set_for_alignment.py` into `~/aligning/`, the consistency model
  (`--prune`), the annotator rename convention.
- **[eda/CLAUDE.md](eda/CLAUDE.md)** — exploratory analysis; corpus-empirics
  findings + `aux.db`. Full write-ups in
  [eda/corpus_empirics/findings.md](eda/corpus_empirics/findings.md).
- **[core/CLAUDE.md](core/CLAUDE.md)** — shared substrate; the rule that `core`
  imports nothing upward.

## Database

SQLite with ~25 tables split into two groups, all with cascade-deleting FKs.

**Scraper tables** (populated by `web_crawler/`):
`dj_sets` (canonical metadata), `dj_set_crawls` (HTML snapshots with ETag dedup),
`dj_set_media_links`, `dj_set_rows`, `dj_set_track_media_links`, `scrape_failures`.

**Audio-pipeline tables** (populated downstream of the scraper):
`set_audio` / `set_stems` / `set_measures` (mix-side audio + demucs stems + beat
grid), `track_audio` / `track_stems` / `track_measures` (ref-track equivalents),
`track_analysis` / `track_identity` / `track_audio_features` / `track_mert_sections` /
`track_sections` (per-ref analysis outputs), `canonical_track_cue_points`
(cue-detr cues keyed by track_id, full-song @ sensitivity=0.5), `track_fingerprints` /
`set_fingerprint_hits` (chromaprint ingestion + mix scan), `set_section_alignment`
(legacy alignment-output table, currently unused), `measure_alignment`,
`set_playback_score`, `set_timeline`, `set_analysis`.

Schema lives in [web_crawler/database/schema.sql](web_crawler/database/schema.sql).

## Storage & cluster

The project runs across four machines (see [Makefile](Makefile) for cluster ops):

- **pi-storage** (Linux aarch64) — canonical state (DB + audio + stems) + scraper services + **CPU-side analysis** (yt-dlp downloads, beat_this, cue-detr, librosa, pyloudnorm). Long-running services live here. Reachable via Tailscale MagicDNS.
- **pi-worker** (Linux aarch64) — AJAX retry drain (`tracklist-ajax-retry.service`). Spare CPU available for batch CPU analysis when idle.
- **Vast.ai spot GPU** (rented, ephemeral) — **GPU-bound analysis** (Demucs, MERT) and **Essentia** (no aarch64 wheels — must run on x86_64). Job pulls audio from pi-storage over Tailscale, runs inference, writes results back, terminates. Cost target ≤$5–10 for the whole 16k-track corpus on a 3090/4090 spot.
- **Mac** (Apple Silicon) — dev driver *and* a second analysis worker. The full Demucs/beat_this/cue-detr/MERT/Essentia pipeline runs locally on the MPS backend via [scripts/mac_analyze_loop.py](scripts/mac_analyze_loop.py) (sibling of [scripts/vast_loop.py](scripts/vast_loop.py)); pulls audio from pi-storage over Tailscale, writes results back. Expect ~200–250 s/track vs ~85 s on a 4090 — useful when no Vast box is rented or for the long tail. Also drives the **labeling** workflow.

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
| Essentia TF model cache | `/mnt/storage/data/essentia_models/*.pb` (synced from Vast.ai or fetched on first use) |

The repo's `data/db/music_database.db` is **a stale local copy for development — never the source of truth.** Services on pi-storage write to the canonical DB continuously; the local copy diverges quickly. To inspect canonical state, query pi-storage directly: `ssh pi-storage 'sqlite3 /mnt/storage/data/db/music_database.db "..."'` or via the FastAPI jobqueue.

**Pi-storage venvs:**
- `venvs/web_crawler/` — scraper, materializer, FastAPI jobqueue (BeautifulSoup, lxml, ddddocr, FastAPI).
- `venvs/audio/` — yt-dlp + spotdl for downloads, plus the CPU analysis stack (PyTorch CPU, beat_this, cue-detr, librosa, pyloudnorm). **Does not include Essentia / Demucs / MERT** — those run on Vast.ai or the Mac.

**Mac venvs (same names, different role):**
- `venvs/audio/` — same stack as pi-storage but with MPS-backed PyTorch, so beat_this / cue-detr / Demucs / MERT all run on Apple Silicon GPU.
- `venvs/essentia/` — Py3.13 sandbox holding the `essentia-tensorflow` wheel (macOS arm64 wheel exists; pi-storage's aarch64 Linux does not). Invoked as a subprocess from `venvs/audio/` whenever the analysis pipeline needs Essentia.

Use [Makefile](Makefile) for cluster ops (`make deploy`, `make status`, `make ssh-storage`).

> **Deploy caveat:** pi-storage systemd units that ran `python -m audio_pipeline.main` / `.vast_worker` must be repointed to `ingest.main` / `analysis.vast_worker` (renamed out of `audio_pipeline/`) before `make deploy`, or services won't restart.

## Git workflow

Use your best judgement on when to commit and push. The default Claude Code rule "only commit when explicitly asked" is **overridden for this project** — proactively commit logical units of work and push them so pi-storage / pi-worker can pick them up via `make deploy`. Group changes into reviewable commits (one feature per commit, not one-giant-blob). Don't push directly to `main` if a pending change is still unstable; otherwise keep it moving.

## Environment

- Python project — no pyproject.toml, uses `requirements.txt` files
- `.env` file (loaded via python-dotenv) holds optional secrets (e.g. the
  email-based captcha fallback — see [web_crawler/CLAUDE.md](web_crawler/CLAUDE.md)).
  The default paths need no secrets.
- Virtual environments in `venvs/` (gitignored)
- `data/`, `profiles/`, `logs/` are gitignored — only `data/djs/*.json` job files are tracked
- Tests/imports run from repo root with `venvs/audio/bin/python`.

# Python Style Guide: Rust-Flavoured Functional Python

The Python style for the project — "Rust-flavoured" in the parts the code
actually practices, not a strict regime:

- **Explicit & typed** — full type hints, `from __future__ import annotations`,
  explicit over clever.
- **Immutable** — frozen dataclasses for records (`data_models.py`, `StemRow`);
  construct new values rather than mutate in place.
- **Pure functions, composed** — small single-purpose functions assembled in a
  thin `main()` (e.g. `scripts/acquire_variant.py`); keep I/O at the edges.
- **Errors as values in core, fail-fast at the edge** — library/core code
  returns a `Result` (`core/result.py`); CLI scripts and entrypoints
  exit on error with `sys.exit`. Don't retrofit monadic Results onto scripts.
