# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tracklist Engine is a pipeline for analyzing recorded DJ mixes against the
tracklists scraped for them. The chain has three stages, each in its own
top-level module:

1. **scrape** — `web_crawler/` extracts DJ set metadata, track listings, and
   streaming links from 1001Tracklists.com.

**Terminology — "labeling" vs "alignment" (do NOT conflate):**

- **labeling** = *manual* ground-truth production. A human aligns a set's
  stems against the mix in Ableton (`~/aligning/`, the `.als`), producing
  ground-truth labels. Tools: `pull_set_for_alignment.py`,
  `tag_aligning_folder.py`. Note many *existing* names use "align" for this
  (`~/aligning/`, the `set_section_alignment` table) — legacy, not renamed.
- **alignment** = *algorithmic* labeling — a model that learns to align
  automatically from the ground truth that manual labeling produces. The old
  `audio_pipeline/alignment/sota.py` (Viterbi) was the suboptimal incumbent —
  now removed (recoverable from git); the ML replacement is not built yet
  (incubates in `workspaces/`).

Target pipeline DAG (modularization in progress):
`core · scrape → ingest → analysis → labeling ⟶ (GT) ⟶ alignment`, with `eda/`
a cross-cutting consumer that reads from multiple stages (per-stage subfolders,
e.g. `eda/corpus/`, `eda/alignment/`).

Everything outside this chain is one of:
- A vendored dependency: `cue-detr/` (DETR-based cue-point detection model,
  consumed only by `audio_pipeline/analysis/canonical_cues.py`).
- Exploration / scratch: `eda/` notebooks.
- Experimental forks of chain modules: `workspaces/` (e.g.
  `workspaces/alignment_workbench` is a fork of `browser_daw/`). Promote a
  fork out of `workspaces/` when it stabilizes (same pattern used for
  `ui/` → `browser_daw/`).
- Archive: `archive/` (e.g. the legacy Streamlit alignment-review app).

New features land inside one of the three chain modules. New top-level folders
require explicit justification.

## Key Commands

### Web Crawler
```bash
pip install -r requirements.txt
playwright install chromium
python web_crawler/main.py          # Run scraper (config-driven via config.yaml)
```

### CUE-DETR (cue point detection)
```bash
pip install -r cue-detr/requirements.txt
python cue-detr/cue_points.py -t /path/to/audio/dir   # Predict cue points
# Flags: -c <checkpoint_dir>, -s <sensitivity>, -r <min_distance>, -p (print)
```

### Data Analysis
Jupyter notebooks in `eda/` — use `common.py` for shared DB access and DataFrame loading.

## Architecture

### Web Crawler (`web_crawler/`)
- **`main.py`** — Entry point. Loads DJ job files from `data/djs/*.json`, initializes DB, runs scraper.
- **`config.py`** — YAML config loader using dataclasses with a Result monad pattern for error handling.
- **`workers.py`** — Core scraping orchestration: page loads, captcha solving, AJAX media link fetching.
- **`scraper.py`** — HTML parsing: extracts set metadata, track info, media links from page content.
- **`database.py`** — SQLite interface. Schema lives in `web_crawler/database/schema.sql`.
- **`browser.py`** — Playwright browser context management with profile rotation.
- **`captcha_solver.py`** — Local CAPTCHA OCR via ddddocr (no API key, no network call). Optional `EmailCaptchaSolver` falls back to a human-in-the-loop email round-trip.
- **`data_models.py`** — Frozen dataclasses for type-safe immutable records (DJSet, DJSetMediaLink, etc.).

### CUE-DETR (`cue-detr/`)
DETR-based model for cue point detection in EDM tracks. Uses a custom COCO-like format with `position` instead of bounding boxes.
- `model/` — Training (`cue_detr_train.py` with W&B), evaluation, inference, data loading.
- `cue_points.py` — Main inference script. Downloads checkpoints from HuggingFace by default.
- Pretrained model: `disco-eth/cue-detr` on HuggingFace.

### Data Analysis (`eda/`)
- `eda.ipynb`, `error_analysis.ipynb`, `tokenizer.ipynb` — Exploratory analysis notebooks.
- `common.py` — Shared utilities for DB queries and pydantic_ai agent integration.

### MERT embedding choice

We use `m-a-p/MERT-v1-95M` at **hidden layer 6** (not the final layer) for
both analysis and alignment paths. The MERT paper shows mid-layers transfer
best to music-ID / structural-matching tasks; the top of the stack is more
tagging-oriented and the bottom too acoustic. Constant lives in
[audio_pipeline/analysis/adapters/mert_adapter.py](audio_pipeline/analysis/adapters/mert_adapter.py)
as `MERT_DEFAULT_LAYER`. (The legacy `mert_align.py` carried a duplicate
`DEFAULT_LAYER` that had to be kept in sync; it was removed with the old
aligner. The future aligner should import the constant from the adapter rather
than redefine it.)
When a learnable scoring head is added on top (post-ground-truth labeling),
replace the single-layer pick with a 13-channel learnable weighted sum
over all hidden states (SUPERB pattern) co-trained with the head.

**Backlog: upgrade to `m-a-p/MERT-v1-330M`.** The 330M variant has 24
transformer layers (vs 12 in 95M), and the deeper stack carries
task-specialized representations at well-defined depths:

| layer band | what it encodes | best for |
|---|---|---|
| 4–7   | low-level acoustic features | beat / tempo, onset detection |
| 8–13  | pitch + harmonic content    | key detection, chord recognition |
| 14–19 | timbre + instrumentation    | acapella-vs-instrumental discrimination, source-separation cues |
| 20–24 | high-level semantic         | genre, mood, structural segmentation |

For this pipeline, **don't pick a single layer** — use a learned
weighted sum across all 25 hidden states (the standard SSL probing
approach, SUPERB / s3prl pattern), co-trained with the scoring head.
That lets each downstream task pull from whichever band is most
informative, instead of forcing one mid-layer compromise across
beat/key/timbre/structure all at once.

Tradeoffs to plan for before flipping the constant:
- ~3.5× parameter count → ~3× inference time on MPS/CUDA. Vast cost
  is still bounded; Pi CPU becomes impractical (re-route 330M jobs to
  Mac MPS or Vast only).
- Cache key changes (layer-pick → weights identifier). The alignment
  cache must be flushed or namespaced when migrating.
- Frame rate (~75 Hz at 24 kHz) is unchanged; downstream measure-pooling
  code stays the same.

## Corpus empirics

Full write-ups (numbers, tables, modeling implications) plus the scripts
that produced them live in [eda/corpus_empirics/](eda/corpus_empirics/).
The findings document is [findings.md](eda/corpus_empirics/findings.md);
each section links to the reproducing script. Headline metrics are also
queryable from `data/analysis/aux.db` via the `analysis_results` table.

Findings, in dependency order:

1. **Acapella/instrumental era choice is orthogonal** — within a mashup
   slot, release-year of the two roles is independent (r ≈ 0). The
   pair-scoring head must not condition on year-proximity.
2. **Acapella choice IS driven by popularity** — acapellas are 3× more
   likely to be Hot 100 year-end hits and have ~200× more Last.fm
   listeners than the instrumentals. Treat the two roles with separate
   popularity priors.
3. **Set views are driven by chart-hit-vocal density** — ~39% of
   per-volume YouTube-views variance explained by acapella chart-rate +
   count. Instrumental popularity is neutral-to-negative.
4. **Peak position matters, breadth doesn't** — top-10 hit rate
   (r = +0.57) beats weekly chart presence; the predictive signal sharpens
   as the chart cut narrows toward "biggest at-release-time hits."
5. **Spotify Top 200 confirms the top-10 pattern** — combining Billboard
   + Spotify top-10 signals lifts R² to 0.44 (apparent ceiling for
   popularity features alone).
6. **Union coverage of popularity proxies** — ~61% of acapellas vs ~27%
   of instrumentals are caught by ≥1 popularity signal. 73% of BB
   instrumentals are obscure on every metric we have — picked for
   compatibility, not popularity.
7. **User-history is for the per-user model, not aggregate** — the
   remaining ~55% of aggregate-views variance is unmeasured production /
   viral / algorithmic factors, not individual taste. User-history data
   belongs in the personalized-inference head, not here.

The **`aux.db`** holding schema (release years, Last.fm, Billboard,
Spotify charts, BB-track ↔ chart-entry pairings, set views, headline
results) is documented at the bottom of [findings.md](eda/corpus_empirics/findings.md#auxiliary-research-database).
Rebuild via [aux_db_sync.py](eda/corpus_empirics/aux_db_sync.py).
Scripts assume `data/analysis/` and `data/db/` paths relative to repo root —
run them from the project root, not from the corpus_empirics folder.

## Configuration

All crawler behavior is controlled via `config.yaml`:
- **paths** — Data dirs, database location, logs, captcha images
- **generator** — Job selection (testing mode, filtering, limits)
- **timing** — Crawl delays (10s default) with jitter
- **browser** — Headless Chrome settings, viewport, timeouts
- **profiles** — Browser profile rotation (retirement after 750 sites)
- **failure** — Error handling modes (fail-fast, ajax_failure behavior, consecutive failure limits)
- **captcha** — Solver mode (ocr/continue/wait/kill), wait timeout, max OCR attempts

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
(legacy algorithmic-alignment output; its producer `sota.py` was removed), `measure_alignment`,
`set_playback_score`, `set_timeline`, `set_analysis`.

Schema lives in [web_crawler/database/schema.sql](web_crawler/database/schema.sql).

## Storage & cluster

The project runs across four machines (see [Makefile](Makefile) for cluster ops):

- **pi-storage** (Linux aarch64) — canonical state (DB + audio + stems) + scraper services + **CPU-side analysis** (yt-dlp downloads, beat_this, cue-detr, librosa, pyloudnorm). Long-running services live here. Reachable via Tailscale MagicDNS.
- **pi-worker** (Linux aarch64) — AJAX retry drain (`tracklist-ajax-retry.service`). Spare CPU available for batch CPU analysis when idle.
- **Vast.ai spot GPU** (rented, ephemeral) — **GPU-bound analysis** (Demucs, MERT) and **Essentia** (no aarch64 wheels — must run on x86_64). Job pulls audio from pi-storage over Tailscale, runs inference, writes results back, terminates. Cost target ≤$5–10 for the whole 16k-track corpus on a 3090/4090 spot.
- **Mac** (Apple Silicon) — dev driver *and* a second analysis worker. The full Demucs/beat_this/cue-detr/MERT/Essentia pipeline runs locally on the MPS backend via [scripts/mac_analyze_loop.py](scripts/mac_analyze_loop.py) (sibling of `vast_loop.py`); pulls audio from pi-storage over Tailscale, writes results back. Expect ~200–250 s/track vs ~85 s on a 4090 — useful when no Vast box is rented or for the long tail. Also drives the **alignment** workflow (see below).

**Analysis split (which dep runs where):**

| Component | Runs on | Why |
|---|---|---|
| yt-dlp downloads (production chain) | pi-storage | CPU-only; cross-arch wheels work. See "Audio downloading" below for the full topology — spotdl was removed from the main chain |
| beat_this (beats/downbeats) | pi-storage CPU **or** Mac MPS | PyTorch has aarch64 + MPS wheels; small model |
| cue-detr (EDM cues) | pi-storage CPU **or** Mac MPS | DETR transformer; small model |
| librosa, pyloudnorm | pi-storage **or** Mac | pure Python |
| **Essentia** (key/BPM/valence/mood/etc.) | **Vast.ai** *or* **Mac** | no aarch64 wheels — Essentia ships only x86_64 manylinux + macOS arm64, so the Mac has a `venvs/essentia/` Py3.13 sandbox and runs Essentia as a subprocess |
| **Demucs** stems | **Vast.ai** *or* **Mac MPS** | GPU-bound; ~30s/track on Pi CPU vs ~1s/track on 4090 vs ~3–5s/track on M-series MPS |
| **MERT** embeddings | **Vast.ai** *or* **Mac MPS** | same; [audio_pipeline/analysis/adapters/mert_adapter.py](audio_pipeline/analysis/adapters/mert_adapter.py) auto-selects `cuda` → `mps` → `cpu` |

The Mac mirrors the pi-storage CPU stack (`venvs/audio/`) plus the `venvs/essentia/` Py3.13 sandbox, so the **entire production analysis pipeline** is exercisable locally — not just for development, but as an actual production worker for batches that don't justify spinning up Vast.

**Audio downloading:**

The download topology is *not* "yt-dlp + spotdl in one chain" — it's a yt-dlp main path, a spotdl retry pass, and a YT Music rescue path. Three distinct entrypoints:

| Tool | Source for URLs | Fallback chain | When to use |
|---|---|---|---|
| [audio_pipeline/main.py](audio_pipeline/main.py) | scraped `dj_set_track_media_links` | `youtube → soundcloud` (see [main.py:76](audio_pipeline/main.py#L76)) | Production. Idempotent over `track_audio`, lands files at `{audio_root}/objects/{track_id}/{track_id}__{platform}__{player_id}.{ext}` |
| [audio_pipeline/main_retry.py](audio_pipeline/main_retry.py) | scraped Spotify URLs | spotdl only | Targeted retry on tracks with a Spotify URL but no `track_audio` row. Slow; needs real `SPOTIFY_CLIENT_ID`/`SECRET` (bundled spotdl creds are globally rate-limited) |
| [scripts/redownload_via_ytmusic.py](scripts/redownload_via_ytmusic.py) | metadata search (`full_name`) | YT Music → yt-dlp | Two-phase rescue: Phase 1 inserts `platform='youtube_music'` rows alongside existing yt-dlp ones; Phase 2 (gated by `--no-replace` default-off) deletes the noisy yt-dlp rows + cascades + unlinks files. Use after a corpus run to upgrade noisy 1001tracklists scrape URLs to clean Topic-channel masters |

Why spotdl is not in the main chain: a 14h production run produced **zero** successes and 174 timeouts ([main.py:65-75](audio_pipeline/main.py#L65)) — spotdl's anonymous YT Music search is rate-limited and slow without real Spotify creds. Inline comment explains the move.

yt-dlp specifics worth knowing: needs Netscape `cookies.txt` for ~5–15% age-gated YouTube ([downloader.py:61](audio_pipeline/adapters/downloader.py#L61)); needs a JS runtime (`node` or `nodejs` in PATH) to deobfuscate YouTube's n-parameter, otherwise stream URLs return only image formats ([downloader.py:35-43](audio_pipeline/adapters/downloader.py#L35)). The `feedback_ytdlp_bot_detection_recipe` memory has the recovery steps when these break.

One-off surgery: [scripts/replace_track_audio.py](scripts/replace_track_audio.py) — swap one track's audio by URL or local file. Destructive (deletes old row + cascades).

**Remix and version-qualifier handling (design rule, not a bug):**

A 1001tracklists track row like `Martin Garrix & Troye Sivan - There For You (Madison Mars Remix) (Instrumental) EPIC AMSTERDAM/STMPD` stores in `track_metadata` as:

```
title:        "There For You"
full_name:    "Martin Garrix & Troye Sivan - There For You (Madison Mars Remix)"
version_tag:  "Remix"
```

The `full_name` field comes from 1001tracklists' `<meta itemprop="name">` ([tokenizer/track_tokenizer.py:258](tokenizer/track_tokenizer.py#L258)) and **carries the remixer qualifier but never the vocal/instrumental qualifier or the label tag**. The tokenizer's `version_tag` enum is `Acappella | Rework | Remix | AltVersion | None` ([tokenizer/track_tokenizer.py:179-205](tokenizer/track_tokenizer.py#L179)) — there is intentionally no `Instrumental` category. The `(Instrumental)` qualifier and label tag are dropped at scrape time.

This is the **intended design**, on two axes:

1. **Remixer qualifier IS preserved in search**: [redownload_via_ytmusic.py:113](scripts/redownload_via_ytmusic.py#L113) sends `full_name` verbatim to YT Music, so the search hits the *Madison Mars Remix* release rather than the original Martin Garrix track. A bare `"Artist - Title"` search would silently resolve to the original — root cause of the corpus's variant-bleed bug, now fixed.
2. **Vocal/instrumental qualifier is deliberately NOT preserved**: YT Music's `filter='songs'` index doesn't reliably carry `(Instrumental)` variants as separate releases, and isolated-vocal/instrumental uploads are sparse and noisy. The system instead resolves to the canonical (vocal) master and lets Demucs extract stems downstream — `version_tag` tells the alignment-side code which stem to use without needing a separately-downloaded instrumental.

Carve-out for unknowns: when `full_name` contains 1001tracklists' "(ID Remix)" / "(ID Bootleg)" placeholders (meaning the remixer is unknown), the script strips back to `"Artist - Title"` since a literal "ID" in the YT Music query corrupts results ([redownload_via_ytmusic.py:70-76](scripts/redownload_via_ytmusic.py#L70)).

**Known scraper gap: sided rows with no `data-trackid` (the "Rvmor gap"):**

Some 1001tracklists `w/` rows (`data-isided="true"`) have a unique
per-set SoundCloud annotation but no global `data-trackid` HTML
attribute — typically obscure fan remixes that lack a 1001tracklists
global track entry. The scraper still extracts the link into
`dj_set_track_media_links` (with `track_id=NULL`), but the rest of the
chain drops it:

1. **Tokenizer** ([tokenizer/track_tokenizer.py:153](tokenizer/track_tokenizer.py#L153)) reads `data-trackid` to mint `track_key` (→ `track_id`). Missing attribute → no `track_metadata` row.
2. **Audio pipeline** keys on `track_id`. Missing → never downloaded into `track_audio`.
3. **Alignment pull** ([labeling/pull_set_for_alignment.py:211-218](labeling/pull_set_for_alignment.py#L211-L218)) filters `dj_set_rows` by `data_attrs_json LIKE '%trackid%'`. Missing → row skipped, slot invisible in manifest.

Field example: BB12 row 150 (`Porter Robinson & Madeon - Shelter (Rvmor Remix)`, tlp_id 2853054), SoundCloud-only (player_id 833168986). Currently handled by manually dropping the audio into `~/aligning/.../tracks/{slot}w{K}__...m4a` outside the canonical pipeline.

Proper fix would require minting a synthetic `track_id` (e.g. `tlp{tlp_id}`) and backfilling: `track_metadata`, `dj_set_track_media_links.track_id`, AND `dj_set_rows.data_attrs_json` (because of the pull-script filter above). Logical home for the synthesis is in the tokenizer when it sees an isolated `tlp_id` with media links but no `data-trackid` — emit a stable synthetic key tied to the tlp_id, mark the source as `synthetic` in `track_metadata`.

See [[project-tlp-gap]] memory and the field-evidence list in [[project-external-groundtruth]] for tracking instances.

**Manual labeling workflow (Mac-driven) — see [labeling/CLAUDE.md](labeling/CLAUDE.md).** Ground-truth production lives in the `labeling/` module: `pull_set_for_alignment.py` stages a set's mix + stems into `~/aligning/` for Ableton, `tag_aligning_folder.py` injects BPM/key tags. The consistency model (delta refresh, `--prune`), the annotator rename convention (`[NNNbpm KK]` / `[no-features]`), and the ephemeral-folder lifecycle are all documented there.

**Canonical paths on pi-storage:**

| Kind | Path |
|---|---|
| DB | `/mnt/storage/data/db/music_database.db` |
| Track audio | `/mnt/storage/objects/{track_id}/{track_id}__{platform}__{player_id}.{ext}` |
| Demucs stems | `/mnt/storage/stems/{track_audio_id}/{vocals,drums,bass,other,instrumental}.{ext}` |
| Human-readable library | `/mnt/storage/library/{Artist}/{Title}/...` (symlinks built by [library/builder.py](library/builder.py)) |
| Essentia TF model cache | `/mnt/storage/data/essentia_models/*.pb` (synced from Vast.ai or fetched on first use) |

The repo's `data/db/music_database.db` is **a stale local copy for development — never the source of truth.** Services on pi-storage write to the canonical DB continuously; the local copy diverges quickly. To inspect canonical state, query pi-storage directly: `ssh pi-storage 'sqlite3 /mnt/storage/data/db/music_database.db "..."'` or via the FastAPI jobqueue.

**Pi-storage venvs:**
- `venvs/web_crawler/` — scraper, materializer, FastAPI jobqueue (BeautifulSoup, lxml, ddddocr, FastAPI).
- `venvs/audio/` — yt-dlp + spotdl for downloads, plus the CPU analysis stack (PyTorch CPU, beat_this, cue-detr, librosa, pyloudnorm). **Does not include Essentia / Demucs / MERT** — those run on Vast.ai or the Mac.

**Mac venvs (same names, different role):**
- `venvs/audio/` — same stack as pi-storage but with MPS-backed PyTorch, so beat_this / cue-detr / Demucs / MERT all run on Apple Silicon GPU.
- `venvs/essentia/` — Py3.13 sandbox holding the `essentia-tensorflow` wheel (macOS arm64 wheel exists; pi-storage's aarch64 Linux does not). Invoked as a subprocess from `venvs/audio/` whenever the analysis pipeline needs Essentia.

Use [Makefile](Makefile) for cluster ops (`make deploy`, `make status`, `make ssh-storage`).

## Git workflow

Use your best judgement on when to commit and push. The default Claude Code rule "only commit when explicitly asked" is **overridden for this project** — proactively commit logical units of work and push them so pi-storage / pi-worker can pick them up via `make deploy`. Group changes into reviewable commits (one feature per commit, not one-giant-blob). Don't push directly to `main` if a pending change is still unstable; otherwise keep it moving.

## Environment

- Python project — no pyproject.toml, uses `requirements.txt` files
- `.env` file (loaded via python-dotenv) holds optional secrets for the email-based captcha fallback (CAPTCHA_EMAIL_SENDER / CAPTCHA_EMAIL_PASSWORD / etc.). The default OCR path needs no secrets.
- Virtual environments in `venvs/` (gitignored)
- `data/`, `profiles/`, `logs/` are gitignored — only `data/djs/*.json` job files are tracked

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
  returns a `Result` (`web_crawler/config.py`); CLI scripts and entrypoints
  exit on error with `sys.exit`. Don't retrofit monadic Results onto scripts.
