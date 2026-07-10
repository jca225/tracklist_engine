# tracklist_engine

> A DAG pipeline that analyzes recorded DJ mixes against their scraped tracklists, building toward an automatic track aligner — step 1 of a longer road to personalized mix generation.

## What it does

Scrapes DJ set tracklists from the web, ingests the underlying track audio (with version/variant/stem QA), and runs per-track audio analysis (Roformer stem separation, beat detection, cue-point detection, loudness, MERT embeddings). Humans produce ground-truth labels by aligning stems against mixes in Ableton (`labeling/`), and that ground truth trains the **automatic aligner** — the active research front, incubating in `workspaces/alignment_prototype/` (evaluation harness, three end-to-end drivers, fingerprint placement, loop-tracing, self-repeat "fiber" analysis). Two GT sets are fully labeled (BB11, BB12) and drive all evaluation.

A separate personalization layer (`personalization/`) derives SoundCloud listener cohorts and per-user taste priors as a read-only export for a future learning repo. It sits off the DAG.

The chain: `core · scrape → ingest → analysis → labeling ⟶ (GT) ⟶ alignment`

Two terms that look alike but must not be conflated:

- **labeling** = *manual* ground-truth production (a human in Ableton)
- **alignment** = *algorithmic* labeling (the model trained on that ground truth)

## Structure

```
tracklist_engine/
├── core/             ← shared db, models, identity (3-axis), result types
├── web_crawler/      ← scrape tracklists; FastAPI jobqueue + workers
├── ingest/           ← audio download topology + version/variant/stem QA
├── analysis/         ← per-track audio analysis pipeline + adapters
├── labeling/         ← manual Ableton ground-truth production (+ .als codec)
├── tokenizer/        ← scrape rows → track_metadata + set_track_slots
├── personalization/  ← SoundCloud cohorts + taste priors (off-DAG)
├── cue-detr/         ← vendored DETR cue-point model
├── eda/              ← cross-cutting analysis notebooks + corpus empirics
├── workspaces/       ← experimental forks (the aligner incubates here)
│   └── alignment_prototype/  ← harness, drivers, fibers, looptrace, agentic
├── scripts/          ← Mac/Pi ops, migrations, batch jobs, guardrails
├── deploy/           ← systemd service units
├── tests/            ← pytest suite
├── docs/             ← design plans, alignment objective, handoffs
├── config.yaml       ← paths + generator/scrape config
└── Makefile          ← cluster ops + alignment entrypoints
```

## Start here

- `CLAUDE.md` — the DAG, the three-axis track identity model (version × stem × variant), and the module index
- `docs/alignment_objective.md` — the aligner north star (target spec)
- `docs/architecture_north_star.md` — architecture map + Point A→B phases
- `workspaces/alignment_prototype/CLAUDE.md` — current aligner state
- `analysis/pipeline.py` — `analyze_track`: the core per-track analysis composition
- `web_crawler/main.py` — tracklist scraper entry point

## How to run

```bash
# local guardrails + fast test subset
make check
# data-side integrity invariants (corpus analogue of make check)
make check-corpus
# race the three e2e aligner drivers against ground truth
make race
# run the aligner kernel on one set
make align SET=<set_id>
# aligner scorecard vs GT
make scorecard
# deploy to both Pis (git pull + pip install)
make deploy
# full test suite
venvs/audio/bin/python -m pytest tests/ -q
```

## Infrastructure

Four machines over Tailscale (see `Makefile`):

- **pi-storage** — canonical state (SQLite DB + audio + stems), scraper services, CPU-side analysis. The repo's local `data/db/` copy is stale dev scratch — never the source of truth.
- **pi-worker** — AJAX retry drain + spare CPU.
- **Vast.ai spot GPU** (ephemeral) — stem separation, MERT, Essentia (x86_64-only).
- **Mac** — dev driver, MPS-backed analysis worker, and the Ableton labeling workflow.

## Stack

Python (FastAPI jobqueue, scraping via `web_crawler/`, PyTorch audio stack: Roformer separation, MERT embeddings, beat_this, cue-detr; Demucs is legacy), SQLite, Raspberry Pis + rented spot GPUs over Tailscale.
