# tracklist_engine

> A DAG pipeline that analyzes recorded DJ mixes against their scraped tracklists, building toward an automatic track aligner.

## What it does

Scrapes DJ set tracklists from the web, ingests and QA's the underlying track stems, and runs per-track audio analysis (Demucs separation, beat detection, cue-point detection, loudness, MERT embeddings). Humans then produce ground-truth labels by aligning stems against mixes in Ableton, which will eventually train an automatic aligner (not built yet — incubating in `workspaces/`). A separate personalization layer derives SoundCloud listener cohorts and per-user taste priors as a read-only export for a future learning repo.

## Structure

```
tracklist_engine/
├── core/             ← shared db, models, identity, result types
├── web_crawler/      ← scrape tracklists; FastAPI jobqueue + workers
├── ingest/           ← stem discovery, version/variant QA
├── analysis/         ← per-track audio analysis pipeline + adapters
├── labeling/         ← manual Ableton ground-truth production tools
├── tokenizer/        ← tokenize tracklists/tracks for the aligner
├── personalization/  ← SoundCloud cohorts + taste priors (off-DAG)
├── cue-detr/         ← vendored DETR cue-point model
├── eda/              ← cross-cutting analysis notebooks
├── workspaces/       ← experimental forks (aligner incubates here)
├── scripts/          ← Mac/Pi ops, migrations, batch jobs
├── deploy/           ← systemd service units
├── tests/            ← pytest suite
├── docs/             ← design plans, alignment objective, handoffs
├── config.yaml       ← paths + generator/scrape config
└── Makefile          ← Mac-side deploy/service ops over Tailscale
```

## Start here

- `CLAUDE.md` — the DAG (`scrape → ingest → analysis → labeling → alignment`) and the three-axis track identity model
- `docs/alignment_objective.md` — the aligner north star (target spec)
- `analysis/pipeline.py` — `analyze_track`: the core per-track analysis composition
- `web_crawler/main.py` — tracklist scraper entry point
- `personalization/main.py` — taste-prior scrape loop CLI

## How to run

```bash
# local guardrails + full test suite
make check
# deploy to both Pis (git pull + pip install)
make deploy
# service control (deliberate)
make start-scraper      # tracklist scraper
make restart-jobqueue   # FastAPI jobqueue on pi-storage
# run components directly
venvs/audio/bin/python -m pytest tests/ -q
```

## Stack

Python (FastAPI jobqueue, Playwright scraping, PyTorch/Demucs/MERT/cue-detr audio analysis), SQLite, deployed to Raspberry Pis over Tailscale.
