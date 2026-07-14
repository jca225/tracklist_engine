# tracklist_engine

> A DAG pipeline that analyzes recorded DJ mixes against their scraped tracklists, building toward an automatic track aligner — step 1 of a longer road to personalized mix generation.

## What it does

Scrapes DJ set tracklists from the web, ingests the underlying track audio (with version/variant/stem QA), and runs per-track audio analysis (Roformer stem separation, beat detection, cue-point detection, loudness, MERT embeddings). Humans produce ground-truth labels by aligning stems against mixes in Ableton (`labeling/`), and that ground truth trains the **automatic aligner** — the active research front, incubating in `workspaces/alignment_prototype/`. Two GT sets are fully labeled (BB11, BB12) and drive all evaluation.

A separate personalization layer (`personalization/`) derives SoundCloud listener cohorts and per-user taste priors as a read-only export for a future learning repo. It sits off the DAG.

The chain: `core · scrape → ingest → analysis → labeling ⟶ (GT) ⟶ alignment`

Two terms that look alike but must not be conflated:

- **labeling** = *manual* ground-truth production (a human in Ableton)
- **alignment** = *algorithmic* labeling (the model trained on that ground truth)

## Structure

```
tracklist_engine/
├── core/                        ← shared substrate; imports nothing upward
│   ├── identity.py              ← RecordingAxes: the 3-axis identity key
│   ├── db.py                    ← SQLite access layer
│   ├── models.py                ← frozen-dataclass record types
│   ├── result.py                ← Result type (errors-as-values in library code)
│   ├── slot_inventory.py        ← derived layer_role (bed/payload/constituent/solo)
│   └── acquisition_case.py      ← acquisition decisions logged as cases
├── web_crawler/                 ← the scrape stage (pending rename to scrape/)
│   ├── main.py                  ← scraper entry point
│   ├── scraper.py               ← + browser.py, captcha_solver.py, workers.py
│   ├── jobqueue/                ← FastAPI jobqueue (serves the cluster on pi-storage)
│   └── database/schema.sql      ← THE schema (~25 tables, both table groups)
├── ingest/                      ← audio acquisition + QA gates
│   ├── main.py                  ← yt-dlp main download loop
│   ├── main_retry.py            ← spotdl retry / YT-Music rescue paths
│   ├── preflight.py             ← yt-dlp bot-detection + JS-runtime recovery
│   ├── identity_gate.py         ← + guards.py: wrong-version + duration gates
│   ├── stem_cascade.py          ← official acappella/instrumental discovery
│   └── corrections.py           ← replace/add ledger (track_audio_correction)
├── analysis/                    ← per-track/set MIR
│   ├── pipeline.py              ← analyze_track: the core composition
│   ├── adapters/                ← Roformer, MERT, beat_this, cue-detr, Essentia
│   ├── persistence.py           ← analysis-side DB writes (vs core/db.py)
│   ├── vast_worker.py           ← GPU loop on rented Vast.ai boxes
│   └── canonical_cues.py        ← cue-detr on stem='regular' refs only
├── labeling/                    ← manual Ableton ground-truth production
│   ├── pull_set_for_alignment.py← mix+refs+stems → ~/aligning/<set>/
│   ├── als/                     ← bidirectional .als codec (parse ∘ print = id)
│   ├── export_als_to_gt.py      ← session → set_ground_truth write-back
│   ├── gt_review_ui.py          ← ground-truth review interface
│   └── ground_truth/            ← + gt_review/, identity_overrides/, fixtures/
├── tokenizer/                   ← scrape rows → track_metadata + set_track_slots
│   ├── materialize.py           ← the writer (run after schema migrations)
│   ├── identity_axes.py         ← authoring home for version/stem/variant parsing
│   └── tokenizer.py             ← + track/text/suggestion tokenizers
├── personalization/             ← SoundCloud cohorts + taste priors (off-DAG)
│   ├── main.py                  ← taste-prior scrape loop CLI
│   ├── cohort_driver.py         ← listener-cohort construction
│   ├── prior_mert.py            ← per-user taste priors over MERT space
│   └── findings.md              ← layer-local findings ledger
├── cue-detr/                    ← vendored DETR cue-point model
├── eda/                         ← cross-cutting exploratory analysis
│   ├── corpus_empirics/         ← bb_*.py analyses + findings.md + aux.db
│   ├── alignment/               ← aligner-side EDA (generalization, etc.)
│   └── queries/                 ← reusable SQL
├── workspaces/                  ← experimental forks; promote out when stable
│   └── alignment_prototype/     ← THE aligner incubator
│       ├── harness/             ← driver-agnostic eval: probes (fp/HuBERT/chroma/
│       │                          continuity/path-decode) + merge + contract
│       ├── drivers/             ← 3 e2e drivers (agentic/classical/ml) + race.py
│       ├── agentic/             ← belief/events/actions/policy loop, DSP probes
│       ├── looptrace/           ← acappella loop-tracing decode
│       ├── fibers/              ← self-repeat structure (gates + evidence)
│       ├── trajectory/          ← trajectory decoder (+ neuro/: precision fusion)
│       ├── evals/               ← eval fixtures + scoring
│       └── attic/EXPERIMENTS.md ← closed-experiments ledger; READ BEFORE RE-TESTING
├── scripts/                     ← ops, migrations, batch jobs
│   ├── guardrails.py            ← mechanical checks (make check)
│   ├── corpus_integrity.py      ← data invariants (make check-corpus)
│   ├── mac_analyze_loop.py      ← Mac MPS analysis worker
│   ├── vast_loop.py             ← Vast.ai rent-run-terminate driver
│   └── migrations/              ← schema migration SQL
├── deploy/                      ← systemd service units
├── tests/                       ← pytest suite (run from repo root)
├── docs/                        ← design plans, north stars, agent handoffs
├── config.yaml                  ← paths + generator/scrape config
└── Makefile                     ← cluster ops + alignment entrypoints
```

New features land inside one of the chain modules; new top-level folders need explicit justification. Each module carries its own `CLAUDE.md` with stage-specific detail.

## Track identity: the three axes

Every recording is keyed on three orthogonal axes (plus optional remixer name). This is the repo's central data-model idea — most identity bugs come from conflating them.

| Axis | Values | Meaning | Lives on |
|------|--------|---------|----------|
| **version** | `original`, `remix`, `rework`, `altversion`, `edit`, `bootleg`, `mashup` | creative version | `track_metadata.version` |
| **stem** | `regular`, `acappella`, `instrumental` | vocal/instrumental form | `track_audio.stem` |
| **variant** | `regular`, `extended` | edit length | `track_audio.variant` |

Concatenated lookup key: `version__stem__variant` (e.g. `remix__acappella__extended`) via `RecordingAxes.key()` in `core/identity.py`.

Distinct layers that must not be merged:

- **Work / recording** — canonical identity (`work` + `recording` tables; `recording_id` ≈ legacy `track_id`).
- **Set claim** — `set_track_slots.claimed_*` = what the DJ *played*, per the scraped tracklist. May disagree with what we downloaded (`identity_mismatch` view flags this).
- **Download** — `track_audio` rows, one per platform rip; `is_reference` picks the analysis reference.
- **Separated stems** — `track_stems.stem_name` (`vocals`, `drums`, …) are Roformer outputs, unrelated to the identity `stem` axis despite the name.

## Data

SQLite, ~25 tables in two groups (schema: `web_crawler/database/schema.sql`):

- **Scraper tables** — `dj_sets`, `dj_set_crawls` (HTML snapshots, ETag-deduped), `dj_set_rows`, media links, `scrape_failures`.
- **Audio-pipeline tables** — identity (`work`/`recording`), audio (`set_audio`, `track_audio`, stems, beat grids), per-track analysis (`track_analysis`, `track_audio_features`, MERT sections, cue points, fingerprints), the per-set spine (`set_track_slots`), and manual GT (`set_ground_truth`).

**The canonical DB lives on pi-storage** (`/mnt/storage/data/db/music_database.db`), written continuously by services. The repo's `data/db/music_database.db` is a stale dev copy — never the source of truth. Query the real thing over SSH:

```bash
ssh pi-storage 'sqlite3 /mnt/storage/data/db/music_database.db "select count(*) from track_audio"'
```

Canonical object storage, also on pi-storage: track audio at `/mnt/storage/objects/{track_id}/…`, separated stems at `/mnt/storage/stems/{track_audio_id}/…`.

## The labeling loop (ground truth)

How a set becomes training data:

1. **Pull** — `labeling/pull_set_for_alignment.py` rsyncs the mix + reference tracks + stems into `~/aligning/<set>/` on the Mac.
2. **Align in Ableton** — a human places each track's audio against the mix in a `.als` session, warping to match. `labeling/als/` is a bidirectional `.als` codec (parse ∘ print = identity), so sessions are both a labeling UI and a machine-readable format.
3. **Write back** — `labeling/write_back_ground_truth.py` parses the session into `set_ground_truth`.

GT status: **BB11 and BB12 are done** and are the evaluation corpus. Rule of thumb from hand-aligning: acappella placements are precise, instrumentals looser — the label heuristics live in `docs/` and inform how the aligner should weight evidence per stem.

## The aligner (current state)

North star (`docs/alignment_objective.md`): consume `{tokenized tracklist, track audios, set audio}` → an Ableton-round-trippable structure, trained on manual GT. Kernel entrypoint: `make align SET=<id>`.

Everything experimental lives in `workspaces/alignment_prototype/`:

- **harness/** — driver-agnostic evaluation harness; `make race` races the end-to-end drivers against GT, `make scorecard` scores against GT.
- **drivers/** — three e2e drivers (agentic owns placement, ml owns ref-decode).
- **agentic/** — belief/events/actions/policy loop with DSP probes and learned probe precisions.
- **looptrace/** — loop-tracing decode for acappellas.
- **fibers/** — self-repeat structure (which parts of a track repeat itself), used as gates and evidence.
- Placement evidence that has earned its place: audio fingerprinting for fine placement, HuBERT features for vocal identity/offset (chroma fails on re-pitched acappellas), per-stem fingerprint matching for instrumentals.

Dead ends are recorded in `workspaces/alignment_prototype/attic/EXPERIMENTS.md` — **read it before re-testing an idea**; many plausible approaches are already ruled out with verdicts.

Corpus-level findings from exploratory analysis live in `lab/corpus_empirics/findings.md`.

## How to run

```bash
# local guardrails + typecheck + fast test subset
make check
# data-side integrity invariants (corpus analogue of make check)
make check-corpus
# race the e2e aligner drivers against ground truth
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

## Environment setup

- Python, no pyproject — `requirements*.txt` per stack (`requirements.txt` scraper, `-audio` analysis, `-essentia`, `-spotdl`, `-msst`).
- Virtualenvs in `venvs/` (gitignored). On the Mac: `venvs/audio/` (MPS-backed PyTorch — run tests and imports with `venvs/audio/bin/python` from repo root) and `venvs/essentia/` (Py3.13 sandbox for the `essentia-tensorflow` wheel, invoked as a subprocess).
- `.env` (python-dotenv) holds optional secrets; default paths need none.
- One-time per clone: `git config core.hooksPath .githooks` (pre-commit runs the guardrails).
- `data/`, `profiles/`, `logs/` are gitignored except tracked `data/djs/*.json` job files.

## Infrastructure

Four machines over Tailscale (ops via `Makefile`: `make status`, `make ssh-storage`, `make logs-*`):

- **pi-storage** — canonical state (DB + audio + stems), scraper services, CPU-side analysis (downloads, beat_this, cue-detr, librosa, loudness). Long-running services live here.
- **pi-worker** — AJAX retry drain + spare CPU for batch analysis.
- **Vast.ai spot GPU** (rented, ephemeral) — GPU-bound analysis (stem separation, MERT) and Essentia (x86_64-only wheels). Pulls audio from pi-storage, writes results back, terminates.
- **Mac** — dev driver, a second analysis worker on the MPS backend (`scripts/mac_analyze_loop.py`), and the Ableton labeling workflow.

## Guardrails

Mechanical checks catch rename drift, stale module names, and data-integrity regressions:

- `make check` — `scripts/guardrails.py` + typecheck + fast pytest, also run by the pre-commit hook and CI (`.github/workflows/guardrails.yml`).
- `make check-corpus` — `scripts/corpus_integrity.py`, identity/inventory invariants over the canonical DB; a daily watcher runs it on pi-storage.

## Stack

Python (FastAPI jobqueue, scraping via `web_crawler/`, PyTorch audio stack: Roformer separation, MERT embeddings, beat_this, cue-detr; Demucs is legacy), SQLite, SSH/rsync over Tailscale to Raspberry Pis + rented spot GPUs.

## Start here

- `CLAUDE.md` — the DAG, the identity model in full, and the per-module index
- `docs/alignment_objective.md` — the aligner north star (target spec)
- `docs/architecture_north_star.md` — architecture map + Point A→B phases
- `workspaces/alignment_prototype/CLAUDE.md` — current aligner state
- `analysis/pipeline.py` — `analyze_track`: the core per-track analysis composition
- `web_crawler/main.py` — tracklist scraper entry point
