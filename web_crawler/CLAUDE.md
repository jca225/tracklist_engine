# web_crawler/ — the scraper (scrape stage)

Extracts DJ set metadata, track listings, and streaming links from
1001Tracklists.com. This is the head of the chain
(`core · scrape → ingest → analysis → labeling ⟶ alignment`). It writes the
*scraper tables* (`dj_sets`, `dj_set_crawls`, `dj_set_rows`,
`dj_set_media_links`, `dj_set_track_media_links`, `scrape_failures`); the
tokenizer then materializes `dj_set_rows` into `track_metadata` (see
[tokenizer/CLAUDE.md](../tokenizer/CLAUDE.md)).

> Pending rename: this module will become `scrape/` (with `config.yaml` moved
> inside it). Deferred — it's the live scraper with several `config.yaml`
> path-readers, so its own careful slice.

## Run

```bash
pip install -r requirements.txt
playwright install chromium
python web_crawler/main.py          # config-driven via config.yaml
```

## Architecture

- **`main.py`** — Entry point. Loads DJ job files from `data/djs/*.json`, initializes DB, runs scraper.
- **`config.py`** — YAML config loader using dataclasses with a `Result` monad pattern for error handling (imports `core/result.py`).
- **`workers.py`** — Core scraping orchestration: page loads, captcha solving, AJAX media link fetching.
- **`scraper.py`** — HTML parsing: extracts set metadata, track info, media links from page content.
- **`database.py`** — SQLite interface. Schema lives in [database/schema.sql](database/schema.sql).
- **`browser.py`** — Playwright browser context management with profile rotation.
- **`captcha_solver.py`** — Local CAPTCHA OCR via ddddocr (no API key, no network call). Optional `EmailCaptchaSolver` falls back to a human-in-the-loop email round-trip.
- **`data_models.py`** — Frozen dataclasses for type-safe immutable records (DJSet, DJSetMediaLink, etc.).

## Configuration (`config.yaml`)

All crawler behavior is controlled via `config.yaml` at repo root:

- **paths** — Data dirs, database location, logs, captcha images
- **generator** — Job selection (testing mode, filtering, limits)
- **timing** — Crawl delays (10s default) with jitter
- **browser** — Headless Chrome settings, viewport, timeouts
- **profiles** — Browser profile rotation (retirement after 750 sites)
- **failure** — Error handling modes (fail-fast, ajax_failure behavior, consecutive failure limits)
- **captcha** — Solver mode (ocr/continue/wait/kill), wait timeout, max OCR attempts

## Captcha fallback secrets

The default OCR path needs no secrets. The optional email-based captcha
fallback reads `CAPTCHA_EMAIL_SENDER` / `CAPTCHA_EMAIL_PASSWORD` / etc. from
`.env` (loaded via python-dotenv).
