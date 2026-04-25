# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git hygiene — DO NOT lose work

`audio_pipeline/` is the main body of recent work and is historically UNTRACKED in git.
Before any bulk delete, refactor, or "prune" — verify the files are committed.

**Mandatory checks before destructive operations:**

1. Run `git status` and confirm the target files appear as tracked (no `??` prefix).
2. If `audio_pipeline/` (or any new module directory) is untracked, **commit it first**
   with a clear "checkpoint before refactor" message — even if the contents are
   work-in-progress. Uncommitted deletions are unrecoverable: `rm` bypasses
   `~/.Trash`, Time Machine, and APFS local snapshots.
3. After committing the checkpoint, proceed with the destructive change.
4. Never run `rm -rf` / bulk `rm` on a directory without first running
   `git ls-files <dir>` to confirm what git actually tracks.

SOTA alignment code was lost once (2026-04-22) because I pruned untracked modules
without this check. The files were not in `~/.Trash`, VSCode backups, or APFS
snapshots. Do not repeat.

**Commit policy for ongoing work:** when the user asks for a feature build,
commit a "WIP checkpoint" before starting any prune/refactor that touches
existing modules. This is cheap insurance and makes every delete reversible.

## Project Overview

Tracklist Engine is a pipeline for analyzing recorded DJ mixes against the
tracklists scraped for them. The chain has three stages, each in its own
top-level module:

1. **scrape** — `web_crawler/` extracts DJ set metadata, track listings, and
   streaming links from 1001Tracklists.com.
2. **align** — `audio_pipeline/` downloads each scraped track, runs Demucs
   stems + MIR + MERT analysis, and aligns the recorded mix audio against the
   tracklist via Viterbi (see [docs/SOTA.md](docs/SOTA.md)).
3. **view** — `browser_daw/` is the canonical viewer (frontend + backend) for
   browsing aligned sets and clipping segments.

Everything outside this chain is one of:
- A vendored dependency: `cue-detr/` (DETR-based cue-point detection model,
  consumed only by `audio_pipeline/analysis/canonical_cues.py`).
- Exploration / scratch: `data_analysis/` notebooks.
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
Jupyter notebooks in `data_analysis/` — use `common.py` for shared DB access and DataFrame loading.

## Architecture

### Web Crawler (`web_crawler/`)
- **`main.py`** — Entry point. Loads DJ job files from `data/djs/*.json`, initializes DB, runs scraper.
- **`config.py`** — YAML config loader using dataclasses with a Result monad pattern for error handling.
- **`workers.py`** — Core scraping orchestration: page loads, captcha solving, AJAX media link fetching.
- **`scraper.py`** — HTML parsing: extracts set metadata, track info, media links from page content.
- **`database.py`** — SQLite interface. Schema lives in `web_crawler/database/schema.sql`.
- **`browser.py`** — Playwright browser context management with profile rotation.
- **`captcha_solver.py`** — Automated CAPTCHA solving via TwoCaptcha API.
- **`data_models.py`** — Frozen dataclasses for type-safe immutable records (DJSet, DJSetMediaLink, etc.).

### CUE-DETR (`cue-detr/`)
DETR-based model for cue point detection in EDM tracks. Uses a custom COCO-like format with `position` instead of bounding boxes.
- `model/` — Training (`cue_detr_train.py` with W&B), evaluation, inference, data loading.
- `cue_points.py` — Main inference script. Downloads checkpoints from HuggingFace by default.
- Pretrained model: `disco-eth/cue-detr` on HuggingFace.

### Data Analysis (`data_analysis/`)
- `eda.ipynb`, `error_analysis.ipynb`, `tokenizer.ipynb` — Exploratory analysis notebooks.
- `common.py` — Shared utilities for DB queries and pydantic_ai agent integration.

## Configuration

All crawler behavior is controlled via `config.yaml`:
- **paths** — Data dirs, database location, logs, captcha images
- **generator** — Job selection (testing mode, filtering, limits)
- **timing** — Crawl delays (10s default) with jitter
- **browser** — Headless Chrome settings, viewport, timeouts
- **profiles** — Browser profile rotation (retirement after 750 sites)
- **failure** — Error handling modes (fail-fast, ajax_failure behavior, consecutive failure limits)
- **captcha** — Solver mode (continue/wait/kill), TwoCaptcha API config

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
(sota.py output, `confidence_source='sota_v2'`), `measure_alignment`,
`set_playback_score`, `set_timeline`, `set_analysis`.

Schema lives in [web_crawler/database/schema.sql](web_crawler/database/schema.sql).

## Environment

- Python project — no pyproject.toml, uses `requirements.txt` files
- `.env` file expected for API keys (TwoCaptcha, etc.) — loaded via python-dotenv
- Virtual environments in `venvs/` (gitignored)
- `data/`, `profiles/`, `logs/` are gitignored — only `data/djs/*.json` job files are tracked

# Python Style Guide: Rust-Flavoured Functional Python

This document defines the programming style for all Python code in this project.
The guiding philosophies are:

- **Rust**: explicit over implicit, errors as values, ownership awareness
- **Lambda calculus**: pure functions, immutability, composition over mutation
- **Linear type theory**: resources are consumed, not shared; use-once semantics enforced structurally

---

## Core Principle: Errors Are Values

In **domain code** — parsing, validation, transformation, business logic — never
raise exceptions for expected failure cases. These functions return `Result[T, E]`
so that the call site is forced to handle both branches.

This rule does **not** apply at I/O boundaries. The stdlib and every third-party
library speaks exceptions, not `Result`. Do not wrap every `requests.get` or
`pathlib.read_text` call in a bare `try/except` to produce a `Result` — that
creates wrapping noise without structural benefit. Instead, write a thin adapter
at the seam between your domain and the library. See the
[Library Boundary Adapters](#library-boundary-adapters) section.

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import TypeVar, Generic, Callable

T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E")
F = TypeVar("F")


@dataclass(frozen=True)
class Ok(Generic[T]):
    value: T

    def map(self, f: Callable[[T], U]) -> "Ok[U]":
        return Ok(f(self.value))

    def flat_map(self, f: Callable[[T], "Result[U, E]"]) -> "Result[U, E]":
        return f(self.value)

    def map_err(self, f: Callable[[E], F]) -> "Ok[T]":
        return self

    def unwrap_or(self, default: T) -> T:
        return self.value

    def is_ok(self) -> bool:
        return True


@dataclass(frozen=True)
class Err(Generic[E]):
    error: E

    def map(self, f: Callable) -> "Err[E]":
        return self

    def flat_map(self, f: Callable) -> "Err[E]":
        return self

    def map_err(self, f: Callable[[E], F]) -> "Err[F]":
        return Err(f(self.error))

    def unwrap_or(self, default: T) -> T:
        return default

    def is_ok(self) -> bool:
        return False


type Result[T, E] = Ok[T] | Err[E]
```

### Usage Pattern

```python
# WRONG — a domain function that raises instead of returning failure
def parse_int(s: str) -> int:
    return int(s)  # caller has no idea this can explode

# RIGHT — failure is explicit in the return type
def parse_int(s: str) -> Result[int, str]:
    try:
        return Ok(int(s))
    except ValueError:
        return Err(f"cannot parse {s!r} as integer")

# Chain operations without nested if-checks
result = (
    parse_int(raw)
    .map(lambda n: n * 2)
    .flat_map(validate_range)
    .unwrap_or(0)
)
```

---

## Option for Nullable Values

`None` is not used as a sentinel for missing values. Use `Option[T]`.

```python
type Option[T] = Ok[T] | Err[None]

def Some(value: T) -> Ok[T]:
    return Ok(value)

def Nothing() -> Err[None]:
    return Err(None)


# WRONG
def find_user(user_id: int) -> User | None:
    ...

# RIGHT
def find_user(user_id: int) -> Option[User]:
    user = db.get(user_id)
    return Some(user) if user else Nothing()
```

---

## Immutability by Default

All data structures are immutable unless mutation is explicitly required and
justified. Prefer `frozen=True` dataclasses and tuples over dicts and lists.

```python
# WRONG — mutable dataclass
@dataclass
class Config:
    host: str
    port: int

# RIGHT — frozen
@dataclass(frozen=True)
class Config:
    host: str
    port: int

# "Updating" a record produces a new one — Rust's struct update syntax
updated = replace(config, port=8080)  # from dataclasses import replace
```

For collections, prefer `tuple[T, ...]` over `list[T]` when the contents are
fixed after construction. Use `frozenset` over `set`.

```python
# Transformation returns new collection, never mutates in-place
def double_all(xs: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(x * 2 for x in xs)
```

---

## Pure Functions

A function should have no observable side effects unless it lives in an explicitly
designated I/O layer. Side effects include: writing to globals, mutating arguments,
printing, logging, network calls, and file I/O.

```python
# WRONG — mutates its argument
def normalize(record: dict) -> None:
    record["name"] = record["name"].strip()

# RIGHT — returns a new value
def normalize(record: Record) -> Record:
    return replace(record, name=record.name.strip())
```

I/O functions are explicitly named and typed to signal their impurity. They
live in a dedicated `effects` or `io` module and are called only from entry points.

```python
# effects.py — impure boundary
def write_record(path: Path, record: Record) -> Result[None, IOError]:
    try:
        path.write_text(record.to_json())
        return Ok(None)
    except IOError as e:
        return Err(e)
```

---

## Library Boundary Adapters

The boundary between domain code and external libraries (stdlib, HTTP clients,
ORMs, etc.) is the one place where `try/except → Result` conversion belongs.
Write a single adapter function per library concern. Domain code calls the adapter,
never the library directly.

The adapter's job is to:
1. Call the library (which may raise)
2. Catch the specific exceptions that library documents as expected
3. Convert them to a typed domain error
4. Let unexpected exceptions propagate — they are programmer errors, not recoverable cases

```python
# adapters/http.py — owns the requests import and its failure modes

import requests
from result import Ok, Err, Result  # or your local Result type
from domain.errors import HttpError

def get(url: str, timeout: float = 10.0) -> Result[bytes, HttpError]:
    """Adapter: converts requests exceptions into domain Result."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return Ok(response.content)
    except requests.Timeout:
        return Err(HttpError(kind="timeout", url=url, status=None))
    except requests.HTTPError as e:
        return Err(HttpError(kind="http", url=url, status=e.response.status_code))
    except requests.ConnectionError:
        return Err(HttpError(kind="connection", url=url, status=None))
    # requests.RequestException and other unexpected errors propagate uncaught
```

Domain code then chains cleanly, never touching `requests` directly:

```python
# domain/pipeline.py — no try/except, no import of requests

from adapters.http import get as http_get

def fetch_schema(endpoint: str) -> Result[Schema, PipelineError]:
    return (
        http_get(endpoint)
        .flat_map(parse_json)
        .flat_map(validate_schema)
    )
```

The same pattern applies to database adapters, file I/O, and any other I/O
boundary. One adapter per library concern, one place where exceptions are caught
and named.

```python
# adapters/fs.py

from pathlib import Path
from domain.errors import FsError

def read_text(path: Path) -> Result[str, FsError]:
    try:
        return Ok(path.read_text())
    except FileNotFoundError:
        return Err(FsError(kind="not_found", path=path))
    except PermissionError:
        return Err(FsError(kind="permission", path=path))
```

### What NOT to do

```python
# WRONG — wrapping every call site individually, not at the boundary
def process(url: str) -> Result[Schema, str]:
    try:
        raw = requests.get(url).content   # library leaks into domain
    except Exception as e:
        return Err(str(e))                # bare Exception catch, string error
    return parse_json(raw)

# WRONG — domain function imports requests directly
from requests import get
def fetch(url: str) -> Result[bytes, str]:
    ...
```

---

## Function Composition

Prefer composing small, single-purpose functions over writing large procedures.
Use a `pipe` utility to express data pipelines left-to-right.

```python
from functools import reduce
from typing import Callable

def pipe(value: T, *fns: Callable) -> object:
    """Apply a sequence of functions left-to-right."""
    return reduce(lambda acc, f: f(acc), fns, value)


# Instead of deeply nested calls:
result = pipe(
    raw_input,
    parse_csv_line,
    lambda row: row.map(validate_fields),
    lambda row: row.flat_map(enrich_with_metadata),
    lambda row: row.map(to_domain_object),
)
```

Partial application is preferred over helper lambdas with repeated logic.

```python
from functools import partial

clamp_to_byte = partial(max, 0) | partial(min, 255)  # pseudocode pattern
# or more explicitly:
def clamp(lo: int, hi: int) -> Callable[[int], int]:
    return lambda x: max(lo, min(hi, x))

normalize_byte = clamp(0, 255)
```

---

## Linear Resource Management: Consume, Don't Share

Resources that carry state (file handles, DB connections, HTTP sessions) follow
linear-style discipline: they are opened once, passed through a pipeline, and
consumed at the end. They are **never** stored in globals or class attributes
that outlive their acquisition scope.

Use the `with` statement as the structural enforcement of single-use semantics.
No resource leaves its `with` block alive.

```python
# WRONG — resource escapes its scope
conn = acquire_connection()
do_work(conn)
conn.close()  # easy to forget, error paths skip this

# RIGHT — linear scope enforced by context manager
def process(query: str) -> Result[Rows, DbError]:
    with acquire_connection() as conn:   # acquired here
        return execute(conn, query)      # consumed here
    # conn is closed — no escape


# WRONG — storing a resource on self for "convenience"
class Service:
    def __init__(self):
        self.conn = acquire_connection()  # lives forever, leaks

# RIGHT — acquire at call time, release at return
class Service:
    def query(self, sql: str) -> Result[Rows, DbError]:
        with acquire_connection() as conn:
            return execute(conn, sql)
```

For resources that must be threaded through multiple steps, model them as an
explicit parameter — like Rust's owned values passed by move.

```python
# The resource is a parameter, not ambient state
def step_one(conn: Connection, data: Data) -> tuple[Connection, Processed]:
    result = conn.fetch(data.query)
    return conn, Processed(result)

def step_two(conn: Connection, processed: Processed) -> Result[None, DbError]:
    return conn.write(processed.output)

# At the call site: conn flows through, never duplicated
with acquire_connection() as conn:
    conn, processed = step_one(conn, data)
    result = step_two(conn, processed)
```

---

## Pattern Matching Over isinstance Chains

Use Python 3.10+ `match` for exhaustive handling of sum types. The goal is that
adding a new variant to a union forces compiler-visible (or at least grep-visible)
update sites.

```python
# WRONG
if isinstance(result, Ok):
    handle_ok(result.value)
elif isinstance(result, Err):
    handle_err(result.error)

# RIGHT
match result:
    case Ok(value):
        handle_ok(value)
    case Err(error):
        handle_err(error)
```

Always include an exhaustive branch or intentionally omit it to signal that
unmatched cases are programmer error.

---

## Type Annotations Are Mandatory

Every function signature is fully annotated. `Any` is banned except when
wrapping third-party code at an explicit boundary. `# type: ignore` requires
a comment explaining why.

```python
# WRONG
def process(data, config):
    ...

# RIGHT
def process(data: RawData, config: Config) -> Result[ProcessedData, ProcessingError]:
    ...
```

Use `TypeAlias` and `type` statements (Python 3.12+) to name complex types
rather than inlining them.

```python
type UserId = int
type Lookup[T] = Callable[[UserId], Option[T]]
```

---

## No Mutable Global State

Module-level mutable state is forbidden. Constants (truly constant) are typed
with `Final`.

```python
from typing import Final

MAX_RETRIES: Final = 3
DEFAULT_TIMEOUT: Final[float] = 30.0

# WRONG
_cache: dict = {}  # mutable global

# RIGHT — pass state explicitly, or use a frozen structure
@dataclass(frozen=True)
class AppState:
    cache: tuple[CacheEntry, ...]
    config: Config
```

---

## Error Type Design

Errors are domain types, not strings. Define a sealed error hierarchy per
module using `dataclass` + union.

```python
@dataclass(frozen=True)
class ParseError:
    raw: str
    reason: str

@dataclass(frozen=True)
class ValidationError:
    field: str
    constraint: str

@dataclass(frozen=True)
class NetworkError:
    status_code: int
    body: str

type PipelineError = ParseError | ValidationError | NetworkError

def describe_error(e: PipelineError) -> str:
    match e:
        case ParseError(raw, reason):
            return f"parse failed on {raw!r}: {reason}"
        case ValidationError(field, constraint):
            return f"field {field!r} violates {constraint!r}"
        case NetworkError(code, _):
            return f"upstream returned HTTP {code}"
```

---

## Summary Checklist

Before submitting any Python code, verify:

- [ ] Domain functions (parsing, validation, business logic) return `Result[T, E]`, not bare values or exceptions
- [ ] Library calls are wrapped in a dedicated adapter in `adapters/`; domain code never imports third-party libraries directly
- [ ] Adapters catch only the specific documented exceptions for a library — bare `except Exception` is forbidden
- [ ] `None` is never used as a sentinel; `Option[T]` is used instead
- [ ] No dataclass is mutable without explicit justification
- [ ] Pure functions are isolated from I/O; impure code lives in `effects/` or `adapters/`
- [ ] Resources are acquired and consumed within a single `with` block
- [ ] All signatures are fully type-annotated — no `Any`, no bare `dict` or `list`
- [ ] Pattern matching is used for union type dispatch
- [ ] No module-level mutable state


# Audio Alignment

In order for an audio alignment to work, a bare-minimum is it must match the .yaml file, indicating hand-written human annotated locations.

## SOTA algorithm — Viterbi (do NOT replace with anything else without re-evaluation)

The single canonical alignment entry point is **[audio_pipeline/alignment/sota.py](audio_pipeline/alignment/sota.py)**:

```bash
venvs/audio/bin/python -m audio_pipeline.alignment.sota --set-id <set_id>
```

Writes rows to `set_section_alignment` with `confidence_source='sota_v2'`, `section_idx = tracklist row_index`. The Streamlit "Alignment review" page reads ONLY this source. Validated on BB11 ground-truth: mean mix IoU **0.891** (vs 0.751 argmax baseline, 0.872 raw Phase 1).

**Two files to know:**
- [audio_pipeline/alignment/sota.py](audio_pipeline/alignment/sota.py) — canonical orchestrator. Loads every tracklist ref with audio + measures, runs the full Viterbi stack, persists.
- [audio_pipeline/alignment/indicators_debug.py](audio_pipeline/alignment/indicators_debug.py) — holds the Viterbi primitives (`viterbi_universe`, `ref_position_viterbi`, `_clean_path`, snap helpers) that `sota.py` imports. Also runs the IoU validation harness against the GT fixture. NOT a persistence writer.

See [docs/SOTA.md](docs/SOTA.md) for the full pipeline diagram (7 stages: stem-routed MERT → ref-position Viterbi → per-universe Viterbi with mutual exclusion → fingerprint anchors → 2-pass cross-universe full-track exclusion → earliest-near-cue cleanup → canonical-cue snap) and [docs/ROADMAP.md](docs/ROADMAP.md) "CURRENT SOTA" for context.

**Cue points are per canonical track, not per audio variant.** They live in `canonical_track_cue_points` (keyed by `track_id`), computed once on the full-song `variant_tag='original'` audio via [audio_pipeline/analysis/canonical_cues.py](audio_pipeline/analysis/canonical_cues.py) at `cue-detr sensitivity=0.5`. All variants (acapella / instrumental / full / remix) read the same cue list.

Dropped experiments are archived in [docs/alignment_archive.md](docs/alignment_archive.md). Do not re-try them without re-running eval and beating the SOTA baseline.

## Hand-annotated ground-truth yamls

Per-set hand-annotated play spans live at:

```
tests/fixtures/<set>_ground_truth.yaml       # e.g. tests/fixtures/bigbootie11_ground_truth.yaml
```

Each file lists the tracks actually played by the DJ (from the user's
Ableton session) with `set_start_s` / `set_end_s` and optional
`ref_segments` for loops/cutups. Schema is documented in the header
comment of every fixture. These are the primary regression anchor for
alignment — every algorithm change must be evaluated against them.

**Evaluation harness:**

```bash
./venvs/audio/bin/python -m audio_pipeline.alignment.eval --db data/db/music_database.db
```

Scores every yaml in `tests/fixtures/*_ground_truth.yaml` against the
current `set_section_alignment` rows. Reports mean mix IoU per set
and per-row (a reference/gold-standard metric: 1.0 = span matches
human annotation exactly, 0.0 = no overlap) plus span-inflation
(>>1 = over-reported, <<1 = under-reported). Implementation:
[audio_pipeline/alignment/eval.py](audio_pipeline/alignment/eval.py).

Any alignment change (new algorithm, tuning knob, pipeline rewire)
MUST be gated on the eval score moving up — or at worst, staying
flat on already-solved rows while unlocking previously-failed rows.
Run the eval before and after; include the delta in any commit
message or PR description that touches the alignment pipeline.