# Tracklist Model Canonicalization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the 1001tracklists data model a single canonical home (`tokenizer/model.py`) and make `materialize.py` write DB rows through named-field dataclasses instead of fragile positional tuples — a behavior-preserving refactor, no schema or output change.

**Architecture:** Introduce one module that declares the persisted table-row structs (`SetTrackSlotRow`, `TrackSuggestionRow`) and re-exports the existing in-memory parse structs (`TrackRow`, `SuggestionRow`, `NoticeRow`, `IDTrack`) so there is one import surface for "the model." Convert the two flush functions to build rows from dataclass instances, deriving INSERT column order from the dataclass field order so column/tuple drift becomes impossible. A characterization test pins current `materialize` output first, so the refactor is provably behavior-preserving.

**Tech Stack:** Python 3.14 (`venvs/audio/bin/python`), stdlib `dataclasses` + `sqlite3`, pytest, BeautifulSoup (already used by the parsers).

## Global Constraints

- **Behavior-preserving only.** This plan MUST NOT change `schema.sql`, add columns, create tables, or alter what rows `materialize` writes. Lossless widening (new columns, `set_notices`, ID-lens wiring, `version_artist` population) and the `tokenizer/ → tracklist_interpreter/` folder rename are explicitly OUT OF SCOPE — separate follow-on plans.
- **No pi deploy.** Local-only. No touching the canonical DB. The pi runs `python -m tokenizer.materialize`; the module name and CLI MUST stay `tokenizer.materialize` with an unchanged signature.
- **Run everything with** `venvs/audio/bin/python` from repo root.
- **House style is functional** (module-level functions + frozen dataclasses), not OO. Match it. Do NOT introduce classes with methods for the row structs.
- **`make check` must stay green** (`scripts/guardrails.py` + fast pytest subset) before any commit that ends a task.
- Keep each change a reviewable commit; work on a branch off `main`, never push to `main`.

---

### Task 1: Characterization test — pin current `materialize` output

> **CRITICAL SETUP:** The local `data/db/music_database.db` is an intentional
> fail-loud text sentinel (the "local-DB footgun" was killed 2026-07-22) — it is
> NOT a database. Real `dj_set_rows` live only on pi-storage. So Step 0 pulls a
> small sample from pi-storage ONCE into a checked-in, gzipped fixture, and the
> tests build a temp DB from that fixture — hermetic, no live DB at run time.

**Files:**
- Create: `tests/tokenizer/__init__.py`
- Create: `tests/tokenizer/fixtures/dj_set_rows_sample.json.gz` (generated in Step 0)
- Create: `tests/tokenizer/conftest.py`
- Create: `tests/tokenizer/test_materialize_characterization.py`
- Read for reference: `tokenizer/materialize.py:231-413` (the `materialize` entrypoint + main loop)

**Interfaces:**
- Consumes: `tokenizer.materialize.materialize(db_path: Path, batch_size: int = 10_000) -> dict[str, int]`
- Produces: fixture builder `build_fixture_db(tmp_path) -> Path` and helper `snapshot(db_path) -> dict` used by later tasks to prove the refactor changed nothing.

- [ ] **Step 0: Pull the fixture sample from pi-storage ONCE (real `raw_html`, no fabrication)**

The parsers depend on exact 1001tracklists selectors, so HTML must be real. Pull a
few full sets' rows from the canonical DB and freeze them as a gzipped JSON fixture.
Run from repo root:

```bash
mkdir -p tests/tokenizer/fixtures
ssh pi-storage "sqlite3 -json /mnt/storage/data/db/music_database.db \
  \"SELECT row_id, set_id, row_index, raw_html FROM dj_set_rows
    WHERE set_id IN (SELECT set_id FROM dj_set_rows GROUP BY set_id
                     HAVING count(*) BETWEEN 20 AND 120 LIMIT 3)
    ORDER BY set_id, row_index;\"" \
  | gzip > tests/tokenizer/fixtures/dj_set_rows_sample.json.gz
# sanity: non-trivial size and row count
gzip -dc tests/tokenizer/fixtures/dj_set_rows_sample.json.gz | \
  venvs/audio/bin/python -c "import json,sys; d=json.load(sys.stdin); print(len(d),'rows')"
```

Expected: prints a few dozen rows. Commit the fixture with Step 5. (Choosing sets
with 20–120 rows ensures a mix of `tlpItem` + `sugTog` + `bItmH` without a huge blob.)

- [ ] **Step 1: Write the fixture builder (loads the checked-in sample — no live DB)**

Create `tests/tokenizer/conftest.py`:

```python
import gzip
import json
import sqlite3
from pathlib import Path
import pytest

_FIXTURE = Path("tests/tokenizer/fixtures/dj_set_rows_sample.json.gz")

def build_fixture_db(tmp_path: Path) -> Path:
    """Temp DB seeded from the checked-in dj_set_rows sample. Hermetic."""
    if not _FIXTURE.exists():
        pytest.skip(f"{_FIXTURE} missing — run Task 1 Step 0 to generate it")
    rows = json.loads(gzip.decompress(_FIXTURE.read_bytes()))
    if not rows:
        pytest.skip("empty fixture sample")
    dst = tmp_path / "fixture.db"
    conn = sqlite3.connect(dst)
    conn.executescript(
        "CREATE TABLE dj_set_rows (row_id INTEGER PRIMARY KEY, set_id TEXT, "
        "row_index INTEGER, raw_html TEXT);"
    )
    conn.executemany(
        "INSERT INTO dj_set_rows (row_id,set_id,row_index,raw_html) VALUES (?,?,?,?)",
        [(r["row_id"], r["set_id"], r["row_index"], r["raw_html"]) for r in rows],
    )
    conn.commit()
    conn.close()
    return dst

def snapshot(db_path: Path) -> dict:
    """Order-stable dump of the two output tables materialize populates."""
    conn = sqlite3.connect(db_path)
    try:
        out = {}
        for tbl, order in (
            ("set_track_slots", "set_id,row_index"),
            ("track_suggestions", "sug_id"),
        ):
            cols = [c[1] for c in conn.execute(f"PRAGMA table_info({tbl})")]
            data_cols = [c for c in cols if c != "parsed_at"]  # drop nondeterministic timestamp
            sel = ",".join(data_cols)
            out[tbl] = conn.execute(f"SELECT {sel} FROM {tbl} ORDER BY {order}").fetchall()
        return out
    finally:
        conn.close()

@pytest.fixture
def fixture_db(tmp_path):
    return build_fixture_db(tmp_path)
```

Also create empty `tests/tokenizer/__init__.py`.

- [ ] **Step 2: Write the characterization test**

`materialize` creates its own output tables via `_MATERIALIZE_DDL`, so the test only needs `dj_set_rows` seeded. Create `tests/tokenizer/test_materialize_characterization.py`:

```python
from pathlib import Path
from tokenizer.materialize import materialize
from tests.tokenizer.conftest import snapshot

def test_materialize_output_is_stable(fixture_db: Path):
    counts = materialize(fixture_db, batch_size=10_000)
    snap = snapshot(fixture_db)
    # Guard: the fixture actually exercised the slot path.
    assert counts["slot"] > 0, f"fixture produced no slots: {counts}"
    # Freeze the exact rows. On first green run, copy the printed repr into
    # GOLDEN below and switch the assertion to compare against it.
    print("SLOT_ROWS", len(snap["set_track_slots"]))
    print("SUG_ROWS", len(snap["track_suggestions"]))
    assert snap == snap  # placeholder; replaced in Step 4
```

- [ ] **Step 3: Run to confirm it exercises real rows**

Run: `venvs/audio/bin/python -m pytest tests/tokenizer/test_materialize_characterization.py -v -s`
Expected: PASS, with `SLOT_ROWS`/`SUG_ROWS` printed > 0. If it SKIPs (no dev DB), stop and tell the reviewer — this refactor needs the dev DB present as a fixture source.

- [ ] **Step 4: Freeze the golden snapshot**

Re-run capturing the snapshot, then persist it as a pickle next to the test so the golden survives across the refactor. Replace the test body with:

```python
import pickle
from pathlib import Path
from tokenizer.materialize import materialize
from tests.tokenizer.conftest import snapshot

_GOLDEN = Path("tests/tokenizer/golden_materialize.pkl")

def test_materialize_output_is_stable(fixture_db: Path):
    counts = materialize(fixture_db, batch_size=10_000)
    assert counts["slot"] > 0, f"fixture produced no slots: {counts}"
    snap = snapshot(fixture_db)
    if not _GOLDEN.exists():                      # first run: record
        _GOLDEN.write_bytes(pickle.dumps(snap))
    golden = pickle.loads(_GOLDEN.read_bytes())
    assert snap == golden, "materialize output changed vs golden snapshot"
```

Run once to record: `venvs/audio/bin/python -m pytest tests/tokenizer/test_materialize_characterization.py -v`
Run again to prove it compares: same command. Expected: PASS both times, `golden_materialize.pkl` now exists.

- [ ] **Step 5: Commit**

```bash
git add tests/tokenizer/__init__.py tests/tokenizer/conftest.py \
        tests/tokenizer/fixtures/dj_set_rows_sample.json.gz \
        tests/tokenizer/test_materialize_characterization.py tests/tokenizer/golden_materialize.pkl
git commit -m "test(tokenizer): characterization snapshot pinning materialize output"
```

---

### Task 2: Create `tokenizer/model.py` — the canonical model home

**Files:**
- Create: `tokenizer/model.py`
- Create: `tests/tokenizer/test_model.py`
- Read for reference: `tokenizer/materialize.py:170-194` (slot INSERT column order), `:136-147` (suggestion INSERT column order)

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class SetTrackSlotRow` with fields in EXACT INSERT column order: `set_id, row_index, tlp_id, recording_id, track_id, source, slot_label, is_concurrent, cue_seconds, cue_time_seconds, claimed_version, claimed_stem, claimed_variant, full_name, title, artists_json, duration_seconds, layer_role, constituents_json`
  - `@dataclass(frozen=True) class TrackSuggestionRow` with fields: `sug_id, set_id, tlp_id, pos, track_slug, track_display, artist_title, suggester_user_id, suggester_name, suggestion_timestamp, is_remix, has_youtube, has_soundcloud, has_spotify`
  - `columns(row_cls) -> tuple[str, ...]` and `as_row(instance) -> tuple` helpers
  - Re-exports: `TrackRow`, `SuggestionRow`, `NoticeRow`, `IDTrack` (from their current modules)

- [ ] **Step 1: Write the failing test**

Create `tests/tokenizer/test_model.py`:

```python
from dataclasses import fields
from tokenizer import model

SLOT_COLS = (
    "set_id","row_index","tlp_id","recording_id","track_id","source","slot_label",
    "is_concurrent","cue_seconds","cue_time_seconds","claimed_version",
    "claimed_stem","claimed_variant","full_name","title","artists_json",
    "duration_seconds","layer_role","constituents_json",
)
SUG_COLS = (
    "sug_id","set_id","tlp_id","pos","track_slug","track_display","artist_title",
    "suggester_user_id","suggester_name","suggestion_timestamp",
    "is_remix","has_youtube","has_soundcloud","has_spotify",
)

def test_slot_row_field_order_matches_insert():
    assert tuple(f.name for f in fields(model.SetTrackSlotRow)) == SLOT_COLS
    assert model.columns(model.SetTrackSlotRow) == SLOT_COLS

def test_suggestion_row_field_order_matches_insert():
    assert tuple(f.name for f in fields(model.TrackSuggestionRow)) == SUG_COLS

def test_as_row_returns_positional_tuple_in_field_order():
    r = model.SetTrackSlotRow(*range(len(SLOT_COLS)))
    assert model.as_row(r) == tuple(range(len(SLOT_COLS)))

def test_parse_structs_reexported():
    for name in ("TrackRow","SuggestionRow","NoticeRow","IDTrack"):
        assert hasattr(model, name)
```

- [ ] **Step 2: Run to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/tokenizer/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tokenizer.model'`

- [ ] **Step 3: Write `tokenizer/model.py`**

```python
"""Canonical 1001tracklists data model — one home for every struct.

Two tiers:
  * PERSISTED table rows (SetTrackSlotRow, TrackSuggestionRow): field order is
    the SQL INSERT column order, so materialize writes by name, never by a
    hand-maintained positional tuple.
  * IN-MEMORY parse objects (TrackRow, SuggestionRow, NoticeRow, IDTrack):
    re-exported from their parsers so callers import "the model" from one place.

Behavior-preserving: these mirror the CURRENT schema. Widening (new columns,
set_notices, ID-lens) is a separate plan — do not add fields here for it.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

# --- re-exported in-memory parse structs (single import surface) ---
from tokenizer.track_tokenizer import TrackRow
from tokenizer.suggestion_tokenizer import SuggestionRow
from tokenizer.text_tokenizer import TextRowToken as NoticeRow
from tokenizer.id_tokenizer import IDTrack

__all__ = [
    "SetTrackSlotRow", "TrackSuggestionRow", "columns", "as_row",
    "TrackRow", "SuggestionRow", "NoticeRow", "IDTrack",
]


@dataclass(frozen=True)
class SetTrackSlotRow:
    """One row of set_track_slots. Field order == INSERT column order."""
    set_id: str
    row_index: int
    tlp_id: int | None
    recording_id: str | None
    track_id: str
    source: str
    slot_label: str | None
    is_concurrent: int
    cue_seconds: int | None
    cue_time_seconds: int | None
    claimed_version: str | None
    claimed_stem: str
    claimed_variant: str
    full_name: str | None
    title: str | None
    artists_json: str | None
    duration_seconds: int | None
    layer_role: str | None
    constituents_json: str | None


@dataclass(frozen=True)
class TrackSuggestionRow:
    """One row of track_suggestions. Field order == INSERT column order."""
    sug_id: int | None
    set_id: str
    tlp_id: int | None
    pos: int | None
    track_slug: str | None
    track_display: str | None
    artist_title: str | None
    suggester_user_id: int | None
    suggester_name: str | None
    suggestion_timestamp: str | None
    is_remix: int | None
    has_youtube: int | None
    has_soundcloud: int | None
    has_spotify: int | None


def columns(row_cls) -> tuple[str, ...]:
    """The SQL column names for a row dataclass, in INSERT order."""
    return tuple(f.name for f in fields(row_cls))


def as_row(instance) -> tuple:
    """Positional value tuple in field order, for executemany()."""
    return tuple(getattr(instance, f.name) for f in fields(instance))
```

- [ ] **Step 4: Run to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/tokenizer/test_model.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add tokenizer/model.py tests/tokenizer/test_model.py
git commit -m "feat(tokenizer): canonical model.py — persisted row structs + parse re-exports"
```

---

### Task 3: Rewrite the flushers to consume dataclasses (kill positional tuples)

**Files:**
- Modify: `tokenizer/materialize.py:136-147` (`_flush_suggestions`), `:170-194` (`_flush_slots`), `:330-354` (slot append), `:363-382` (suggestion append)
- Read: `tokenizer/model.py` (from Task 2)

**Interfaces:**
- Consumes: `model.SetTrackSlotRow`, `model.TrackSuggestionRow`, `model.columns`, `model.as_row`
- Produces: `_flush_slots(conn, buf: list[SetTrackSlotRow])` and `_flush_suggestions(conn, buf: list[TrackSuggestionRow])` — buffers now hold dataclasses, not tuples.

> **PLAN CORRECTION (execution finding, 2026-07-23):** The original plan claimed the
> legacy `row[:17]` branch in `_flush_slots` was dead because `_MATERIALIZE_DDL` creates
> the current schema. FALSE — `_MATERIALIZE_DDL`'s inline `set_track_slots` is STALE (17
> columns; missing `layer_role` + `constituents_json` that `schema.sql:637-638` has). On a
> fresh DB (including the hermetic test) the stale DDL builds a 17-col table, so the legacy
> branch runs and the Task-1 golden was captured on that non-production path. Fix the stale
> DDL FIRST (Steps 0a/0b), recapture the golden on the real 19-col path, THEN the legacy
> branch is genuinely dead and safe to delete.

- [ ] **Step 0a: Fix the stale `_MATERIALIZE_DDL` to match `schema.sql`**

In `tokenizer/materialize.py`, the inline `set_track_slots` DDL is missing two columns.
Replace this line (currently ~line 217):

```python
    full_name TEXT, title TEXT, artists_json TEXT, duration_seconds INTEGER,
    parsed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
```

with (adds `layer_role` + `constituents_json`, matching `schema.sql:637-638`):

```python
    full_name TEXT, title TEXT, artists_json TEXT, duration_seconds INTEGER,
    layer_role TEXT, constituents_json TEXT,
    parsed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
```

Leave the `track_suggestions` inline DDL unchanged (its 14 columns already match the INSERT;
`parsed_at` is not written and the snapshot excludes it).

- [ ] **Step 0b: Recapture the golden on the real 19-col production path**

The DDL fix changes the fresh-DB path from 17 → 19 columns, so the Task-1 golden (captured on
the stale path) is now wrong and MUST be regenerated on this DDL-fixed but pre-conversion code.

```bash
rm tests/tokenizer/golden_materialize.pkl
venvs/audio/bin/python -m pytest tests/tokenizer/test_materialize_characterization.py -v   # records fresh golden
venvs/audio/bin/python -m pytest tests/tokenizer/test_materialize_characterization.py -v   # confirms it compares
```

Expected: PASS both times; the new golden's `set_track_slots` rows now carry 19 values
(including `layer_role`/`constituents_json`) — the production path. Verify drift detection
still bites (corrupt a value in the pkl → FAIL, restore → PASS). Commit this as its own step:

```bash
git add tokenizer/materialize.py tests/tokenizer/golden_materialize.pkl
git commit -m "fix(tokenizer): un-stale _MATERIALIZE_DDL to match schema.sql; recapture golden on 19-col path"
```

This commit is behavior-preserving in PRODUCTION (table pre-exists at 19 cols, so the fixed
inline DDL is a no-op there) and a bug FIX for fresh DBs (which previously silently dropped
`layer_role`/`constituents_json`). Now the legacy branch is genuinely dead.

- [ ] **Step 1: Add the import**

At `tokenizer/materialize.py:44` (near the existing `from tokenizer.track_tokenizer import TrackRow`), add:

```python
from tokenizer import model
```

- [ ] **Step 2: Rewrite `_flush_suggestions` to build SQL from the dataclass**

Replace the body of `_flush_suggestions` (currently lines 136-147) with:

```python
def _flush_suggestions(conn: sqlite3.Connection, buf: "list[model.TrackSuggestionRow]") -> None:
    if not buf:
        return
    cols = model.columns(model.TrackSuggestionRow)
    conn.executemany(
        f"INSERT OR REPLACE INTO track_suggestions ({','.join(cols)}) "
        f"VALUES ({','.join('?' * len(cols))})",
        [model.as_row(r) for r in buf],
    )
    conn.commit()
```

- [ ] **Step 3: Rewrite `_flush_slots` — remove the `row[:17]` positional slicing**

Replace the whole `_flush_slots` (currently lines 170-194, including the fragile PRAGMA/legacy-slice branch) with:

```python
def _flush_slots(conn: sqlite3.Connection, buf: "list[model.SetTrackSlotRow]") -> None:
    if not buf:
        return
    cols = model.columns(model.SetTrackSlotRow)
    conn.executemany(
        f"INSERT OR REPLACE INTO set_track_slots ({','.join(cols)}) "
        f"VALUES ({','.join('?' * len(cols))})",
        [model.as_row(r) for r in buf],
    )
    conn.commit()
```

Note: the old legacy-schema branch (`layer_role` absent) is dropped intentionally — after Step 0a un-staled `_MATERIALIZE_DDL`, a fresh DB now gets the 19-col table, so the modern branch is the only reachable path. The recaptured golden (Step 0b) guards this — it now pins the 19-col production path, so if this deletion changed anything the characterization test fails.

- [ ] **Step 4: Replace the slot tuple append with a dataclass**

At `tokenizer/materialize.py:330-354`, the `slot_buf.append((...))` currently appends a 19-value positional tuple. Replace `slot_buf.append(( ... ))` with a named construction:

```python
slot_buf.append(
    model.SetTrackSlotRow(
        set_id=sid,
        row_index=int(row["row_index"]),
        tlp_id=tr.data_id,
        recording_id=tr.track_key,
        track_id=tr.track_key,
        source=source,
        slot_label=label,
        is_concurrent=int(tr.is_concurrent),
        cue_seconds=tr.cue_seconds,
        cue_time_seconds=tr.cue_time_seconds,
        claimed_version=scrape_claimed_version(tr.version_tag),
        claimed_stem=claimed_stem,
        claimed_variant=derive_claimed_variant(tr.full_name),
        full_name=tr.full_name,
        title=tr.title,
        artists_json=(json.dumps(list(tr.artists), ensure_ascii=False)
                      if tr.artists else None),
        duration_seconds=tr.duration_seconds,
        layer_role=layer_role,
        constituents_json=None,
    )
)
```

- [ ] **Step 5: Replace the suggestion tuple append with a dataclass**

At `tokenizer/materialize.py:363-382`, replace `sug_buf.append(( ... ))` with:

```python
sug_buf.append(
    model.TrackSuggestionRow(
        sug_id=sug.sug_id,
        set_id=row["set_id"],
        tlp_id=sug.tlp_id,
        pos=sug.pos,
        track_slug=sug.track_slug,
        track_display=sug.track_display,
        artist_title=sug.artist_title,
        suggester_user_id=sug.suggester_user_id,
        suggester_name=sug.suggester_name,
        suggestion_timestamp=sug.suggestion_timestamp,
        is_remix=(int(bool(sug.is_remix)) if sug.is_remix is not None else None),
        has_youtube=int(bool(sug.has_youtube)),
        has_soundcloud=int(bool(sug.has_soundcloud)),
        has_spotify=int(bool(sug.has_spotify)),
    )
)
```

Note the field order here follows the CONSTRUCTOR (named args), so it is safe regardless of column order — but the values must match what the old tuple passed. Cross-check against the old `sug_buf.append` tuple: old positions were `sug_id, set_id, tlp_id, pos, track_slug, track_display, artist_title, suggester_user_id, suggester_name, suggestion_timestamp, is_remix, has_youtube, has_soundcloud, has_spotify`. (The old code omitted several — keep the exact same omissions; do NOT add fields.)

- [ ] **Step 6: Update the buffer type hints**

At `tokenizer/materialize.py:259-260`, change:

```python
slot_buf: list[model.SetTrackSlotRow] = []
sug_buf: list[model.TrackSuggestionRow] = []
```

- [ ] **Step 7: Run the characterization test — the whole point**

Run: `venvs/audio/bin/python -m pytest tests/tokenizer/ -v`
Expected: PASS. The characterization snapshot from Task 1 proves `materialize` produces byte-identical output after the tuple→dataclass conversion.

- [ ] **Step 8: Commit**

```bash
git add tokenizer/materialize.py
git commit -m "refactor(tokenizer): materialize writes named dataclasses, not positional tuples"
```

---

### Task 4: Drift guardrail — dataclass fields must match the live DDL

**Files:**
- Create: `tests/tokenizer/test_schema_alignment.py`
- Read: `web_crawler/database/schema.sql` (canonical `set_track_slots` / `track_suggestions` DDL)

**Interfaces:**
- Consumes: `model.columns`, `web_crawler/database/schema.sql`
- Produces: a test that fails if someone adds a DB column without adding the dataclass field (or vice versa) — the permanent replacement for the deleted "lossy" ambiguity.

- [ ] **Step 1: Write the failing test**

Create `tests/tokenizer/test_schema_alignment.py`:

```python
import sqlite3
from pathlib import Path
from tokenizer import model

_SCHEMA = Path("web_crawler/database/schema.sql")

def _table_columns(table: str) -> set[str]:
    conn = sqlite3.connect(":memory:")
    conn.executescript(_SCHEMA.read_text())
    cols = {c[1] for c in conn.execute(f"PRAGMA table_info({table})")}
    conn.close()
    return cols

def test_slot_row_is_subset_of_schema():
    # Every dataclass field must be a real column. (track_id/parsed_at etc. may
    # exist in schema without a struct field; the struct must not invent columns.)
    schema_cols = _table_columns("set_track_slots")
    missing = set(model.columns(model.SetTrackSlotRow)) - schema_cols
    assert not missing, f"struct fields not in set_track_slots schema: {missing}"

def test_suggestion_row_is_subset_of_schema():
    schema_cols = _table_columns("track_suggestions")
    missing = set(model.columns(model.TrackSuggestionRow)) - schema_cols
    assert not missing, f"struct fields not in track_suggestions schema: {missing}"
```

- [ ] **Step 2: Run to verify it passes on current code**

Run: `venvs/audio/bin/python -m pytest tests/tokenizer/test_schema_alignment.py -v`
Expected: PASS (both structs are subsets of the real schema today). If it FAILS, a struct field name is wrong — fix the field name in `model.py`, do not weaken the test.

- [ ] **Step 3: Commit**

```bash
git add tests/tokenizer/test_schema_alignment.py
git commit -m "test(tokenizer): guardrail — model structs must match schema columns"
```

---

### Task 5: Point parse-struct consumers at the canonical import surface

**Files:**
- Modify: `tokenizer/materialize.py:43-45` (imports)
- Read: `scripts/match_stem_library.py`, `scripts/ingest_stem_url.py` (external references found in scan — verify, don't touch unless they import the moved structs)

**Interfaces:**
- Consumes: `tokenizer.model` re-exports
- Produces: `materialize.py` imports parse structs via `model`, establishing the single surface. (Original `*_tokenizer.py` modules keep their definitions — model.py re-exports them, so nothing else breaks.)

- [ ] **Step 1: Switch materialize's struct imports to the canonical surface**

At `tokenizer/materialize.py:43-45`, keep the `parse_track_row`/`parse_suggestion_row` FUNCTION imports (those live in the parsers) but source the `TrackRow` TYPE from `model`:

```python
from tokenizer.track_tokenizer import parse_track_row as parse_track_main
from tokenizer.suggestion_tokenizer import parse_suggestion_row
from tokenizer.model import TrackRow  # canonical surface (re-export)
```

Remove the now-redundant `from tokenizer.track_tokenizer import TrackRow` line if present.

- [ ] **Step 2: Run the full tokenizer suite + guardrails**

Run: `venvs/audio/bin/python -m pytest tests/tokenizer/ -v`
Expected: PASS (characterization + model + schema-alignment all green)

Run: `make check`
Expected: guardrails + fast pytest subset green.

- [ ] **Step 3: Commit**

```bash
git add tokenizer/materialize.py
git commit -m "refactor(tokenizer): import model structs from the canonical surface"
```

---

## Out of scope (explicit — do NOT do in this plan)

These are the follow-on plans, each needing its own spec because each changes behavior and/or touches the canonical pi:

1. **Lossless widening** — `ALTER TABLE set_track_slots` (mashup/edge/ID-status columns), `ALTER TABLE track_suggestions` (data_type/polls/cue), `CREATE TABLE set_notices`, wire `parse_id_row` into the tlpItem branch, populate `recording.version_artist`. Needs a canonical-DB migration + coordinated pi deploy (pi ~92 behind, issue #73).
2. **Folder rename** `tokenizer/ → tracklist_interpreter/` — bundle with the already-queued `web_crawler/ → scrape/` rename to pay the pi-systemd-entrypoint coordination cost once. Use the `refactor-safety` skill.
3. **Physically relocating the parse structs** out of the four `*_tokenizer.py` files into `model.py` (vs today's re-export) — pure churn; only worth it during the folder rename.

## Self-Review

- **Spec coverage:** model.py home (Task 2) ✓; named-field writes killing positional tuples (Task 3) ✓; drift guardrail replacing the "lossy" ambiguity (Task 4) ✓; single import surface (Task 5) ✓; behavior-preserving proof (Task 1 characterization) ✓. Widening + rename correctly deferred.
- **Placeholder scan:** none — every step has concrete code/commands.
- **Type consistency:** `SetTrackSlotRow`/`TrackSuggestionRow`/`columns`/`as_row` names used identically across Tasks 2–5; field lists match the INSERT column lists read from `materialize.py:176-181` and `:141-144`.
- **Risk note:** The local `data/db/music_database.db` is a fail-loud text sentinel, NOT a database (confirmed 2026-07-23). The characterization safety net therefore depends on Task 1 Step 0 pulling a real sample from pi-storage into a checked-in gzipped fixture. pi-storage was confirmed reachable (`ssh pi-storage`, 1.4M `dj_set_rows`). If pi is unreachable at execution time, Step 0 cannot run and the plan is blocked until the fixture exists.
