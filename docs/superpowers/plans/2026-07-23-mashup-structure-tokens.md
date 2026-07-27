# Mashup-Structure Tokens (WS-A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture 1001tracklists' explicit mashup-grouping attributes (`data-mashup`, `data-mashpos`, `data-subpos`) in the tokenizer and land them as additive `set_track_slots` columns, and populate the currently-NULL `constituents_json`, so the live slot-inventory / routing / scorer path (and later the aligner) reads 1001tracklists' own mashup structure instead of re-deriving it from noisy `w/`/`con` text.

**Architecture:** Two phases. **Phase A (Tasks 1–6, additive):** the tokenizer parses three new attributes into `TrackRow`; `materialize.py` writes them into three new nullable `set_track_slots` columns; a post-materialize pass groups each mashup parent (`data-mashup="N"`) with the N following `data-mashpos` constituent rows and writes their display names into `constituents_json`. Phase A changes nothing about which rows become slots or `layer_role`, so it lands the signal risk-free. **Phase B (Tasks 7–8, behavior change):** use the explicit attributes to fix the slot model — fold `data-mashpos` metadata rows into their parent instead of minting bogus `w{K}` slots, and refine `layer_role`/`is_concurrent`. Phase B *does* move the slot denominator, so it is **sequenced to run after the parallel RT1 agent lands** — it reconciles against the settled `fix/rt1-form-centric-remeasure` denominator rather than racing it.

**Implementation gate:** This whole plan executes **after the parallel RT1/denominator agent completes** (per the user). Do not start until that work is merged; then implement Phase A, then Phase B.

**Tech Stack:** Python 3, BeautifulSoup (`BS_PARSER`), SQLite, pytest. Run everything from repo root with `venvs/audio/bin/python`.

## Global Constraints

- **Phase A is additive; Phase B changes slot behavior — keep them in that order.** Tasks 1–6 must NOT modify `is_concurrent`, `_slot_label`, the `w{K}` counter (`materialize.py:302-312`), `derive_layer_role`, or which rows become slots — this keeps the signal-landing safe and independently shippable. Tasks 7–8 *do* change those, and run only after the parallel RT1 agent's denominator work is merged (the user has authorized touching slot behavior once that lands).
- **DDL is duplicated — change BOTH.** `set_track_slots` DDL lives in `tokenizer/materialize.py:210-220` (`_MATERIALIZE_DDL`) AND `web_crawler/database/schema.sql:619-643`. Both must gain the identical columns.
- **`core/` imports nothing upward** (substrate rule). New parsing stays in `tokenizer/`; no tokenizer/analysis imports into `core/`.
- **Canonical DB is on pi-storage (~8GB).** Schema change ships as `ALTER TABLE ADD COLUMN` (non-rewriting in SQLite) + a re-run of `python -m tokenizer.materialize`. Coordinate per AGENTS.md before running against canonical state; local dev + tests use in-memory / fixture DBs only.
- **Attribute semantics (verified against `data/html/1000d3bt_20260214T151317Z.html`):** a mashup **parent** row carries `data-subpos="true"` and `data-mashup="N"` (N = number of constituent rows that follow); each **constituent** row carries `data-mashpos="true"`, classes `con tgHid`, and frequently has NO `data-trackid` and a bogus `data-id`. Constituent identity is therefore stored as **display name**, not id.
- Existing columns keep their meaning: `constituents_json` comment in schema is "w/ recording_ids for primary rows" — we widen it to "constituent display names (or recording_ids when present) for mashup/w-parent rows".
- `make check` must pass before any PR.

---

### Task 1: Parse the three mashup attributes into `TrackRow`

**Files:**
- Modify: `tokenizer/track_tokenizer.py:40-52` (dataclass fields) and `tokenizer/track_tokenizer.py:228-234` (attribute reads)
- Test: `tests/tokenizer/test_mashup_attrs.py` (create)

**Interfaces:**
- Produces: `TrackRow.mashup_count: Optional[int]`, `TrackRow.is_mashpos: bool`, `TrackRow.is_subpos: bool`, set by `parse_track_row(row_html: str) -> TrackRow`.

- [ ] **Step 1: Write the failing test**

```python
# tests/tokenizer/test_mashup_attrs.py
from tokenizer.track_tokenizer import parse_track_row

PARENT = (
    '<div class="tlpTog bItm tlpItem subPosTog" data-id="7319917" '
    'data-trackid="135wzxcp" data-trno="2" data-mashup="1" data-subpos="true">'
    '<div class="bPlay"><span id="tlp0_tracknumber_value">03</span></div>'
    '<div class="bCont"><div class="fontL" '
    'itemtype="http://schema.org/MusicRecording">'
    '<meta itemprop="name" content="Calvin Harris vs. Jay Pryor - How Deep"/>'
    '<span class="trackValue">Calvin Harris vs. Jay Pryor - How Deep</span>'
    '</div></div></div>'
)
CONSTITUENT = (
    '<div class="tlpSubTog bItm tlpItem con tgHid" data-trno="3" data-id="3" '
    'data-mashpos="true">'
    '<div class="bCont"><div class="fontL" '
    'itemtype="http://schema.org/MusicRecording">'
    '<meta itemprop="name" content="Calvin Harris ft. Ina Wroldsen - How Deep"/>'
    '<span class="trackValue">Calvin Harris ft. Ina Wroldsen - How Deep</span>'
    '</div></div></div>'
)


def test_parent_carries_mashup_count_and_subpos():
    tr = parse_track_row(PARENT)
    assert tr.mashup_count == 1
    assert tr.is_subpos is True
    assert tr.is_mashpos is False


def test_constituent_carries_mashpos_flag_only():
    tr = parse_track_row(CONSTITUENT)
    assert tr.is_mashpos is True
    assert tr.mashup_count is None
    assert tr.is_subpos is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/tokenizer/test_mashup_attrs.py -v`
Expected: FAIL — `AttributeError: 'TrackRow' object has no attribute 'mashup_count'`

- [ ] **Step 3: Add the dataclass fields**

In `tokenizer/track_tokenizer.py`, after the `is_instrumental` field (line 52), add:

```python
    is_instrumental: bool = False               # mirrors claimed_stem == instrumental

    # mashup structure (1001tracklists explicit grouping — WS-A)
    mashup_count: Optional[int] = None          # data-mashup="N": parent owns N following data-mashpos rows
    is_mashpos: bool = False                    # data-mashpos="true": this row is a constituent of the mashup above
    is_subpos: bool = False                     # data-subpos="true": this row has collapsible sub-position children
```

- [ ] **Step 4: Read the attributes in `parse_track_row`**

In `tokenizer/track_tokenizer.py`, immediately after line 234 (`tr.is_ided = (row.get("data-isided") == "true")`), add:

```python
    # mashup grouping (explicit 1001tracklists structure)
    tr.mashup_count = _as_int(row.get("data-mashup"))
    tr.is_mashpos = (row.get("data-mashpos") == "true")
    tr.is_subpos = (row.get("data-subpos") == "true")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/tokenizer/test_mashup_attrs.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add tokenizer/track_tokenizer.py tests/tokenizer/test_mashup_attrs.py
git commit -m "feat(tokenizer): parse data-mashup/data-mashpos/data-subpos into TrackRow"
```

---

### Task 2: Add the three columns to both DDLs + a migration

**Files:**
- Modify: `web_crawler/database/schema.sql:619-643` (canonical DDL)
- Modify: `tokenizer/materialize.py:210-220` (`_MATERIALIZE_DDL`)
- Create: `scripts/migrations/migrate_mashup_structure.sql`
- Test: `tests/tokenizer/test_slot_schema_columns.py` (create)

**Interfaces:**
- Produces: `set_track_slots` columns `mashup_count INTEGER`, `is_mashpos INTEGER DEFAULT 0`, `is_subpos INTEGER DEFAULT 0`.

- [ ] **Step 1: Write the failing test**

```python
# tests/tokenizer/test_slot_schema_columns.py
import sqlite3
from tokenizer.materialize import _MATERIALIZE_DDL


def test_materialize_ddl_creates_mashup_columns():
    conn = sqlite3.connect(":memory:")
    conn.executescript(_MATERIALIZE_DDL)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(set_track_slots)")}
    assert {"mashup_count", "is_mashpos", "is_subpos"} <= cols
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/tokenizer/test_slot_schema_columns.py -v`
Expected: FAIL — assert `{...} <= cols` is False (columns absent)

- [ ] **Step 3: Add columns to `_MATERIALIZE_DDL`**

In `tokenizer/materialize.py`, inside the `set_track_slots` block, change the two lines ending the column list (currently `layer_role TEXT, constituents_json TEXT, parsed_at ...`):

```python
    full_name TEXT, title TEXT, artists_json TEXT, duration_seconds INTEGER,
    layer_role TEXT, constituents_json TEXT,
    mashup_count INTEGER, is_mashpos INTEGER DEFAULT 0, is_subpos INTEGER DEFAULT 0,
    parsed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (set_id, row_index)
);
```

- [ ] **Step 4: Add the same columns to `schema.sql`**

In `web_crawler/database/schema.sql`, in the `set_track_slots` table (after the `constituents_json` line at :639), add:

```sql
    layer_role        TEXT,                  -- bed | payload | constituent | solo
    constituents_json TEXT,                  -- constituent display names (or recording_ids) for mashup/w-parent rows
    mashup_count      INTEGER,               -- data-mashup="N": parent owns N following data-mashpos rows
    is_mashpos        INTEGER DEFAULT 0,     -- data-mashpos="true": row is a mashup constituent
    is_subpos         INTEGER DEFAULT 0,     -- data-subpos="true": row has collapsible sub-position children
    parsed_at         DATETIME DEFAULT CURRENT_TIMESTAMP,
```

- [ ] **Step 5: Write the migration (mirrors `migrate_layer_role.sql`)**

```sql
-- scripts/migrations/migrate_mashup_structure.sql
-- WS-A: explicit 1001tracklists mashup grouping on the slot spine.
-- Additive, non-rewriting. Run once against the canonical DB, then re-run
-- `python -m tokenizer.materialize` to backfill values.
ALTER TABLE set_track_slots ADD COLUMN mashup_count INTEGER;
ALTER TABLE set_track_slots ADD COLUMN is_mashpos INTEGER DEFAULT 0;
ALTER TABLE set_track_slots ADD COLUMN is_subpos INTEGER DEFAULT 0;
```

- [ ] **Step 6: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/tokenizer/test_slot_schema_columns.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add web_crawler/database/schema.sql tokenizer/materialize.py scripts/migrations/migrate_mashup_structure.sql tests/tokenizer/test_slot_schema_columns.py
git commit -m "feat(schema): add mashup_count/is_mashpos/is_subpos to set_track_slots"
```

---

### Task 3: Write the new columns in `materialize`

**Files:**
- Modify: `tokenizer/materialize.py:170-194` (`_flush_slots`) and `tokenizer/materialize.py:330-354` (buf tuple)
- Test: `tests/tokenizer/test_materialize_mashup.py` (create)

**Interfaces:**
- Consumes: `TrackRow.mashup_count/is_mashpos/is_subpos` (Task 1), the new columns (Task 2).
- Produces: materialized `set_track_slots` rows whose `mashup_count/is_mashpos/is_subpos` reflect the parsed attributes.

- [ ] **Step 1: Write the failing test**

```python
# tests/tokenizer/test_materialize_mashup.py
import sqlite3
from tokenizer.materialize import _MATERIALIZE_DDL, _flush_slots


def _base_slot(row_index, mashup_count, is_mashpos, is_subpos):
    # 22-field tuple matching the extended INSERT column order.
    return (
        "SET1", row_index, 1, "rid", "rid", "scraped", "001",
        0, None, None, "original", "regular", "regular",
        "A vs B - X", "X", None, None, "solo", None,
        mashup_count, is_mashpos, is_subpos,
    )


def test_flush_writes_mashup_columns():
    conn = sqlite3.connect(":memory:")
    conn.executescript(_MATERIALIZE_DDL)
    _flush_slots(conn, [
        _base_slot(0, 1, 0, 1),   # parent: mashup_count=1, is_subpos=1
        _base_slot(1, None, 1, 0),  # constituent: is_mashpos=1
    ])
    rows = list(conn.execute(
        "SELECT row_index, mashup_count, is_mashpos, is_subpos "
        "FROM set_track_slots ORDER BY row_index"
    ))
    assert rows[0] == (0, 1, 0, 1)
    assert rows[1] == (1, None, 1, 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/tokenizer/test_materialize_mashup.py -v`
Expected: FAIL — `sqlite3.OperationalError` (22 values vs 19-column INSERT) or column mismatch.

- [ ] **Step 3: Extend `_flush_slots` with a 22-column branch**

Replace the body of `_flush_slots` (`tokenizer/materialize.py:170-194`) with:

```python
def _flush_slots(conn: sqlite3.Connection, buf: list[tuple]) -> None:
    if not buf:
        return
    cols = {r[1] for r in conn.execute("PRAGMA table_info(set_track_slots)")}
    if "mashup_count" in cols:
        conn.executemany(
            "INSERT OR REPLACE INTO set_track_slots ("
            "set_id, row_index, tlp_id, recording_id, track_id, source, slot_label, "
            "is_concurrent, cue_seconds, cue_time_seconds, claimed_version, "
            "claimed_stem, claimed_variant, full_name, title, artists_json, "
            "duration_seconds, layer_role, constituents_json, "
            "mashup_count, is_mashpos, is_subpos"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            buf,
        )
    elif "layer_role" in cols:
        conn.executemany(
            "INSERT OR REPLACE INTO set_track_slots ("
            "set_id, row_index, tlp_id, recording_id, track_id, source, slot_label, "
            "is_concurrent, cue_seconds, cue_time_seconds, claimed_version, "
            "claimed_stem, claimed_variant, full_name, title, artists_json, "
            "duration_seconds, layer_role, constituents_json"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [row[:19] for row in buf],
        )
    else:
        conn.executemany(
            "INSERT OR REPLACE INTO set_track_slots ("
            "set_id, row_index, tlp_id, recording_id, track_id, source, slot_label, "
            "is_concurrent, cue_seconds, cue_time_seconds, claimed_version, "
            "claimed_stem, claimed_variant, full_name, title, artists_json, duration_seconds"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [row[:17] for row in buf],
        )
    conn.commit()
```

Note: the `layer_role` and legacy branches now slice `buf` (`[:19]` / `[:17]`) so callers can always emit the full 22-field tuple regardless of the destination DB's column set.

- [ ] **Step 4: Append the three values to the buf tuple**

In the streaming loop, the `slot_buf.append((...))` tuple ends with `layer_role,` then `None,` (constituents_json) at `tokenizer/materialize.py:351-352`. Change the tail of that tuple to:

```python
                                layer_role,
                                None,          # constituents_json — filled by backfill pass (Task 4)
                                tr.mashup_count,
                                int(tr.is_mashpos),
                                int(tr.is_subpos),
                            )
                        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/tokenizer/test_materialize_mashup.py tests/tokenizer/test_slot_schema_columns.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tokenizer/materialize.py tests/tokenizer/test_materialize_mashup.py
git commit -m "feat(materialize): write mashup_count/is_mashpos/is_subpos to slots"
```

---

### Task 4: Backfill `constituents_json` for mashup parents

**Files:**
- Modify: `tokenizer/materialize.py` (add `backfill_constituents(conn)`; call it after the main streaming loop, before the final `conn.close()`)
- Test: `tests/tokenizer/test_constituents_backfill.py` (create)

**Interfaces:**
- Consumes: materialized rows with `mashup_count`/`is_mashpos` (Task 3).
- Produces: `set_track_slots.constituents_json` = JSON array of constituent display names for each mashup parent. Read shape matches `core/slot_inventory.py:154-164` (`json.loads` → list of strings).

- [ ] **Step 1: Write the failing test**

```python
# tests/tokenizer/test_constituents_backfill.py
import json
import sqlite3
from tokenizer.materialize import _MATERIALIZE_DDL, _flush_slots, backfill_constituents


def _slot(row_index, name, mashup_count=None, is_mashpos=0):
    return (
        "SET1", row_index, None, "rid", "rid", "scraped",
        "001" if mashup_count is not None else None,
        0, None, None, "original", "regular", "regular",
        name, name, None, None, "solo", None,
        mashup_count, is_mashpos, 1 if mashup_count else 0,
    )


def test_backfill_attaches_constituent_names_to_parent():
    conn = sqlite3.connect(":memory:")
    conn.executescript(_MATERIALIZE_DDL)
    _flush_slots(conn, [
        _slot(10, "Calvin vs Jay - How Deep", mashup_count=2),
        _slot(11, "Calvin - How Deep (orig)", is_mashpos=1),
        _slot(12, "Jay Pryor - How Deep (edit)", is_mashpos=1),
        _slot(13, "Some Solo Track", mashup_count=None),
    ])
    backfill_constituents(conn)
    got = dict(conn.execute(
        "SELECT row_index, constituents_json FROM set_track_slots"
    ))
    assert json.loads(got[10]) == [
        "Calvin - How Deep (orig)", "Jay Pryor - How Deep (edit)"
    ]
    assert got[11] is None and got[13] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/tokenizer/test_constituents_backfill.py -v`
Expected: FAIL — `ImportError: cannot import name 'backfill_constituents'`

- [ ] **Step 3: Implement `backfill_constituents`**

Add to `tokenizer/materialize.py` (near `_flush_slots`):

```python
def backfill_constituents(conn: sqlite3.Connection) -> int:
    """Fill constituents_json for mashup parents from the following data-mashpos rows.

    A parent row has mashup_count=N; the next N rows (in row_index order, same
    set) with is_mashpos=1 are its constituents. Stores their display names.
    Returns the number of parent rows updated. Additive: touches only rows whose
    mashup_count is set; never changes slot_label/layer_role/is_concurrent.
    """
    updated = 0
    for (set_id,) in conn.execute(
        "SELECT DISTINCT set_id FROM set_track_slots WHERE mashup_count IS NOT NULL"
    ).fetchall():
        rows = conn.execute(
            "SELECT row_index, mashup_count, is_mashpos, "
            "COALESCE(full_name, title, '') AS name "
            "FROM set_track_slots WHERE set_id = ? ORDER BY row_index",
            (set_id,),
        ).fetchall()
        for i, r in enumerate(rows):
            n = r["mashup_count"]
            if not n:
                continue
            names: list[str] = []
            for follow in rows[i + 1:]:
                if len(names) >= n:
                    break
                if follow["is_mashpos"]:
                    names.append(follow["name"])
            if names:
                conn.execute(
                    "UPDATE set_track_slots SET constituents_json = ? "
                    "WHERE set_id = ? AND row_index = ?",
                    (json.dumps(names, ensure_ascii=False), set_id, r["row_index"]),
                )
                updated += 1
    conn.commit()
    return updated
```

- [ ] **Step 4: Call it after the main loop**

In `materialize()`, after the final `_flush_slots(conn, slot_buf)` / suggestion flush and before returning counts, add:

```python
    log.info("backfilling constituents_json for mashup parents")
    n_mashup = backfill_constituents(conn)
    log.info("constituents_json filled for %s mashup parents", f"{n_mashup:,}")
    counts["mashup_parents"] = n_mashup
```

- [ ] **Step 5: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/tokenizer/test_constituents_backfill.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tokenizer/materialize.py tests/tokenizer/test_constituents_backfill.py
git commit -m "feat(materialize): backfill constituents_json from data-mashpos grouping"
```

---

### Task 5: Expose the new columns to the alignment pull spine (WS-B seam)

**Files:**
- Modify: `labeling/aligning/pull_set_for_alignment.py:287-294` (slot spine SELECT — additive columns only)
- Test: `tests/labeling/test_pull_slot_select_columns.py` (create)

**Interfaces:**
- Consumes: the new `set_track_slots` columns.
- Produces: slot rows that carry `constituents_json` so `core/slot_inventory.py:slot_claim_from_row` can populate `constituent_ids`. This is the seam WS-B (aligner conditioning) builds on; it does NOT change pull behavior beyond adding columns.

- [ ] **Step 1: Write the failing test**

```python
# tests/labeling/test_pull_slot_select_columns.py
import re
from pathlib import Path


def test_slot_spine_select_includes_constituents():
    src = Path("labeling/aligning/pull_set_for_alignment.py").read_text()
    # The first-pass slot SELECT (from set_track_slots) must carry the new
    # structure columns so slot_claim_from_row can read constituents.
    m = re.search(r"FROM set_track_slots\s+WHERE set_id", src)
    assert m, "slot spine SELECT not found"
    window = src[max(0, m.start() - 400): m.start()]
    assert "constituents_json" in window
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/labeling/test_pull_slot_select_columns.py -v`
Expected: FAIL — `constituents_json` absent from the SELECT window.

- [ ] **Step 3: Add the columns to the slot spine SELECT**

In `labeling/aligning/pull_set_for_alignment.py`, change the SELECT at :288-294 to:

```python
    slot_rows = ssh_sqlite(f"""
        SELECT row_index, slot_label,
               COALESCE(recording_id, track_id) AS track_id,
               claimed_stem, claimed_variant,
               constituents_json, mashup_count, is_mashpos, is_subpos,
               COALESCE(full_name, title, '') AS slot_title
        FROM set_track_slots
        WHERE set_id = '{set_id}'
        ORDER BY row_index;
    """)
```

Leave the `labeled = [...]` comprehension (:297-308) unchanged — it selects a fixed tuple and is not affected by extra columns. (Threading these into `manifest.json`/`ManifestRow` and `infer.py` is WS-B, out of scope here.)

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/labeling/test_pull_slot_select_columns.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add labeling/aligning/pull_set_for_alignment.py tests/labeling/test_pull_slot_select_columns.py
git commit -m "feat(pull): carry mashup structure columns on the slot spine (WS-B seam)"
```

---

### Task 6: Full-suite gate + corpus smoke check

**Files:** none (verification only)

- [ ] **Step 1: Run the tokenizer + labeling test subset**

Run: `venvs/audio/bin/python -m pytest tests/tokenizer tests/labeling -q`
Expected: PASS (all)

- [ ] **Step 2: Run the guardrails**

Run: `make check`
Expected: PASS (no stale-name/path/dead-flag or entropy-fence failures)

- [ ] **Step 3: Smoke-materialize a real mashup snapshot into a temp DB**

Run:
```bash
venvs/audio/bin/python - <<'PY'
import sqlite3, json
from bs4 import BeautifulSoup
from tokenizer.track_tokenizer import parse_track_row
# Confirm the real fixture parses as expected (parent count + constituent flag).
html = open("data/html/1000d3bt_20260214T151317Z.html", encoding="utf-8", errors="replace").read()
soup = BeautifulSoup(html, "lxml")
tl = soup.find("div", id="tlTab")
rows = tl.find_all("div", recursive=False)
parents = [r for r in rows if r.get("data-mashup")]
assert parents, "expected at least one mashup parent in the fixture"
tr = parse_track_row(str(parents[0]))
print("parent mashup_count =", tr.mashup_count, "is_subpos =", tr.is_subpos)
assert tr.mashup_count and tr.mashup_count >= 1
print("OK")
PY
```
Expected: prints a `mashup_count >= 1` and `OK`.

- [ ] **Step 4: Commit any doc/index updates**

If `docs/design_docs_index.md` tracks plans, no entry needed (plans live under `docs/superpowers/plans/`). Otherwise nothing to commit.

---

---

## Phase B — fix the slot model (Tasks 7–8, runs after the parallel RT1 agent lands)

Phase B uses the now-captured attributes to correct a latent bug and de-noise `layer_role`. It moves the slot denominator, so it executes only after the RT1 denominator work is merged, and its verification compares against that settled baseline.

**Recommended rule (the principled distinction the explicit attributes finally allow):**
- A `con` row that is **`is_mashpos=1`** is *metadata* — a named component of one played mashup object (the parent). It has no independently-played audio and usually no real `data-trackid` (bogus `data-id` → today it mints a colliding synthetic `tlp{data_id}` slot). **It should NOT be its own slot;** it belongs in the parent's `constituents_json`.
- A `con` row that is a **genuine `w/` overlay** (track number shows `w/`, real `data-trackid`, `is_mashpos=0`) *is* separately-played audio. **It stays a slot** (current behavior).

This both removes bogus/colliding constituent slots from the denominator and keeps real overlays intact — a strictly more correct slot model.

### Task 7 (Phase B): Characterize mashpos-vs-overlay on real sets before changing behavior

**Files:** none (analysis only; write findings into this plan's PR description).

- [ ] **Step 1:** On BB11 (`2nvzlh2k`), BB12 (`1fsnxchk`), and a 500-set corpus sample, count from the materialized `set_track_slots` (post Phase A): (a) rows with `is_mashpos=1`, (b) how many of those have a real `track_id` (non-`tlp`-synthetic) vs bogus, (c) how many current `w{K}` slots are `is_mashpos=1` (candidates to fold) vs genuine `w/` overlays (`is_mashpos=0`).
- [ ] **Step 2:** Confirm the fold rule does not drop any row that has independently-resolvable audio (`track_audio` join on its `track_id`). If any `is_mashpos=1` row has real audio, widen the keep-as-slot condition and record it.
- [ ] **Step 3:** Record the exact denominator delta (slots removed per set) and reconcile it against the merged RT1 form-centric denominator — the two must agree on what a "played form" is.

### Task 8 (Phase B): Fold mashpos rows out of the slot spine; refine layer_role

**Files:**
- Modify: `tokenizer/materialize.py` (slot streaming loop :292-355; gather constituent names during parse instead of the post-hoc `backfill_constituents`, since folded rows no longer exist as slots)
- Modify: `core/slot_inventory.py:derive_layer_role` (use explicit flags when present)
- Test: `tests/tokenizer/test_mashpos_fold.py`, `tests/core/test_layer_role_explicit.py`

- [ ] **Step 1:** TDD the fold: a `is_mashpos=1 && not genuine-w/` row is captured into a per-set pending-parent buffer (name appended to the parent's constituents) and **not** appended to `slot_buf`; the `w_ctr` does not advance for it. Genuine `w/` overlays (`is_mashpos=0`, real `track_id`) still become `w{K}` slots. (Replaces Task 4's post-hoc backfill with in-loop gathering.)
- [ ] **Step 2:** TDD `derive_layer_role` to consume the explicit flags: `is_mashpos` → `constituent`; a `is_subpos` parent with `mashup_count` → still `bed`/`solo` by its own stem; drop the `w_idx==1 and stem==regular → payload` guess where an explicit signal exists.
- [ ] **Step 3:** Re-run the scorecard on BB11/BB12 and confirm the denominator matches Task 7's reconciled figure; commit with the before/after numbers.

*(Full TDD code for Phase B is intentionally written at implementation time, not now: the exact fold conditions depend on Task 7's empirical characterization and the merged RT1 denominator. Writing precise code before those facts exist would be speculative.)*

---

## Deferred (separate plans)

- **WS-B — aligner conditioning:** thread the new columns through `core/contracts/manifest.py:ManifestRow`, the manifest writer, and `alignment/infer.py:fetch_slot_rows`, then make the model condition on mashup structure. Task 5 lays the DB-read seam.
- **Other discarded tokens** (`rework of track X` source edge, `version_artist`, `feat.`/`featuring`, `radio` variant, `deleted` rows) — separate plans.

## pi-storage rollout (run once, coordinated)

1. Back up canonical DB.
2. `make deploy` (ships the code).
3. Apply migration: `ssh pi-storage 'sqlite3 /mnt/storage/data/db/music_database.db < scripts/migrations/migrate_mashup_structure.sql'` (or via the repo's migration runner).
4. Re-run `python -m tokenizer.materialize` against the canonical DB (rebuilds slots incl. new columns + constituents_json).
5. Spot-check BB11 (`2nvzlh2k`) / BB12 (`1fsnxchk`): `SELECT slot_label, mashup_count, is_mashpos, constituents_json FROM set_track_slots WHERE set_id='1fsnxchk' AND (mashup_count IS NOT NULL OR is_mashpos=1);`
