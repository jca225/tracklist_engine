# Correction & Notice Layer (WS-C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist the 1001tracklists community-correction + notice + ID-resolution layer that the tokenizer already parses but `materialize.py` throws away — so "track wasn't played", "correct cue time", "wrong track", "rework of X", availability/section notices, and linked-ID resolution hints become queryable signal for the scorer/aligner.

**Architecture:** Three additive phases, each "call an existing parser + widen/add a table." **Phase 1 (suggestions):** widen `track_suggestions` with the fields `SuggestionRow` already parses (type/cue/poll/media/labels) and write them. **Phase 2 (notices):** call `text_tokenizer.parse_bItmH_row` (today `materialize.py:385` only counts `bItmH`) and store to a new `set_notices` table. **Phase 3 (ID workflow):** call `id_tokenizer.parse_track_row` for ID rows and populate the already-declared-but-empty `track_id_links` table plus a new `set_slot_id_meta` table (protected/rbcst/watchers/pre-save). No existing behavior changes; no `set_track_slots` column changes, so this is **independent of WS-A** (they touch disjoint tables and buffers).

**Tech Stack:** Python 3, BeautifulSoup, SQLite, pytest. Run from repo root with `venvs/audio/bin/python`.

## Global Constraints

- **Additive only.** New columns/tables; no change to `set_track_slots`, slot creation, `layer_role`, or the WS-A buffers. WS-C and WS-A can land in either order.
- **Reuse parsers as-is.** `SuggestionRow` (`tokenizer/suggestion_tokenizer.py:79`), `parse_bItmH_row(html) -> TextRowToken` (`tokenizer/text_tokenizer.py:268`), `parse_track_row(outer_div: Tag) -> IDTrack` (`tokenizer/id_tokenizer.py:125`) already produce everything needed — do NOT modify them. For ID rows, pass the **already-parsed `outer` Tag** to `id_tokenizer.parse_track_row` (no second BeautifulSoup parse).
- **DDL is duplicated — change BOTH** `tokenizer/materialize.py:201-228` (`_MATERIALIZE_DDL`) and `web_crawler/database/schema.sql`. `_flush_*` helpers PRAGMA-guard new columns (mirror `_flush_slots`) so an un-migrated DB still writes the legacy shape.
- **Canonical DB on pi-storage (~8GB):** ships as `ALTER TABLE ADD COLUMN` + `CREATE TABLE IF NOT EXISTS` + a re-run of `python -m tokenizer.materialize`. Coordinate per AGENTS.md; local dev + tests use in-memory DBs.
- **Calibration:** suggestions are *unresolved proposals*, not truth. Store them with their poll counts; downstream consumers weight by poll (accept/review/abstain), never apply blindly.
- `make check` must pass before any PR.

---

## Phase 1 — Suggestion payload

### Task 1: Widen `track_suggestions` + create `set_notices`, `set_slot_id_meta`; ensure `track_id_links` in the materialize DDL

**Files:**
- Modify: `tokenizer/materialize.py:201-228` (`_MATERIALIZE_DDL`) and the clean-rebuild `DELETE` block (`materialize.py:246-250`)
- Modify: `web_crawler/database/schema.sql` (`track_suggestions` at :675; add `set_notices`, `set_slot_id_meta`; `track_id_links` already exists at :700)
- Create: `scripts/migrations/migrate_correction_layer.sql`
- Test: `tests/tokenizer/test_correction_schema.py`

**Interfaces:**
- Produces tables: `track_suggestions` (+16 cols), `set_notices`, `set_slot_id_meta`, `track_id_links` (created by materialize too).

- [ ] **Step 1: Write the failing test**

```python
# tests/tokenizer/test_correction_schema.py
import sqlite3
from tokenizer.materialize import _MATERIALIZE_DDL


def _cols(conn, t):
    return {r[1] for r in conn.execute(f"PRAGMA table_info({t})")}


def test_correction_layer_tables_and_columns():
    conn = sqlite3.connect(":memory:")
    conn.executescript(_MATERIALIZE_DDL)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    assert {"set_notices", "set_slot_id_meta", "track_id_links"} <= tables
    sug = _cols(conn, "track_suggestions")
    assert {"data_type", "cue_seconds", "poll_correct", "poll_not_correct",
            "poll_unsure", "is_id_remix", "has_apple", "labels_json"} <= sug
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/tokenizer/test_correction_schema.py -v`
Expected: FAIL — tables/columns absent.

- [ ] **Step 3: Extend `_MATERIALIZE_DDL`**

In `tokenizer/materialize.py`, replace the `track_suggestions` block inside `_MATERIALIZE_DDL` and append the three tables:

```python
CREATE TABLE IF NOT EXISTS track_suggestions (
    sug_id INTEGER PRIMARY KEY, set_id TEXT NOT NULL, tlp_id INTEGER,
    pos INTEGER, track_slug TEXT, track_display TEXT, artist_title TEXT,
    suggester_user_id INTEGER, suggester_name TEXT,
    suggestion_timestamp TEXT, is_remix INTEGER, has_youtube INTEGER,
    has_soundcloud INTEGER, has_spotify INTEGER,
    data_type INTEGER, cue_seconds INTEGER, play_cue_seconds INTEGER,
    suggester_guest_id INTEGER, suggester_kind TEXT, track_page_path TEXT,
    track_id_numeric INTEGER, is_id_remix INTEGER, has_apple INTEGER,
    has_affiliate INTEGER, has_live_video INTEGER,
    poll_correct INTEGER, poll_not_correct INTEGER, poll_unsure INTEGER,
    labels_json TEXT, google_search_url TEXT
);
CREATE TABLE IF NOT EXISTS set_notices (
    set_id TEXT NOT NULL, row_index INTEGER NOT NULL,
    row_type TEXT, text TEXT, links_json TEXT, icons_json TEXT,
    parsed_json TEXT, parsed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (set_id, row_index)
);
CREATE TABLE IF NOT EXISTS set_slot_id_meta (
    set_id TEXT NOT NULL, row_index INTEGER NOT NULL, tlp_id INTEGER,
    is_id INTEGER DEFAULT 0, protected INTEGER DEFAULT 0, rbcst INTEGER DEFAULT 0,
    watchers INTEGER, presave_count INTEGER,
    parsed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (set_id, row_index)
);
CREATE TABLE IF NOT EXISTS track_id_links (
    set_id TEXT NOT NULL, tlp_id INTEGER NOT NULL,
    linker_user_name TEXT, linker_user_href TEXT, linker_user_followers TEXT,
    linked_tracklist_href TEXT, linked_tracklist_text TEXT,
    parsed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (set_id, tlp_id, linker_user_name, linked_tracklist_href)
);
```

- [ ] **Step 4: Delete the new tables in the clean-rebuild block**

In `materialize()`, the rebuild `DELETE` (`materialize.py:246-250`) currently clears `track_metadata`/`set_track_slots`/`track_suggestions` and deliberately leaves `track_id_links`. Change it to also clear the WS-C tables (materialize now owns `track_id_links`):

```python
    conn.executescript("""
        DELETE FROM track_metadata;
        DELETE FROM set_track_slots;
        DELETE FROM track_suggestions;
        DELETE FROM set_notices;
        DELETE FROM set_slot_id_meta;
        DELETE FROM track_id_links;
    """)
```

Also delete the stale comment above it that says `track_id_links` "is populated by a separate focused pass (to be written)".

- [ ] **Step 5: Mirror the changes in `schema.sql`**

In `web_crawler/database/schema.sql`, add the same 16 columns to the `track_suggestions` table (after `has_spotify INTEGER,` at :688), and add `set_notices` + `set_slot_id_meta` tables (near `track_id_links` at :700) with the identical DDL from Step 3.

- [ ] **Step 6: Write the migration**

```sql
-- scripts/migrations/migrate_correction_layer.sql
-- WS-C: persist the community-correction + notice + ID-resolution layer.
-- Additive, non-rewriting. Run once, then re-run `python -m tokenizer.materialize`.
ALTER TABLE track_suggestions ADD COLUMN data_type INTEGER;
ALTER TABLE track_suggestions ADD COLUMN cue_seconds INTEGER;
ALTER TABLE track_suggestions ADD COLUMN play_cue_seconds INTEGER;
ALTER TABLE track_suggestions ADD COLUMN suggester_guest_id INTEGER;
ALTER TABLE track_suggestions ADD COLUMN suggester_kind TEXT;
ALTER TABLE track_suggestions ADD COLUMN track_page_path TEXT;
ALTER TABLE track_suggestions ADD COLUMN track_id_numeric INTEGER;
ALTER TABLE track_suggestions ADD COLUMN is_id_remix INTEGER;
ALTER TABLE track_suggestions ADD COLUMN has_apple INTEGER;
ALTER TABLE track_suggestions ADD COLUMN has_affiliate INTEGER;
ALTER TABLE track_suggestions ADD COLUMN has_live_video INTEGER;
ALTER TABLE track_suggestions ADD COLUMN poll_correct INTEGER;
ALTER TABLE track_suggestions ADD COLUMN poll_not_correct INTEGER;
ALTER TABLE track_suggestions ADD COLUMN poll_unsure INTEGER;
ALTER TABLE track_suggestions ADD COLUMN labels_json TEXT;
ALTER TABLE track_suggestions ADD COLUMN google_search_url TEXT;

CREATE TABLE IF NOT EXISTS set_notices (
    set_id TEXT NOT NULL, row_index INTEGER NOT NULL,
    row_type TEXT, text TEXT, links_json TEXT, icons_json TEXT,
    parsed_json TEXT, parsed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (set_id, row_index)
);
CREATE TABLE IF NOT EXISTS set_slot_id_meta (
    set_id TEXT NOT NULL, row_index INTEGER NOT NULL, tlp_id INTEGER,
    is_id INTEGER DEFAULT 0, protected INTEGER DEFAULT 0, rbcst INTEGER DEFAULT 0,
    watchers INTEGER, presave_count INTEGER,
    parsed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (set_id, row_index)
);
-- track_id_links already exists (schema.sql:700).
```

- [ ] **Step 7: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/tokenizer/test_correction_schema.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add tokenizer/materialize.py web_crawler/database/schema.sql scripts/migrations/migrate_correction_layer.sql tests/tokenizer/test_correction_schema.py
git commit -m "feat(schema): correction/notice/id-links tables + track_suggestions payload cols"
```

---

### Task 2: Write the full suggestion payload in materialize

**Files:**
- Modify: `tokenizer/materialize.py:136-147` (`_flush_suggestions`) and `:363-382` (`sug_buf.append`)
- Test: `tests/tokenizer/test_suggestion_payload.py`

**Interfaces:**
- Consumes: `SuggestionRow` fields (`data_type`, `cue_seconds`, `play_cue_seconds`, `suggester_guest_id`, `suggester_kind`, `track_page_path`, `track_id_numeric`, `is_id_remix`, `has_apple`, `has_affiliate`, `has_live_video`, `poll_correct`, `poll_not_correct`, `poll_unsure`, `labels`, `google_search_url`).
- Produces: `track_suggestions` rows carrying the correction semantics.

- [ ] **Step 1: Write the failing test**

```python
# tests/tokenizer/test_suggestion_payload.py
import sqlite3
from bs4 import BeautifulSoup
from tokenizer.materialize import _MATERIALIZE_DDL, _flush_suggestions
from tokenizer.suggestion_tokenizer import parse_suggestion_row

# A type-5 "track wasn't played" suggestion row.
ROW = (
    '<div class="bItm con ntB sugTog" data-type="5" data-tlp="123" data-pos="7" '
    'data-guest="999">'
    "track wasn't played [24-11-09 10:01:29] MeCaddy [poll: 3 / 1 / 0 ]"
    '</div>'
)


def _buf_from(html, set_id="SET1"):
    sug = parse_suggestion_row(BeautifulSoup(html, "lxml").find("div"))
    # mirror the materialize.py append shape
    from tokenizer.materialize import _suggestion_tuple  # helper added in Step 3
    return _suggestion_tuple(sug, set_id)


def test_suggestion_type_and_poll_persist():
    conn = sqlite3.connect(":memory:")
    conn.executescript(_MATERIALIZE_DDL)
    _flush_suggestions(conn, [_buf_from(ROW)])
    row = conn.execute(
        "SELECT data_type, poll_correct, poll_not_correct, poll_unsure "
        "FROM track_suggestions"
    ).fetchone()
    assert row == (5, 3, 1, 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/tokenizer/test_suggestion_payload.py -v`
Expected: FAIL — `ImportError: cannot import name '_suggestion_tuple'`.

- [ ] **Step 3: Add a `_suggestion_tuple` helper and use it**

In `tokenizer/materialize.py`, add near `_flush_suggestions`:

```python
def _suggestion_tuple(sug, set_id: str) -> tuple:
    def b(x):
        return int(bool(x)) if x is not None else None
    labels_json = (
        json.dumps([[n, p] for n, p in sug.labels], ensure_ascii=False)
        if sug.labels else None
    )
    return (
        sug.sug_id, set_id, sug.tlp_id, sug.pos, sug.track_slug,
        sug.track_display, sug.artist_title, sug.suggester_user_id,
        sug.suggester_name, sug.suggestion_timestamp, b(sug.is_remix),
        b(sug.has_youtube), b(sug.has_soundcloud), b(sug.has_spotify),
        sug.data_type, sug.cue_seconds, sug.play_cue_seconds,
        sug.suggester_guest_id, sug.suggester_kind, sug.track_page_path,
        sug.track_id_numeric, b(sug.is_id_remix), b(sug.has_apple),
        b(sug.has_affiliate), b(sug.has_live_video),
        sug.poll_correct, sug.poll_not_correct, sug.poll_unsure,
        labels_json, sug.google_search_url,
    )
```

Replace the `sug_buf.append((...))` block (`materialize.py:363-382`) with:

```python
                    sug_buf.append(_suggestion_tuple(sug, row["set_id"]))
```

- [ ] **Step 4: Extend `_flush_suggestions` with a PRAGMA-guarded 30-column INSERT**

Replace `_flush_suggestions` (`materialize.py:136-147`) with:

```python
def _flush_suggestions(conn: sqlite3.Connection, buf: list[tuple]) -> None:
    if not buf:
        return
    cols = {r[1] for r in conn.execute("PRAGMA table_info(track_suggestions)")}
    if "data_type" in cols:
        conn.executemany(
            "INSERT OR REPLACE INTO track_suggestions ("
            "sug_id, set_id, tlp_id, pos, track_slug, track_display, artist_title, "
            "suggester_user_id, suggester_name, suggestion_timestamp, "
            "is_remix, has_youtube, has_soundcloud, has_spotify, "
            "data_type, cue_seconds, play_cue_seconds, suggester_guest_id, "
            "suggester_kind, track_page_path, track_id_numeric, is_id_remix, "
            "has_apple, has_affiliate, has_live_video, poll_correct, "
            "poll_not_correct, poll_unsure, labels_json, google_search_url"
            ") VALUES (" + ",".join("?" * 30) + ")",
            buf,
        )
    else:
        conn.executemany(
            "INSERT OR REPLACE INTO track_suggestions ("
            "sug_id, set_id, tlp_id, pos, track_slug, track_display, artist_title, "
            "suggester_user_id, suggester_name, suggestion_timestamp, "
            "is_remix, has_youtube, has_soundcloud, has_spotify"
            ") VALUES (" + ",".join("?" * 14) + ")",
            [row[:14] for row in buf],
        )
    conn.commit()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/tokenizer/test_suggestion_payload.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tokenizer/materialize.py tests/tokenizer/test_suggestion_payload.py
git commit -m "feat(materialize): persist suggestion type/cue/poll/labels payload"
```

---

## Phase 2 — Notice / section rows

### Task 3: Parse + persist `bItmH` rows to `set_notices`

**Files:**
- Modify: `tokenizer/materialize.py` — import `parse_bItmH_row`; add `_flush_notices`; replace the `bItmH` dispatch (`:385-386`); add `notice_buf` + tail flush
- Test: `tests/tokenizer/test_notices.py`

**Interfaces:**
- Consumes: `parse_bItmH_row(html) -> TextRowToken` (`.row_type`, `.text`, `.links`, `.icons`, `.parsed`).
- Produces: `set_notices` rows.

- [ ] **Step 1: Write the failing test**

```python
# tests/tokenizer/test_notices.py
import sqlite3
from tokenizer.materialize import _MATERIALIZE_DDL, _flush_notices
from tokenizer.text_tokenizer import parse_bItmH_row
from tokenizer.materialize import _notice_tuple

RECYCLE = (
    '<div class="bItmH"><i class="fa fa-recycle fa-24"></i> '
    'This tracklist contains identical tracklist(s) (parts)</div>'
)


def test_notice_row_type_persists():
    conn = sqlite3.connect(":memory:")
    conn.executescript(_MATERIALIZE_DDL)
    tok = parse_bItmH_row(RECYCLE)
    _flush_notices(conn, [_notice_tuple(tok, "SET1", 12)])
    row = conn.execute(
        "SELECT set_id, row_index, row_type FROM set_notices"
    ).fetchone()
    assert row == ("SET1", 12, "recycle_notice")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/tokenizer/test_notices.py -v`
Expected: FAIL — `ImportError: cannot import name '_flush_notices'`.

- [ ] **Step 3: Add `_notice_tuple` + `_flush_notices` and wire the dispatch**

Add near the top imports of `materialize.py`:

```python
from tokenizer.text_tokenizer import parse_bItmH_row
```

Add helpers near `_flush_suggestions`:

```python
def _notice_tuple(tok, set_id: str, row_index: int) -> tuple:
    return (
        set_id, row_index, tok.row_type, tok.text,
        json.dumps(tok.links, ensure_ascii=False) if tok.links else None,
        json.dumps(tok.icons, ensure_ascii=False) if tok.icons else None,
        json.dumps(tok.parsed, ensure_ascii=False) if tok.parsed else None,
    )


def _flush_notices(conn: sqlite3.Connection, buf: list[tuple]) -> None:
    if not buf:
        return
    conn.executemany(
        "INSERT OR REPLACE INTO set_notices ("
        "set_id, row_index, row_type, text, links_json, icons_json, parsed_json"
        ") VALUES (?,?,?,?,?,?,?)",
        buf,
    )
    conn.commit()
```

In `materialize()`, declare `notice_buf: list[tuple] = []` next to `sug_buf`, and replace the dispatch branch (`materialize.py:385-386`):

```python
                elif "bItmH" in outer_classes:
                    tok = parse_bItmH_row(raw)
                    notice_buf.append(_notice_tuple(tok, row["set_id"], int(row["row_index"])))
                    counts["text"] += 1
```

Add a flush guard next to the suggestion flush (after `materialize.py:394`):

```python
            if len(notice_buf) >= _BATCH_INSERT:
                _flush_notices(conn, notice_buf)
                notice_buf.clear()
```

And a tail flush next to `_flush_suggestions(conn, sug_buf)` (`materialize.py:413`):

```python
    _flush_notices(conn, notice_buf)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/tokenizer/test_notices.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tokenizer/materialize.py tests/tokenizer/test_notices.py
git commit -m "feat(materialize): parse+persist bItmH notices to set_notices"
```

---

## Phase 3 — ID-resolution workflow

### Task 4: Populate `track_id_links` + `set_slot_id_meta` for ID rows

**Files:**
- Modify: `tokenizer/materialize.py` — import `id_tokenizer.parse_track_row`; add `_flush_id_links`, `_flush_id_meta`; in the `tlpItem` branch, when the row is an ID row, run `id_tokenizer` on the already-parsed `outer` Tag and buffer links + meta
- Test: `tests/tokenizer/test_id_links.py`

**Interfaces:**
- Consumes: `id_tokenizer.parse_track_row(outer_div: Tag) -> IDTrack` (`.protected`, `.rbcst`, `.watchers`, `.spotify_presave_count`, `.linked_items[]` → `LinkedIDItem`).
- Produces: `track_id_links` + `set_slot_id_meta` rows.

- [ ] **Step 1: Write the failing test**

```python
# tests/tokenizer/test_id_links.py
import sqlite3
from tokenizer.materialize import _MATERIALIZE_DDL, _flush_id_meta, _id_meta_tuple


class _FakeID:
    protected = True
    rbcst = False
    watchers = 42
    spotify_presave_count = None


def test_id_meta_persists():
    conn = sqlite3.connect(":memory:")
    conn.executescript(_MATERIALIZE_DDL)
    _flush_id_meta(conn, [_id_meta_tuple(_FakeID(), "SET1", 3, tlp_id=555, is_id=True)])
    row = conn.execute(
        "SELECT is_id, protected, rbcst, watchers FROM set_slot_id_meta"
    ).fetchone()
    assert row == (1, 1, 0, 42)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/tokenizer/test_id_links.py -v`
Expected: FAIL — `ImportError: cannot import name '_flush_id_meta'`.

- [ ] **Step 3: Add helpers + flushes**

Add import near the other tokenizer imports:

```python
from tokenizer.id_tokenizer import parse_track_row as parse_id_row
```

Add helpers near `_flush_suggestions`:

```python
def _id_meta_tuple(idt, set_id: str, row_index: int, tlp_id, is_id: bool) -> tuple:
    def b(x):
        return int(bool(x))
    return (
        set_id, row_index, tlp_id, b(is_id), b(idt.protected), b(idt.rbcst),
        idt.watchers, idt.spotify_presave_count,
    )


def _flush_id_meta(conn: sqlite3.Connection, buf: list[tuple]) -> None:
    if not buf:
        return
    conn.executemany(
        "INSERT OR REPLACE INTO set_slot_id_meta ("
        "set_id, row_index, tlp_id, is_id, protected, rbcst, watchers, presave_count"
        ") VALUES (?,?,?,?,?,?,?,?)",
        buf,
    )
    conn.commit()


def _id_link_tuples(idt, set_id: str, tlp_id) -> list[tuple]:
    out = []
    for li in idt.linked_items:
        out.append((
            set_id, tlp_id, li.user_name, li.user_href,
            li.user_followers_text, li.linked_tracklist_href,
            li.linked_tracklist_text,
        ))
    return out


def _flush_id_links(conn: sqlite3.Connection, buf: list[tuple]) -> None:
    if not buf:
        return
    conn.executemany(
        "INSERT OR REPLACE INTO track_id_links ("
        "set_id, tlp_id, linker_user_name, linker_user_href, "
        "linker_user_followers, linked_tracklist_href, linked_tracklist_text"
        ") VALUES (?,?,?,?,?,?,?)",
        buf,
    )
    conn.commit()
```

- [ ] **Step 4: Wire the ID-row branch in the `tlpItem` path**

In `materialize()`, declare `id_meta_buf: list[tuple] = []` and `id_link_buf: list[tuple] = []` next to `slot_buf`. Inside the `if "tlpItem" in outer_classes:` block, after the slot append (`materialize.py:355`), add:

```python
                        is_id_row = tr.is_ided or (outer.get("data-isid") == "true")
                        if is_id_row:
                            idt = parse_id_row(outer)
                            id_meta_buf.append(
                                _id_meta_tuple(idt, sid, int(row["row_index"]),
                                               tr.data_id, is_id_row)
                            )
                            if idt.linked_items and tr.data_id is not None:
                                id_link_buf.extend(_id_link_tuples(idt, sid, tr.data_id))
```

Add flush guards next to the slot flush and tail flushes next to `_flush_slots(conn, slot_buf)` (`materialize.py:412`):

```python
    _flush_id_meta(conn, id_meta_buf)
    _flush_id_links(conn, id_link_buf)
```

(Batch-flush guards inside the loop mirror the `slot_buf` pattern at `materialize.py:357-359`.)

- [ ] **Step 5: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/tokenizer/test_id_links.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tokenizer/materialize.py tests/tokenizer/test_id_links.py
git commit -m "feat(materialize): populate track_id_links + set_slot_id_meta for ID rows"
```

---

### Task 5: Full-suite gate + corpus smoke

**Files:** none (verification only)

- [ ] **Step 1: Run the WS-C test subset**

Run: `venvs/audio/bin/python -m pytest tests/tokenizer -q`
Expected: PASS (all)

- [ ] **Step 2: Guardrails**

Run: `make check`
Expected: PASS

- [ ] **Step 3: Smoke — parse a real correction row from the corpus**

Run:
```bash
venvs/audio/bin/python - <<'PY'
import glob
from bs4 import BeautifulSoup
from tokenizer.suggestion_tokenizer import parse_suggestion_row
found = False
for f in sorted(glob.glob("data/html/*.html"))[:1500]:
    html = open(f, encoding="utf-8", errors="replace").read()
    if "sugTog" not in html:
        continue
    tl = BeautifulSoup(html, "lxml").find("div", id="tlTab")
    for r in tl.find_all("div", recursive=False):
        if "sugTog" in (r.get("class") or []) and r.get("data-type") == "14":
            s = parse_suggestion_row(r)
            print("type-14 cue correction: cue_seconds =", s.cue_seconds,
                  "poll =", (s.poll_correct, s.poll_not_correct, s.poll_unsure))
            found = True
            break
    if found:
        break
print("OK" if found else "no type-14 in sample (fine)")
PY
```
Expected: prints a parsed cue correction (or "OK"/none-in-sample).

---

## pi-storage rollout (run once, coordinated)

1. Back up canonical DB.
2. `make deploy`.
3. `ssh pi-storage 'sqlite3 /mnt/storage/data/db/music_database.db < scripts/migrations/migrate_correction_layer.sql'`.
4. Re-run `python -m tokenizer.materialize` (rebuilds suggestions with payload + notices + id links/meta).
5. Spot-check: `SELECT data_type, COUNT(*) FROM track_suggestions GROUP BY data_type;` (expect the 18-type distribution, incl. 5/14/1/17) and `SELECT row_type, COUNT(*) FROM set_notices GROUP BY row_type;`.

## Deferred (separate plans)

- **Consumers:** weight suggestions by poll (accept/review/abstain), apply type-14 cue corrections / type-5 "wasn't played" to the scorer + aligner input, use `track_id_links` for transitive ID resolution. WS-C only *lands the data*.
- **WS-B** (identity edges: `version_artist`, `byArtist`, `rework-of` source) and **WS-A** (mashup structure) — separate plans.
