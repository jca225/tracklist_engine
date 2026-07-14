# Appleseed "Any Song" Bridge + HPF Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** (A) High-pass the acapella in the renderer so every mashup sounds pro (bass-swap: one low end at a time); (B) let a friend add ANY song to Appleseed by typing a name or pasting a link — the last barrier to entry — without putting a downloader in the product repo.

**Architecture:** The product repo (`mashup_compiler`) stays downloader-free. The server records a **song request** in its SQLite; a **separate fulfiller process** living in the research repo (`tracklist_engine/lab/appleseed/appleseed_librarian.py`) polls that DB, fetches the best match with the existing ingest stack, drops a `.wav` into `server/library/`, and marks the request done/failed. The server's existing library scan + analyze pipeline turns that file into a ready, mashable song. Cross-repo coupling is one SQLite file path + one folder — no network, no downloader in the product.

**Tech Stack:** existing repo stacks. New: `scipy.signal` (already in the mashup_compiler venv) for the HPF; the fulfiller reuses `ingest.adapters.ytmusic_adapter.search` + `ingest.adapters.downloader.download_one` from tracklist_engine.

## Global Constraints

- Two repos. **$MC** = `/Users/johnnycabrahams/Desktop/mashup_compiler` (Tasks 1–3, 5a). **$TLE** = `/Users/johnnycabrahams/Desktop/tracklist_engine` (Task 4, 5b). Commit in the repo you're editing.
- **The product repo (`$MC`) must contain NO downloader / yt-dlp / spotdl / network-fetch code.** The fulfiller (which does the fetching) lives ONLY in `$TLE`. This is the load-bearing legal/architecture boundary — do not cross it.
- `engine/` in $MC is FROZEN.
- $MC offline suite stays green after each $MC task: `venv/bin/python -m pytest tests/ -m "not integration"` (currently 48 passed, 1 deselected).
- A global hook ruff-formats .py writes — fine.
- The fulfiller runs in $TLE's `venvs/audio` (has yt-dlp + ingest). The server runs in $MC's `venv`. They share only `$MC/server/state.db` (path) and `$MC/server/library/` (folder).
- Song-request states: `searching | done | failed`. When `done`, the wav is already in `library/` and the normal scan/analyze takes over — the request row is terminal, not a song.

---

### Task 1: HPF the acapella (bass-swap) in the renderer

**Files:**
- Modify: `$MC/compiler/render.py`
- Test: append to `$MC/tests/test_render.py`

**Interfaces:**
- Produces: a `_highpass(y, sr, hz)` helper and its application to `role == "acappella"` clips inside `_process`, BEFORE the fades. Constant `ACAPPELLA_HPF_HZ = 90.0`.

- [ ] **Step 1: Write the failing test (append to tests/test_render.py)**

```python
@needs_rubberband
def test_acappella_is_highpassed(tmp_path: Path) -> None:
    # a 50 Hz tone (sub-bass) — an acappella clip should lose most of it.
    sr = 44100
    t = np.arange(sr * 2) / sr
    sub = (0.5 * np.sin(2 * np.pi * 50 * t)).astype(np.float32)
    sf.write(tmp_path / "sub.wav", sub, sr)
    def _rms(y):
        return float(np.sqrt((y.astype(np.float64) ** 2).mean()))
    from compiler.render import _highpass
    filtered = _highpass(np.stack([sub, sub], axis=1), sr, 90.0)
    assert _rms(filtered) < 0.3 * _rms(sub)     # sub-bass strongly attenuated
    # and a 1 kHz tone should survive
    mid = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    kept = _highpass(np.stack([mid, mid], axis=1), sr, 90.0)
    assert _rms(kept) > 0.8 * _rms(mid)
```

- [ ] **Step 2: Run to verify FAIL** — `venv/bin/python -m pytest tests/test_render.py::test_acappella_is_highpassed -v` → ImportError (`_highpass`).

- [ ] **Step 3: Implement in `compiler/render.py`**

At the top imports add `from scipy.signal import butter, sosfilt` and beside `FADE_S` add `ACAPPELLA_HPF_HZ = 90.0`.

```python
def _highpass(y: np.ndarray, sr: int, hz: float) -> np.ndarray:
    """4th-order Butterworth high-pass. Bass-swap: strip the acapella's low
    end so only the instrumental owns the sub (DJ craft: one low end at a
    time). y is (n,) or (n, ch); filter each channel along axis 0."""
    sos = butter(4, hz, btype="highpass", fs=sr, output="sos")
    if y.ndim == 1:
        return sosfilt(sos, y).astype(y.dtype, copy=False)
    return np.stack([sosfilt(sos, y[:, c]) for c in range(y.shape[1])], axis=1).astype(y.dtype, copy=False)
```

In `_process`, after the stretch/pitch/gain but BEFORE the edge-fade block, add:

```python
    if clip.role == "acappella":
        y = _highpass(y, sr, ACAPPELLA_HPF_HZ)
```

(so the HPF runs on the final-rate audio, then the fades taper the filtered edges).

- [ ] **Step 4: Run to verify PASS** — the new test + full offline suite (49 passed, 1 deselected). If `sosfilt` dtype-casts to float64, the `.astype` guard keeps the pipeline's float consistency — verify `test_render_places_and_stretches` still passes.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: high-pass the acapella (~90Hz bass-swap) — one low end at a time (DJ craft)"`

---

### Task 2: Song-request store + API (server)

**Files:**
- Modify: `$MC/server/db.py` (new table + migration), `$MC/server/app.py` (endpoints)
- Test: `$MC/tests/test_requests.py`

**Interfaces:**
- Produces: table `song_requests(id, query, kind[name|url], status[searching|done|failed], error, created_by, created_at)`; `POST /api/request` (form `query`) → inserts a `searching` request, returns `{"id": N}`; `GET /api/requests` → JSON list of non-`done` requests (the UI polls this); the library page passes pending requests to the template.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_requests.py
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from server import db as sdb

    monkeypatch.setattr(sdb, "DB_PATH", tmp_path / "state.db")
    sdb.init()
    from server import app as sapp, jobs

    monkeypatch.setattr(jobs, "enqueue", lambda kind, rid: None)
    monkeypatch.setattr(sapp, "_scan", lambda: None)
    from fastapi.testclient import TestClient

    return TestClient(sapp.app)


def test_request_by_name_inserts_searching(client) -> None:
    r = client.post("/api/request", data={"query": "freed from desire"})
    assert r.status_code == 200
    rid = r.json()["id"]
    from server import db as sdb

    row = sdb.connect().execute("SELECT * FROM song_requests WHERE id=?", (rid,)).fetchone()
    assert row["status"] == "searching" and row["kind"] == "name"
    assert row["query"] == "freed from desire"


def test_request_detects_url_kind(client) -> None:
    r = client.post("/api/request", data={"query": "https://youtu.be/abc123"})
    rid = r.json()["id"]
    from server import db as sdb

    row = sdb.connect().execute("SELECT * FROM song_requests WHERE id=?", (rid,)).fetchone()
    assert row["kind"] == "url"


def test_request_rejects_blank(client) -> None:
    assert client.post("/api/request", data={"query": "   "}).status_code == 400


def test_requests_endpoint_lists_pending(client) -> None:
    client.post("/api/request", data={"query": "song one"})
    listed = client.get("/api/requests").json()["requests"]
    assert len(listed) == 1 and listed[0]["status"] == "searching"
```

- [ ] **Step 2: Run to verify FAIL** — no `song_requests` table / no `/api/request`.

- [ ] **Step 3: Implement**

`server/db.py` — add to `_SCHEMA`:

```sql
CREATE TABLE IF NOT EXISTS song_requests(
  id INTEGER PRIMARY KEY,
  query TEXT NOT NULL,
  kind TEXT NOT NULL DEFAULT 'name',
  status TEXT NOT NULL DEFAULT 'searching',
  error TEXT,
  created_by TEXT NOT NULL DEFAULT '',
  created_at REAL NOT NULL
);
```

(the `CREATE TABLE IF NOT EXISTS` in `executescript` covers both fresh and existing DBs — no ALTER needed since it's a new table.)

`server/app.py` — add:

```python
import re as _re  # if not already imported as re; reuse existing re


def _request_kind(q: str) -> str:
    return "url" if _re.match(r"https?://", q.strip(), _re.I) else "name"


@app.post("/api/request")
def create_request(request: Request, query: str = Form(...)):
    q = query.strip()
    if not q:
        return JSONResponse({"error": "type a song name or paste a link"}, status_code=400)
    con = db.connect()
    cur = con.execute(
        "INSERT INTO song_requests(query, kind, status, created_by, created_at)"
        " VALUES(?,?,?,?,?)",
        (q[:200], _request_kind(q), "searching", _name(request) or "someone", db.now()),
    )
    con.commit()
    rid = cur.lastrowid
    con.close()
    return {"id": rid}


@app.get("/api/requests")
def list_requests():
    con = db.connect()
    rows = con.execute(
        "SELECT id, query, kind, status, error FROM song_requests"
        " WHERE status != 'done' ORDER BY id DESC"
    ).fetchall()
    con.close()
    return {"requests": [dict(r) for r in rows]}
```

(the existing module already `import re` — reuse it; the `_re` alias above is only illustrative, use the module's existing `re`.)

- [ ] **Step 4: Run to verify PASS** (4 tests) + full offline suite green.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(server): song-request store + /api/request (name or url)"`

---

### Task 3: Library UI — add any song by name or link

**Files:**
- Modify: `$MC/server/app.py` (thread pending requests into the library view), `$MC/server/templates/library.html`, `$MC/server/static/app.js`, `$MC/server/static/style.css`
- (No new tests — UI; covered by Task 2's API tests + manual check.)

**Interfaces:**
- Consumes: `GET /api/requests`, `POST /api/request`.
- Produces: a search input at the top of the library ("Add any song — name or paste a link"), a pending-requests list that polls `/api/requests` and reloads when one clears, mono-styled to match Appleseed.

- [ ] **Step 1: Pass pending requests to the library template** — in `server/app.py`'s `library()` handler, before rendering, add:

```python
    reqs = con.execute(
        "SELECT id, query, status, error FROM song_requests WHERE status != 'done' ORDER BY id DESC"
    ).fetchall()
```

and include `"requests": reqs` in the TemplateResponse context (keep `con` open until after this query; close after).

- [ ] **Step 2: Add the search box + request list to `library.html`** (above the existing upload form):

```html
<form class="addsong" method="post" action="/api/request" id="add-song">
  <input name="query" id="add-query" placeholder="Add any song — name or paste a link"
         autocomplete="off" required>
  <button class="pill small">Add</button>
</form>
<ul class="reqs" id="reqs">
{% for r in requests %}
  <li class="req" data-req="{{ r['id'] }}" data-status="{{ r['status'] }}">
    <span class="req-q">{{ r['query'] }}</span>
    {% if r['status'] == 'failed' %}
      <span class="chip bad" title="{{ r['error'] }}">not found</span>
    {% else %}
      <span class="chip dimchip">searching<span class="dots" aria-hidden="true"></span></span>
    {% endif %}
  </li>
{% endfor %}
</ul>
```

- [ ] **Step 3: Wire the JS** — append to `server/static/app.js`:

```js
// library: submit add-song via fetch (no full nav), then poll requests
const addForm = document.querySelector("#add-song");
if (addForm) {
  addForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const input = document.querySelector("#add-query");
    const q = input.value.trim();
    if (!q) return;
    const btn = addForm.querySelector("button");
    btn.disabled = true;
    await fetch("/api/request", { method: "POST", body: new URLSearchParams({ query: q }) });
    input.value = "";
    btn.disabled = false;
    location.reload();
  });
  // poll while any request is unresolved OR any song is still analyzing
  const poll = setInterval(async () => {
    try {
      const s = await (await fetch("/api/requests")).json();
      const open = s.requests.filter((r) => r.status === "searching").length;
      const shown = document.querySelectorAll('.req[data-status="searching"]').length;
      if (open !== shown) { clearInterval(poll); location.reload(); }
    } catch { /* keep polling */ }
  }, 4000);
}
```

- [ ] **Step 4: Style (append to `style.css`, mono Appleseed idiom)**

```css
.addsong { display: flex; gap: 8px; margin-bottom: 12px; }
.addsong input {
  flex: 1; background: var(--panel2); border: 1px solid var(--line);
  border-radius: 7px; color: var(--paper); font: 500 15px var(--body); padding: 11px 12px;
}
.addsong input:focus-visible { border-color: var(--accent); outline: none; }
.reqs { list-style: none; margin: 0 0 16px; padding: 0; display: grid; gap: 6px; }
.req { display: flex; justify-content: space-between; align-items: center; gap: 10px;
  padding: 10px 12px; background: var(--panel); border: 1px solid var(--line); border-radius: 7px; }
.req-q { font-size: 14px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
```

- [ ] **Step 5: Manual render check** — start on a scratch port, `curl -s localhost:8533/library | grep -q "Add any song" && echo OK`; kill scratch server. Full offline suite still green (no test changes → 52 passed from Task 2's additions? confirm the number matches prior + 0).

- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat(server): library 'add any song' search box + pending-request list"`

---

### Task 4: The librarian fulfiller (research repo)

**Files:**
- Create: `$TLE/lab/appleseed/appleseed_librarian.py`
- Test: `$TLE/tests/test_appleseed_librarian.py` (unit-test the pure pieces with the network mocked)

**Interfaces:**
- Consumes (READ their real signatures first — they are the contract): `ingest.adapters.ytmusic_adapter.search(...)` (returns validated `YTMSearchHit`s with `.url`/`.title`/`.duration` — inspect the dataclass at ingest/adapters/ytmusic_adapter.py:59) and `ingest.adapters.downloader.download_one(...)` (ingest/adapters/downloader.py:85 — inspect its args: url, out dir/template, `DownloadConfig`).
- Produces: a poll loop `run(db_path, library_dir, once=False)` that: reads `song_requests WHERE status='searching'` from the $MC state.db; for each, resolves audio (kind='url' → `download_one`; kind='name' → `search` then `download_one` on the best validated hit); transcodes to 44.1k wav into `library_dir/`; marks the request `done` (wav present → server scan takes over) or `failed` with a reason. Pure helpers `_best_hit(hits)` and `_safe_stem(query)` are unit-tested.

- [ ] **Step 1: Read the two adapter files** (`ingest/adapters/ytmusic_adapter.py` around `search`/`YTMSearchHit`/`_ytdlp_download`, and `ingest/adapters/downloader.py` around `download_one`/`DownloadConfig`) and note the exact call signatures. Record them in the report — the rest of this task depends on them.

- [ ] **Step 2: Write the failing unit tests** (`tests/test_appleseed_librarian.py`) — test the pure logic without network:

```python
from __future__ import annotations

import sqlite3
from pathlib import Path

from lab.appleseed.appleseed_librarian import _best_hit, _safe_stem, _claim_one


def test_safe_stem_sanitizes() -> None:
    assert _safe_stem("AC/DC - Back in Black!! ") == "AC_DC - Back in Black"


def test_best_hit_prefers_validated_then_shortest_reasonable(monkeypatch) -> None:
    # _best_hit takes a list of hit-like objects (title, url, duration_s) and
    # returns the one to download, or None. A too-short (<60s) preview loses.
    class H:
        def __init__(self, title, url, dur):
            self.title, self.url, self.duration_s = title, url, dur
    hits = [H("Song (Preview)", "u1", 30), H("Song", "u2", 210), H("Song (Live)", "u3", 240)]
    assert _best_hit(hits).url == "u2"          # full studio-length, not preview, not live
    assert _best_hit([H("x", "u", 20)]) is None  # only a preview → no confident pick


def test_claim_one_is_atomic(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE song_requests(id INTEGER PRIMARY KEY, query TEXT, kind TEXT,"
                " status TEXT, error TEXT, created_by TEXT, created_at REAL)")
    con.execute("INSERT INTO song_requests(id,query,kind,status,created_at) VALUES"
                "(1,'a','name','searching',0),(2,'b','name','done',0)")
    con.commit(); con.close()
    first = _claim_one(db)
    assert first is not None and first["id"] == 1
    # once claimed it flips to 'searching'->in-progress marker so a 2nd poller skips it
    second = _claim_one(db)
    assert second is None
```

(`_claim_one` marks the row it returns with an in-progress sentinel — implement it so two fulfillers don't double-fetch; the sentinel can be setting `status='searching'`→ a distinct value like `'fetching'`, but keep the server's `!= 'done'` filter showing it. Use `'searching'`→update to `'fetching'` and treat `'fetching'` as still-pending in the server list query if needed; simplest: add `'fetching'` to the server's non-done statuses — but to avoid a Task-2/3 change, `_claim_one` can instead select+update to `'fetching'` and the server already shows non-`done`, so `'fetching'` still displays as pending. Confirm the server list query is `status != 'done'` — it is.)

- [ ] **Step 3: Run to verify FAIL** (ModuleNotFoundError).

- [ ] **Step 4: Implement `lab/appleseed/appleseed_librarian.py`** — CLI + loop. Structure (fill the ingest calls from Step 1's real signatures):

```python
"""Appleseed librarian — fulfills song_requests from Appleseed's SQLite by
fetching audio with the ingest stack into the app's library folder. Runs in
tracklist_engine (has yt-dlp); the product repo stays downloader-free.

    venvs/audio/bin/python -m lab.appleseed.appleseed_librarian \
        --db ~/Desktop/mashup_compiler/server/state.db \
        --library ~/Desktop/mashup_compiler/server/library [--once]
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

MIN_FULL_S = 60  # shorter than this = a preview, not the song


def _safe_stem(q: str) -> str:
    return re.sub(r"[^\w\s\-\(\)&,']", "_", q).strip()[:80]


def _best_hit(hits):
    ok = [h for h in hits if getattr(h, "duration_s", 0) and h.duration_s >= MIN_FULL_S
          and "live" not in h.title.lower() and "preview" not in h.title.lower()]
    if not ok:
        return None
    return min(ok, key=lambda h: h.duration_s)   # shortest full-length = likely the studio cut


def _claim_one(db_path: Path):
    con = sqlite3.connect(db_path, timeout=10)
    con.row_factory = sqlite3.Row
    try:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute("SELECT * FROM song_requests WHERE status='searching' ORDER BY id LIMIT 1").fetchone()
        if row is None:
            con.rollback(); return None
        con.execute("UPDATE song_requests SET status='fetching' WHERE id=?", (row["id"],))
        con.commit()
        return row
    finally:
        con.close()


def _mark(db_path: Path, rid: int, status: str, error: str | None = None) -> None:
    con = sqlite3.connect(db_path, timeout=10)
    con.execute("UPDATE song_requests SET status=?, error=? WHERE id=?", (status, error, rid))
    con.commit(); con.close()


def _fetch_to_library(row, library: Path) -> Path:
    """Resolve the request to a downloaded source file, transcode to 44.1k wav
    in `library`, return the wav path. Raises on failure."""
    from ingest.adapters import downloader, ytmusic_adapter  # noqa: F401

    library.mkdir(parents=True, exist_ok=True)
    # kind == 'url': download_one(url, ...) directly.
    # kind == 'name': ytmusic_adapter.search(query, ...) -> hits -> _best_hit -> download_one.
    # (Fill args from the real signatures read in Step 1; download into a temp
    # dir, then: subprocess ffmpeg -i <src> -ar 44100 <library>/<safe_stem>.wav)
    raise NotImplementedError  # replace with the wired ingest calls

    # after producing `wav`:
    # return wav


def run(db_path: Path, library: Path, once: bool = False) -> int:
    while True:
        row = _claim_one(db_path)
        if row is None:
            if once:
                return 0
            time.sleep(5)
            continue
        try:
            _fetch_to_library(row, library)
            _mark(db_path, row["id"], "done")
            print(f"fulfilled #{row['id']}: {row['query']}")
        except Exception as e:  # noqa: BLE001 — surface any fetch failure to the UI
            _mark(db_path, row["id"], "failed", str(e)[:300])
            print(f"failed #{row['id']}: {e}", file=sys.stderr)
        if once:
            return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--once", action="store_true")
    a = p.parse_args(argv)
    return run(a.db, a.library, once=a.once)


if __name__ == "__main__":
    sys.exit(main())
```

Wire `_fetch_to_library`'s two branches using the real ingest signatures from Step 1. Keep the ffmpeg transcode (kills the m4a/libsndfile problem the same way the server's upload path does). If `_best_hit` returns None (name search found only previews/live), raise a clear "no full-length match found" so the UI shows it.

- [ ] **Step 5: Run unit tests to verify PASS** (`venvs/audio/bin/python -m pytest tests/test_appleseed_librarian.py -v`). The network path (`_fetch_to_library`) is NOT unit-tested here (it's integration) — it gets the manual end-to-end check in Task 5.

- [ ] **Step 6: Guardrails + commit** — run `make check` if fast; `git add -A && git commit -m "feat: appleseed librarian — fulfill song requests via ingest into the app library"`

---

### Task 5: End-to-end wiring, docs, and the live demo

**Files:**
- Modify: `$MC/README.md`, `$MC/BACKLOG.md`; create `$TLE/lab/appleseed/appleseed_librarian.md` (run instructions)

- [ ] **Step 1: $MC README** — document: HPF/bass-swap now applied to acapellas; "Add any song" needs the librarian running (`$TLE` fulfiller command), and that the downloader deliberately lives outside the product repo. Commit in $MC.
- [ ] **Step 2: $TLE run doc** (`lab/appleseed/appleseed_librarian.md`) — the exact two-process runbook: server (`$MC/venv/bin/uvicorn server.app:app --host 0.0.0.0 --port 8500`) + librarian (`$TLE/venvs/audio/bin/python -m lab.appleseed.appleseed_librarian --db … --library …`), the request lifecycle (searching→fetching→wav lands→scan→analyzing→ready), and yt-dlp cookie/bot-check caveat (link to feedback_ytdlp_bot_detection_recipe). Commit in $TLE.
- [ ] **Step 3: Manual end-to-end (John, real network)** — start server + librarian; in the library UI type a song name (e.g. one of the five-friend ideas) → watch it go searching → ready; then mash it with an existing song and confirm the HPF makes the acapella sit cleaner over the bed. Record the outcome in the $TLE doc. (Not a blocking automated gate — it's the real acceptance test.)
- [ ] **Step 4: BACKLOG** ($MC) — mark HPF + any-song DONE; add: search-match confirmation UI (let the user pick among hits when the name is ambiguous), and the streaming-partner catalog as the eventual replacement for the local fulfiller. Commit in $MC.

---

## Deferred (explicitly NOT in this plan)
- Search disambiguation UI (auto-picks best hit for now; retry or paste-a-link if wrong).
- Full bass-swap sidechain (v1 = static HPF only; per-band ducking is a later NEEDS-NEW-SIGNAL item).
- Public catalog / streaming-partner integration (the fulfiller is the tailnet-era stand-in).
- The learned-space info-dynamics critic (deferred after the chroma P0 NULL).

## Self-review notes
- Boundary held: all fetch/download code is in Task 4 ($TLE) only; $MC Tasks 1–3 add a request *record* and UI, never a downloader. ✔
- HPF ordering: runs on final-rate audio before fades (Task 1 Step 3), so fades taper filtered edges. ✔
- Cross-process safety: `_claim_one` uses `BEGIN IMMEDIATE` + a `fetching` sentinel so a second fulfiller (or a double-poll) can't double-download. ✔
- Type/flow consistency: request `kind` set by `_request_kind` (Task 2) matches the fulfiller's branch on `row["kind"]` (Task 4). Request statuses `searching|fetching|done|failed`; server lists `!= 'done'`; `fetching` still shows as pending. ✔
- Known soft spot: Task 4's `_fetch_to_library` can't be fully code-complete until the implementer reads the two real ingest signatures (Step 1) — flagged, with the contract and the NotImplementedError placeholder to replace. This is the one task requiring live discovery.
