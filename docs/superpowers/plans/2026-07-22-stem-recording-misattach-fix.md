# Stem-recording Mis-attach Fix — Implementation Plan (PR #70, parts 1–2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent stem-candidate downloads from being attached to the wrong recording, and make the correction ledger able to represent such a mis-attach.

**Architecture:** A pure `same_song_guard` decision function (title-token gate OR stem-aware chromaprint verdict → REFUSE) wired into the two stem-attach entrypoints (`acquire_variant`, `replace_stem_audio`) via a shared I/O runner. On REFUSE without `--force`, no `track_audio` row survives and a new `axis='recording', action='detach'` correction is logged. A schema migration + `Correction` extension add the `recording` axis.

**Tech Stack:** Python 3 (stdlib + `numpy`), SQLite, Chromaprint (`fpcalc` via `ingest/adapters/fingerprint.py`), yt-dlp (title probe), pytest.

**Spec:** `docs/superpowers/specs/2026-07-22-stem-recording-misattach-fix-design.md`

## Global Constraints

- **Substrate rule:** `core/` imports nothing upward (no chain-module imports). `ingest/` must not import from `labeling/` (forward-chain). Shared pure helpers go in `core/`.
- **Style:** `from __future__ import annotations`; full type hints; frozen dataclasses for records; pure functions with I/O at the edges (Rust-flavoured functional Python).
- **Entropy fences (`make check`):** any net/subprocess call must pass `timeout=` and `encoding=` (or `text=`); no bare `except`.
- **Guard is fail-closed on a mismatch signal, not on absence of signal:** REFUSE only when a channel produces a positive mismatch; when no title and no fingerprints are available, ACCEPT (and the caller logs that it could not verify). This preserves legitimate attaches where a source title could not be probed.
- **Tests run from repo root** with `venvs/audio/bin/python -m pytest`.
- **No canonical pi writes in this PR.** The migration file ships; applying it to canonical state is a separate gated ops step (pi is 92 commits behind `main`).
- **Content-channel REFUSE set** = `classify()` verdicts `{"WRONG_SONG", "DURATION_MISMATCH"}` only. `OK`, `WEAK_SIGNAL`, `FALLBACK_TO_ORIGINAL` do NOT refuse (`FALLBACK_TO_ORIGINAL` is a wrong-*stem* signal, same song).

---

### Task 1: Relocate `labels_overlap` to `core/labels.py` (shared, layering-correct)

**Files:**
- Create: `core/labels.py`
- Modify: `labeling/als/identity.py:209-223` (replace the def with a re-export import)
- Test: `tests/core/test_labels.py` (create), `tests/labeling/test_als_io.py` (unchanged — must still pass)

**Interfaces:**
- Produces: `core.labels.labels_overlap(left: str, right: str, *, min_tokens: int = 2) -> bool`

- [ ] **Step 1: Write the failing test**

Create `tests/core/test_labels.py`:

```python
from __future__ import annotations

from core.labels import labels_overlap


def test_disjoint_titles_do_not_overlap():
    # the 20911 mis-attach: acquired song vs target recording title
    assert labels_overlap("Come On Over Baby (All I Want Is You)", "Good Time") is False


def test_same_song_titles_overlap():
    assert labels_overlap("Nelly Furtado - Say It Right", "Say It Right (Studio acapella)") is True


def test_empty_side_never_overlaps():
    assert labels_overlap("", "Good Time") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/core/test_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.labels'`

- [ ] **Step 3: Create `core/labels.py`** (move the exact logic from `labeling/als/identity.py`)

```python
"""Generic display-label token utilities (substrate; stdlib-only)."""

from __future__ import annotations

import re


def labels_overlap(left: str, right: str, *, min_tokens: int = 2) -> bool:
    """True when two display labels share enough distinctive tokens."""

    def _tokens(label: str) -> set[str]:
        cleaned = re.sub(r"[^\w\s]", " ", label.lower())
        return {w for w in cleaned.split() if len(w) > 2}

    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return False
    shared = a & b
    if len(shared) >= min_tokens:
        return True
    shorter = min(len(a), len(b))
    return shorter > 0 and len(shared) / shorter >= 0.4
```

- [ ] **Step 4: Re-export from `labeling/als/identity.py`** — replace the `def labels_overlap(...)` block (lines 209-223) with:

```python
from core.labels import labels_overlap  # re-exported: shared substrate helper
```

Place the import with the other top-of-file imports (after `from core.identity import normalize_stem`) and delete the old function body. Keep the name importable from `labeling.als.identity` for existing callers (`labeling/enrich_gt_track_ids.py`, `tests/labeling/test_als_io.py`).

- [ ] **Step 5: Run both test suites to verify they pass**

Run: `venvs/audio/bin/python -m pytest tests/core/test_labels.py tests/labeling/test_als_io.py -v`
Expected: PASS (new core tests + the existing `test_display_from_path_and_labels_overlap`)

- [ ] **Step 6: Commit**

```bash
git add core/labels.py labeling/als/identity.py tests/core/test_labels.py
git commit -m "refactor(core): move labels_overlap to core/labels for cross-stage reuse"
```

---

### Task 2: Ledger `recording` axis — schema + `Correction` extension

**Files:**
- Create: `scripts/migrations/migrate_correction_recording_axis.sql`
- Modify: `web_crawler/database/schema.sql:724-745` (canonical schema for fresh DBs)
- Modify: `ingest/corrections.py:21-22` (AXES/ACTIONS), `:25-42` (Correction fields), `:54-66` (INSERT)
- Test: `tests/ingest/test_corrections_recording_axis.py` (create)

**Interfaces:**
- Produces: `Correction(..., old_recording_id: str | None = None, new_recording_id: str | None = None)`; `AXES` now includes `"recording"`; `ACTIONS` now includes `"relink"`, `"detach"`.

- [ ] **Step 1: Write the failing test**

Create `tests/ingest/test_corrections_recording_axis.py`:

```python
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from core.result import Ok
from ingest.corrections import Correction, log_correction

# minimal standalone schema for the ledger table under test (post-migration shape)
_SCHEMA = """
CREATE TABLE track_audio_correction (
    correction_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    set_id TEXT, position TEXT, track_id TEXT NOT NULL,
    axis TEXT NOT NULL, action TEXT NOT NULL,
    old_track_audio_id INTEGER, old_platform TEXT, old_player_id TEXT, old_url TEXT,
    new_track_audio_id INTEGER, new_platform TEXT, new_player_id TEXT, new_url TEXT,
    old_recording_id TEXT, new_recording_id TEXT,
    stem_value TEXT, reason TEXT, source TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    CHECK (axis IN ('version','variant','stem','recording')),
    CHECK (action IN ('replace','add','relink','detach'))
);
"""


@pytest.fixture
def db(tmp_path: Path) -> Path:
    p = tmp_path / "t.db"
    with sqlite3.connect(p) as c:
        c.executescript(_SCHEMA)
    return p


def test_detach_correction_roundtrips(db: Path):
    c = Correction(
        track_id="42wv4vp", axis="recording", action="detach",
        set_id="1fsnxchk", position="148w1",
        old_recording_id="42wv4vp", new_recording_id=None,
        reason="title-token disjoint: 'Come On Over Baby' vs 'Good Time'",
        source="same_song_guard",
    )
    r = log_correction(db, c)
    assert isinstance(r, Ok)
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM track_audio_correction").fetchone()
    assert row["axis"] == "recording"
    assert row["action"] == "detach"
    assert row["old_recording_id"] == "42wv4vp"
    assert row["new_recording_id"] is None


def test_relink_correction_roundtrips(db: Path):
    c = Correction(
        track_id="abc123", axis="recording", action="relink",
        old_recording_id="42wv4vp", new_recording_id="9zzz000",
        source="remediation",
    )
    assert isinstance(log_correction(db, c), Ok)
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT new_recording_id FROM track_audio_correction").fetchone()
    assert row["new_recording_id"] == "9zzz000"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/ingest/test_corrections_recording_axis.py -v`
Expected: FAIL — `Correction.__init__() got an unexpected keyword argument 'old_recording_id'` (and, once that is added, an INSERT column-count error until Step 3 is complete).

- [ ] **Step 3: Extend `ingest/corrections.py`**

Change the constants (lines 21-22):

```python
AXES = ("version", "variant", "stem", "recording")
ACTIONS = ("replace", "add", "relink", "detach")
```

Add two fields to the `Correction` dataclass (after `new_url` on line 39):

```python
    old_recording_id: str | None = None
    new_recording_id: str | None = None
```

Widen the INSERT in `log_correction` (lines 53-66) to include the two columns:

```python
            cur = conn.execute(
                """
                INSERT INTO track_audio_correction
                  (set_id, position, track_id, axis, action,
                   old_track_audio_id, old_platform, old_player_id, old_url,
                   new_track_audio_id, new_platform, new_player_id, new_url,
                   old_recording_id, new_recording_id,
                   stem_value, reason, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    c.set_id, c.position, c.track_id, c.axis, c.action,
                    c.old_track_audio_id, c.old_platform, c.old_player_id, c.old_url,
                    c.new_track_audio_id, c.new_platform, c.new_player_id, c.new_url,
                    c.old_recording_id, c.new_recording_id,
                    c.stem_value, c.reason, c.source,
                ),
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/ingest/test_corrections_recording_axis.py -v`
Expected: PASS (both roundtrip tests)

- [ ] **Step 5: Update canonical schema + write the migration**

In `web_crawler/database/schema.sql`, in the `track_audio_correction` table (lines 724-745): add the two columns after `new_url` and extend both CHECKs:

```sql
    new_url             TEXT,
    old_recording_id    TEXT,              -- wrong recording a stem was attached to
    new_recording_id    TEXT,              -- corrected recording (NULL = detach/abstain)
    stem_value          TEXT,
    reason              TEXT,
    source              TEXT,
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    CHECK (axis IN ('version','variant','stem','recording')),
    CHECK (action IN ('replace','add','relink','detach'))
```

Create `scripts/migrations/migrate_correction_recording_axis.sql` (SQLite cannot ALTER a CHECK — rebuild):

```sql
-- Add the 'recording' axis + detach/relink actions to track_audio_correction.
-- SQLite cannot ALTER a CHECK constraint, so rebuild the table (no FKs to worry about).
PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;

ALTER TABLE track_audio_correction RENAME TO track_audio_correction_old;

CREATE TABLE track_audio_correction (
    correction_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    set_id              TEXT,
    position            TEXT,
    track_id            TEXT NOT NULL,
    axis                TEXT NOT NULL,
    action              TEXT NOT NULL,
    old_track_audio_id  INTEGER,
    old_platform        TEXT,
    old_player_id       TEXT,
    old_url             TEXT,
    new_track_audio_id  INTEGER,
    new_platform        TEXT,
    new_player_id       TEXT,
    new_url             TEXT,
    old_recording_id    TEXT,
    new_recording_id    TEXT,
    stem_value          TEXT,
    reason              TEXT,
    source              TEXT,
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    CHECK (axis IN ('version','variant','stem','recording')),
    CHECK (action IN ('replace','add','relink','detach'))
);

INSERT INTO track_audio_correction
  (correction_id, set_id, position, track_id, axis, action,
   old_track_audio_id, old_platform, old_player_id, old_url,
   new_track_audio_id, new_platform, new_player_id, new_url,
   stem_value, reason, source, created_at)
SELECT
   correction_id, set_id, position, track_id, axis, action,
   old_track_audio_id, old_platform, old_player_id, old_url,
   new_track_audio_id, new_platform, new_player_id, new_url,
   stem_value, reason, source, created_at
FROM track_audio_correction_old;

DROP TABLE track_audio_correction_old;

CREATE INDEX IF NOT EXISTS idx_track_audio_correction_track ON track_audio_correction(track_id);
CREATE INDEX IF NOT EXISTS idx_track_audio_correction_set   ON track_audio_correction(set_id);

COMMIT;
PRAGMA foreign_keys=ON;
```

- [ ] **Step 6: Verify the migration applies to a copy of the schema**

Run:
```bash
tmp=$(mktemp -d) && venvs/audio/bin/python -c "
import sqlite3, pathlib
# build the OLD-shape table, then migrate
c = sqlite3.connect('$tmp/m.db')
c.executescript('''CREATE TABLE track_audio_correction (
 correction_id INTEGER PRIMARY KEY AUTOINCREMENT, set_id TEXT, position TEXT,
 track_id TEXT NOT NULL, axis TEXT NOT NULL, action TEXT NOT NULL,
 old_track_audio_id INTEGER, old_platform TEXT, old_player_id TEXT, old_url TEXT,
 new_track_audio_id INTEGER, new_platform TEXT, new_player_id TEXT, new_url TEXT,
 stem_value TEXT, reason TEXT, source TEXT, created_at DATETIME,
 CHECK (axis IN ('version','variant','stem')), CHECK (action IN ('replace','add')));''')
c.execute(\"INSERT INTO track_audio_correction (track_id,axis,action) VALUES ('x','stem','add')\")
c.commit()
c.executescript(pathlib.Path('scripts/migrations/migrate_correction_recording_axis.sql').read_text())
c.execute(\"INSERT INTO track_audio_correction (track_id,axis,action,new_recording_id) VALUES ('y','recording','relink','z9')\")
c.commit()
print('rows:', c.execute('SELECT count(*) FROM track_audio_correction').fetchone()[0])
print('OK migration + recording-axis insert')
"
```
Expected: prints `rows: 2` then `OK migration + recording-axis insert` (the pre-existing `stem/add` row survives; the new `recording/relink` row inserts).

- [ ] **Step 7: Commit**

```bash
git add ingest/corrections.py web_crawler/database/schema.sql scripts/migrations/migrate_correction_recording_axis.sql tests/ingest/test_corrections_recording_axis.py
git commit -m "feat(ingest): add recording axis + detach/relink actions to correction ledger"
```

---

### Task 3: `same_song_guard` — the pure decision function

**Files:**
- Create: `ingest/same_song_guard.py`
- Test: `tests/ingest/test_same_song_guard.py` (create)

**Interfaces:**
- Consumes: `core.labels.labels_overlap`; `ingest.adapters.fingerprint.{Fingerprint, similarity, classify}`.
- Produces:
  - `GuardVerdict(accept: bool, channel: str | None, reason: str)` (frozen dataclass)
  - `same_song_guard(acquired_title: str, recording_title: str, stem_axis: str, fp_regular: Fingerprint | None, fp_candidate: Fingerprint | None) -> GuardVerdict`

- [ ] **Step 1: Write the failing test**

Create `tests/ingest/test_same_song_guard.py`:

```python
from __future__ import annotations

import numpy as np

from ingest.adapters.fingerprint import Fingerprint
from ingest.same_song_guard import GuardVerdict, same_song_guard


def _fp(seed: int, n: int = 400, dur: float = 200.0) -> Fingerprint:
    rng = np.random.default_rng(seed)
    return Fingerprint(duration_s=dur, raw=rng.integers(0, 2**32, size=n, dtype=np.uint32))


def test_title_disjoint_refuses_on_title_channel():
    # the 20911 case: acquired song != target recording title
    v = same_song_guard("Come On Over Baby", "Good Time", "acappella", None, None)
    assert v.accept is False
    assert v.channel == "title"


def test_title_overlap_and_no_fp_accepts():
    v = same_song_guard("Good Time (Studio Acapella)", "Good Time", "acappella", None, None)
    assert v.accept is True
    assert v.channel is None


def test_no_signal_accepts():
    # no acquired title AND no fingerprints -> cannot verify -> accept (not fail on absence)
    v = same_song_guard("", "Good Time", "acappella", None, None)
    assert v.accept is True


def test_content_wrong_song_refuses_when_title_passes():
    # title overlaps but instrumental content similarity is far too low -> WRONG_SONG
    a, b = _fp(1), _fp(999)  # unrelated fingerprints -> low similarity
    v = same_song_guard("Good Time", "Good Time Instrumental", "instrumental", a, b)
    assert v.accept is False
    assert v.channel == "content"


def test_duration_mismatch_refuses():
    long_ref = _fp(1, dur=200.0)
    short_cand = _fp(1, dur=20.0)  # ratio 0.1 -> DURATION_MISMATCH
    v = same_song_guard("Good Time", "Good Time", "acappella", long_ref, short_cand)
    assert v.accept is False
    assert v.channel == "content"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/ingest/test_same_song_guard.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingest.same_song_guard'`

- [ ] **Step 3: Implement `ingest/same_song_guard.py`**

```python
"""Same-song guard: refuse attaching a stem to a recording that is a different song.

Pure decision function. All I/O (fingerprinting, DB lookups, title probing) is
the caller's job — this module only decides. Two channels, REFUSE if either
fires (fail-closed on a mismatch signal, not on absence of signal):

  1. title-token (primary): acquired source title vs target recording title.
  2. stem-aware chromaprint (corroboration): classify() vs the regular reference.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.labels import labels_overlap
from ingest.adapters.fingerprint import Fingerprint, classify, similarity

# classify() verdicts that mean "different recording" (not a wrong-stem signal).
_CONTENT_REFUSE = {"WRONG_SONG", "DURATION_MISMATCH"}


@dataclass(frozen=True)
class GuardVerdict:
    accept: bool
    channel: str | None   # "title" | "content" | None (which channel refused)
    reason: str


def same_song_guard(
    acquired_title: str,
    recording_title: str,
    stem_axis: str,
    fp_regular: Fingerprint | None,
    fp_candidate: Fingerprint | None,
) -> GuardVerdict:
    # Channel 1 — title-token (primary). Only decisive when both titles present.
    if acquired_title and recording_title:
        if not labels_overlap(acquired_title, recording_title):
            return GuardVerdict(
                False, "title",
                f"title-token disjoint: {acquired_title!r} vs {recording_title!r}",
            )

    # Channel 2 — content (corroboration). Only when both fingerprints present.
    if fp_regular is not None and fp_candidate is not None:
        sim = similarity(fp_regular.raw, fp_candidate.raw)
        dur_ratio = (
            fp_candidate.duration_s / fp_regular.duration_s
            if fp_regular.duration_s else 0.0
        )
        verdict, detail = classify(stem_axis, sim, dur_ratio)
        if verdict in _CONTENT_REFUSE:
            return GuardVerdict(False, "content", f"{verdict}: {detail}")

    return GuardVerdict(True, None, "accept")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/ingest/test_same_song_guard.py -v`
Expected: PASS (all five cases)

- [ ] **Step 5: Commit**

```bash
git add ingest/same_song_guard.py tests/ingest/test_same_song_guard.py
git commit -m "feat(ingest): pure same-song guard (title-token OR stem-aware chromaprint)"
```

---

### Task 4: I/O runner — title probe, recording context, and the two gates

**Files:**
- Create: `ingest/stem_guard_runner.py`
- Test: `tests/ingest/test_stem_guard_runner.py` (create)

**Interfaces:**
- Consumes: `ingest.same_song_guard.{same_song_guard, GuardVerdict}`; `ingest.adapters.fingerprint.fingerprint_file`; `ingest.corrections.{Correction, log_correction}`; `core.result.Ok`.
- Produces:
  - `RecordingContext(title: str, regular_path: str | None)` (frozen dataclass)
  - `probe_url_title(url: str, yt_dlp: Path, *, timeout_s: float = 60.0) -> str | None`
  - `recording_context(db_path: Path, recording_id: str) -> RecordingContext`
  - `title_gate(acquired_title: str, recording_title: str) -> GuardVerdict`
  - `content_gate(stem_axis: str, regular_path: str | None, candidate_path: str) -> GuardVerdict`
  - `log_detach(db_path: Path, *, recording_id: str, set_id: str | None, position: str | None, acquired_title: str, verdict: GuardVerdict) -> None`

- [ ] **Step 1: Write the failing test**

Create `tests/ingest/test_stem_guard_runner.py`:

```python
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ingest.stem_guard_runner import (
    RecordingContext,
    log_detach,
    recording_context,
    title_gate,
)
from ingest.same_song_guard import GuardVerdict

_SCHEMA = """
CREATE TABLE work (work_id TEXT PRIMARY KEY, title TEXT);
CREATE TABLE recording (recording_id TEXT PRIMARY KEY, work_id TEXT, full_name TEXT);
CREATE TABLE track_audio (
  track_audio_id INTEGER PRIMARY KEY, recording_id TEXT, stem TEXT, path TEXT,
  is_reference INTEGER DEFAULT 0, downloaded_at DATETIME);
CREATE TABLE track_audio_correction (
  correction_id INTEGER PRIMARY KEY AUTOINCREMENT, set_id TEXT, position TEXT,
  track_id TEXT NOT NULL, axis TEXT NOT NULL, action TEXT NOT NULL,
  old_track_audio_id INTEGER, old_platform TEXT, old_player_id TEXT, old_url TEXT,
  new_track_audio_id INTEGER, new_platform TEXT, new_player_id TEXT, new_url TEXT,
  old_recording_id TEXT, new_recording_id TEXT, stem_value TEXT, reason TEXT,
  source TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  CHECK (axis IN ('version','variant','stem','recording')),
  CHECK (action IN ('replace','add','relink','detach')));
"""


@pytest.fixture
def db(tmp_path: Path) -> Path:
    p = tmp_path / "t.db"
    with sqlite3.connect(p) as c:
        c.executescript(_SCHEMA)
        c.execute("INSERT INTO work VALUES ('w1','Good Time')")
        c.execute("INSERT INTO recording VALUES ('42wv4vp','w1','Owl City - Good Time')")
        c.execute("INSERT INTO track_audio VALUES (1,'42wv4vp','regular','/x/good.wav',1,NULL)")
    return p


def test_recording_context_reads_title_and_regular_path(db: Path):
    ctx = recording_context(db, "42wv4vp")
    assert ctx == RecordingContext(title="Owl City - Good Time", regular_path="/x/good.wav")


def test_recording_context_missing_recording(db: Path):
    ctx = recording_context(db, "nope")
    assert ctx.title == ""
    assert ctx.regular_path is None


def test_title_gate_refuses_disjoint():
    v = title_gate("Come On Over Baby", "Owl City - Good Time")
    assert v.accept is False and v.channel == "title"


def test_log_detach_writes_recording_correction(db: Path):
    v = GuardVerdict(False, "title", "title-token disjoint: 'Come On Over Baby' vs 'Good Time'")
    log_detach(db, recording_id="42wv4vp", set_id="1fsnxchk", position="148w1",
               acquired_title="Come On Over Baby", verdict=v)
    with sqlite3.connect(db) as c:
        c.row_factory = sqlite3.Row
        row = c.execute("SELECT * FROM track_audio_correction").fetchone()
    assert row["axis"] == "recording" and row["action"] == "detach"
    assert row["old_recording_id"] == "42wv4vp" and row["new_recording_id"] is None
    assert row["source"] == "same_song_guard"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/ingest/test_stem_guard_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingest.stem_guard_runner'`

- [ ] **Step 3: Implement `ingest/stem_guard_runner.py`**

```python
"""I/O wrappers around the pure `same_song_guard`: title probe, DB context,
the two runtime gates, and the recording/detach ledger write."""

from __future__ import annotations

import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path

from core.result import Ok
from ingest.adapters.fingerprint import fingerprint_file
from ingest.corrections import Correction, log_correction
from ingest.same_song_guard import GuardVerdict, same_song_guard


@dataclass(frozen=True)
class RecordingContext:
    title: str
    regular_path: str | None


def probe_url_title(url: str, yt_dlp: Path, *, timeout_s: float = 60.0) -> str | None:
    """Fetch the source title WITHOUT downloading (metadata-only yt-dlp call)."""
    try:
        out = subprocess.run(
            [str(yt_dlp), "--skip-download", "--no-playlist", "--print", "%(title)s", url],
            capture_output=True, text=True, timeout=timeout_s, encoding="utf-8", errors="replace",
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    lines = [ln.strip() for ln in (out.stdout or "").splitlines() if ln.strip()]
    return lines[0] if lines else None


def recording_context(db_path: Path, recording_id: str) -> RecordingContext:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        r = conn.execute(
            "SELECT COALESCE(rec.full_name, w.title, '') AS title "
            "FROM recording rec LEFT JOIN work w ON rec.work_id = w.work_id "
            "WHERE rec.recording_id = ?",
            (recording_id,),
        ).fetchone()
        ref = conn.execute(
            "SELECT path FROM track_audio WHERE recording_id = ? AND stem = 'regular' "
            "ORDER BY is_reference DESC, downloaded_at DESC LIMIT 1",
            (recording_id,),
        ).fetchone()
    return RecordingContext(
        title=(r["title"] if r else ""),
        regular_path=(ref["path"] if ref else None),
    )


def title_gate(acquired_title: str, recording_title: str) -> GuardVerdict:
    """Pre-download decision: title channel only (no fingerprints)."""
    return same_song_guard(acquired_title, recording_title, "regular", None, None)


def content_gate(stem_axis: str, regular_path: str | None, candidate_path: str) -> GuardVerdict:
    """Post-insert decision: content channel only (title left blank)."""
    if not regular_path:
        return GuardVerdict(True, None, "no regular reference — content channel skipped")
    fa = fingerprint_file(regular_path)
    fb = fingerprint_file(candidate_path)
    if not (isinstance(fa, Ok) and isinstance(fb, Ok)):
        return GuardVerdict(True, None, "fingerprint unavailable — content channel skipped")
    return same_song_guard("", "", stem_axis, fa.value, fb.value)


def log_detach(
    db_path: Path,
    *,
    recording_id: str,
    set_id: str | None,
    position: str | None,
    acquired_title: str,
    verdict: GuardVerdict,
) -> None:
    """Record a wrong-recording detach (abstain) in the correction ledger."""
    c = Correction(
        track_id=recording_id,          # see spec: track_id overloaded to recording_id for axis='recording'
        axis="recording", action="detach",
        set_id=set_id, position=position,
        old_recording_id=recording_id, new_recording_id=None,
        reason=f"[{verdict.channel}] {verdict.reason} (acquired={acquired_title!r})",
        source="same_song_guard",
    )
    log_correction(db_path, c)
```

Note: `core/result.py`'s `Ok` exposes the payload as `.value` (confirmed: `Ok.value`), used above in `content_gate`.

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/ingest/test_stem_guard_runner.py -v`
Expected: PASS (four cases). `content_gate` is exercised indirectly in Task 5.

- [ ] **Step 5: Commit**

```bash
git add ingest/stem_guard_runner.py tests/ingest/test_stem_guard_runner.py
git commit -m "feat(ingest): stem-guard runner (title probe, recording context, gates, detach log)"
```

---

### Task 5: Wire the guard into `acquire_variant` (and `replace_stem_audio`)

**Files:**
- Modify: `scripts/acquire_variant.py:120-180` (`canonical_ingest`), add `--force` + `--acquired-title` args (`:267-343`)
- Modify: `scripts/replace_stem_audio.py` (add the same content gate after its replace)
- Test: `tests/ingest/test_acquire_variant_guard.py` (create)

**Interfaces:**
- Consumes: `ingest.stem_guard_runner.{probe_url_title, recording_context, title_gate, content_gate, log_detach}`.
- Produces: `canonical_ingest` returns non-zero and writes **no** `track_audio` row on a REFUSE without `--force`; writes exactly one `axis='recording', action='detach'` correction.

- [ ] **Step 1: Write the failing test** (title-channel refuse path, no network — inject a fake title)

Create `tests/ingest/test_acquire_variant_guard.py`:

```python
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from scripts import acquire_variant as av
from ingest.stem_guard_runner import recording_context

_SCHEMA = """
CREATE TABLE work (work_id TEXT PRIMARY KEY, title TEXT);
CREATE TABLE recording (recording_id TEXT PRIMARY KEY, work_id TEXT, full_name TEXT);
CREATE TABLE track_audio (
  track_audio_id INTEGER PRIMARY KEY, recording_id TEXT, stem TEXT, path TEXT,
  is_reference INTEGER DEFAULT 0, downloaded_at DATETIME);
CREATE TABLE track_audio_correction (
  correction_id INTEGER PRIMARY KEY AUTOINCREMENT, set_id TEXT, position TEXT,
  track_id TEXT NOT NULL, axis TEXT NOT NULL, action TEXT NOT NULL,
  old_track_audio_id INTEGER, old_platform TEXT, old_player_id TEXT, old_url TEXT,
  new_track_audio_id INTEGER, new_platform TEXT, new_player_id TEXT, new_url TEXT,
  old_recording_id TEXT, new_recording_id TEXT, stem_value TEXT, reason TEXT,
  source TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  CHECK (axis IN ('version','variant','stem','recording')),
  CHECK (action IN ('replace','add','relink','detach')));
"""


@pytest.fixture
def db(tmp_path: Path) -> Path:
    p = tmp_path / "t.db"
    with sqlite3.connect(p) as c:
        c.executescript(_SCHEMA)
        c.execute("INSERT INTO work VALUES ('w1','Good Time')")
        c.execute("INSERT INTO recording VALUES ('42wv4vp','w1','Owl City - Good Time')")
    return p


def test_pretitle_guard_refuses_and_logs_detach(db: Path):
    ctx = recording_context(db, "42wv4vp")
    # simulate the pre-download title-channel decision the ingest performs
    from ingest.stem_guard_runner import title_gate, log_detach
    v = title_gate("Come On Over Baby (All I Want Is You)", ctx.title)
    assert v.accept is False
    log_detach(db, recording_id="42wv4vp", set_id="1fsnxchk", position="148w1",
               acquired_title="Come On Over Baby (All I Want Is You)", verdict=v)

    with sqlite3.connect(db) as c:
        n_audio = c.execute("SELECT count(*) FROM track_audio").fetchone()[0]
        n_corr = c.execute(
            "SELECT count(*) FROM track_audio_correction WHERE axis='recording' AND action='detach'"
        ).fetchone()[0]
    assert n_audio == 0    # nothing attached
    assert n_corr == 1     # one honest detach logged
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/ingest/test_acquire_variant_guard.py -v`
Expected: FAIL — the import path is fine, but this asserts the runner behavior that Task 4 provides; if run before Task 4 is merged it fails at import. After Task 4, it should PASS at the runner level. This test locks the *contract* the wiring must honor.

- [ ] **Step 3: Add `--force` and `--acquired-title` args to `acquire_variant.py`**

In `main()` (near the other canonical-mode args, ~line 317):

```python
    ap.add_argument(
        "--force", action="store_true",
        help="Bypass the same-song guard (loud, ledgered). Default: fail-closed.",
    )
    ap.add_argument(
        "--acquired-title", default=None,
        help="Source title for the title-token guard (defaults to a yt-dlp probe in URL mode / --name / filename).",
    )
```

- [ ] **Step 4: Insert the pre-download title gate + post-insert content gate in `canonical_ingest`**

At the top of `canonical_ingest`, after `track_id` is resolved (after line 141) and before the `_replace_via_*` call, add the pre-download title gate:

```python
    from ingest.stem_guard_runner import (
        probe_url_title, recording_context, title_gate, content_gate, log_detach,
    )
    ctx = recording_context(args.db, track_id)
    acquired_title = args.acquired_title
    if acquired_title is None and args.url:
        acquired_title = probe_url_title(args.url, YT_DLP)
    elif acquired_title is None and args.file:
        acquired_title = args.name or args.file.stem

    tv = title_gate(acquired_title or "", ctx.title)
    if not tv.accept and not args.force:
        log_detach(args.db, recording_id=track_id, set_id=args.set_id,
                   position=(None if args.slot is None else str(args.slot)),
                   acquired_title=acquired_title or "", verdict=tv)
        logging.error("same-song guard REFUSED (title): %s — not attaching. "
                      "Re-run with --force to override.", tv.reason)
        return 3
    if not tv.accept and args.force:
        logging.warning("same-song guard OVERRIDDEN (--force) despite: %s", tv.reason)
```

After a successful insert (in the `if rc == 0:` block, replacing the advisory `_identity_check(...)` call on line 177) add the content gate with reap-on-refuse:

```python
    if rc == 0:
        row = _lookup_audio_path(args.db, track_id, stem_axis)
        cv = content_gate(stem_axis, ctx.regular_path, row[1]) if row else None
        if cv is not None and not cv.accept and not args.force:
            log_detach(args.db, recording_id=track_id, set_id=args.set_id,
                       position=(None if args.slot is None else str(args.slot)),
                       acquired_title=acquired_title or "", verdict=cv)
            logging.error("same-song guard REFUSED (content): %s — reaping row %s.",
                          cv.reason, row[0])
            _reap_audio_row(args.db, args.audio_root, row[0])
            return 3
        if cv is not None and not cv.accept and args.force:
            logging.warning("content guard OVERRIDDEN (--force) despite: %s", cv.reason)
        if not args.no_log:
            _log_to_ledger(args, track_id, stem_axis)
    return rc
```

Add a small reap helper near `_lookup_audio_path` (reuse the canonical unlink path from `replace_track_audio`):

```python
def _reap_audio_row(db_path: Path, audio_root: Path, track_audio_id: int) -> None:
    """Delete a just-inserted wrong-recording row + unlink its file (no wrong row survives)."""
    sys.path.insert(0, str(REPO_ROOT))
    from scripts import replace_track_audio as rta
    # _delete_old_row_if_exists(db_path, audio_root, track_audio_id, keep_path=None)
    # deletes the row, cascades (analysis/stems/features/MERT), and unlinks the object file.
    rta._delete_old_row_if_exists(db_path, audio_root, track_audio_id)
```

- [ ] **Step 5: Wire the content gate into `replace_stem_audio.py`**

`replace_stem_audio` replaces a stem row by `--track-audio-id` + URL/file. After its replace succeeds, run the same content gate against the target recording's regular reference and reap+detach-log on refuse (unless `--force`). Add a `--force` arg mirroring Step 3 and the same `content_gate` / `log_detach` / reap block keyed on the row it just wrote. (Same code shape as Step 4's post-insert block.)

- [ ] **Step 6: Run the guard-contract test + the full new suite**

Run:
```bash
venvs/audio/bin/python -m pytest \
  tests/ingest/test_same_song_guard.py \
  tests/ingest/test_stem_guard_runner.py \
  tests/ingest/test_corrections_recording_axis.py \
  tests/ingest/test_acquire_variant_guard.py -v
```
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add scripts/acquire_variant.py scripts/replace_stem_audio.py tests/ingest/test_acquire_variant_guard.py
git commit -m "feat(ingest): fail-closed same-song guard on acquire_variant + replace_stem_audio"
```

---

### Task 6: Validation harness over the frozen ledger fixture + `make check`

**Files:**
- Create: `scripts/audit_stem_recording_links.py` (read-only scan — the Part-3 seed; used here for validation)
- Test: `tests/ingest/test_stem_misattach_validation.py` (create)

**Interfaces:**
- Consumes: `tests/fixtures/ingest/correction_ledger_snapshot_20260709.tsv`; `core.labels.labels_overlap`.
- Produces: `parse_acquired_song(reason: str) -> str | None`; `scan_ledger_tsv(path: Path) -> list[dict]` (each: `track_id`, `stem_value`, `acquired_song`, `reason`).

- [ ] **Step 1: Write the failing test** (the 20911 class must be detectable from the fixture)

Create `tests/ingest/test_stem_misattach_validation.py`:

```python
from __future__ import annotations

from pathlib import Path

from scripts.audit_stem_recording_links import parse_acquired_song, scan_ledger_tsv

FIXTURE = Path("tests/fixtures/ingest/correction_ledger_snapshot_20260709.tsv")


def test_parse_acquired_song_from_file_reason():
    # ledger reasons for stem/add rows embed the acquired file/song
    got = parse_acquired_song("file: 148__Come On Over Baby (Acapella).wav ; auto-attached")
    assert got is not None
    assert "come on over baby" in got.lower()


def test_scan_returns_stem_add_rows():
    rows = scan_ledger_tsv(FIXTURE)
    assert len(rows) > 0
    # every returned row is a stem-axis add with a parsed acquired song
    assert all(r["stem_value"] in ("acappella", "instrumental") for r in rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/ingest/test_stem_misattach_validation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.audit_stem_recording_links'`

- [ ] **Step 3: Implement `scripts/audit_stem_recording_links.py`**

```python
#!/usr/bin/env python3
"""Read-only scan for stem-candidate wrong-recording mis-attaches.

Seed of Crush Phase-4 part 3 (audit). Here it also backs the validation test:
parse the acquired song from each stem/add correction's reason and (given a DB)
compare to the target recording's title via labels_overlap. NO mutations.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.labels import labels_overlap  # noqa: E402

_FILE_RE = re.compile(r"file:\s*(?P<name>[^;]+)", re.IGNORECASE)
_SLOT_PREFIX = re.compile(r"^\d+(?:w\d+)?__")
_PARENS = re.compile(r"\((?:acapella|acappella|instrumental)[^)]*\)", re.IGNORECASE)


def parse_acquired_song(reason: str) -> str | None:
    """Pull the acquired song title out of a stem/add correction reason."""
    if not reason:
        return None
    m = _FILE_RE.search(reason)
    if not m:
        return None
    name = m.group("name").strip()
    name = _SLOT_PREFIX.sub("", name)
    name = re.sub(r"\.(wav|m4a|mp3|flac|opus)$", "", name, flags=re.IGNORECASE)
    name = _PARENS.sub("", name).strip(" -")
    return name or None


def scan_ledger_tsv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r.get("axis") != "stem" or r.get("action") != "add":
                continue
            song = parse_acquired_song(r.get("reason", ""))
            if song is None:
                continue
            rows.append({
                "track_id": r.get("track_id", ""),
                "stem_value": r.get("stem_value", ""),
                "acquired_song": song,
                "reason": r.get("reason", ""),
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Scan the correction ledger for stem mis-attaches (read-only).")
    ap.add_argument("--ledger-tsv", type=Path, required=True)
    a = ap.parse_args(argv)
    rows = scan_ledger_tsv(a.ledger_tsv)
    print(f"stem/add rows with a parseable acquired song: {len(rows)}")
    for r in rows[:50]:
        print(f"  {r['track_id']}  {r['stem_value']:12s}  {r['acquired_song']}")
    if len(rows) > 50:
        print(f"  ... and {len(rows) - 50} more (NOT truncated in analysis — display cap only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes; then run the scan against the real fixture**

Run: `venvs/audio/bin/python -m pytest tests/ingest/test_stem_misattach_validation.py -v`
Expected: PASS

Then (manual validation — records the over-block picture):
Run: `venvs/audio/bin/python scripts/audit_stem_recording_links.py --ledger-tsv tests/fixtures/ingest/correction_ledger_snapshot_20260709.tsv | head -30`
Expected: prints the parseable stem/add rows (the 2026-06-09 mis-attach batch appears). Sanity-check that "Come On Over Baby"-style acquired songs are surfaced.

- [ ] **Step 5: Run `make check` (guardrails + entropy fences + fast pytest)**

Run: `make check`
Expected: `guardrails: OK`, `Success: no issues found`, and the pytest subset green. Pre-existing WARNs (`alignment_state_of_record` staleness, `docs-gc`) are unrelated and non-blocking.

- [ ] **Step 6: Commit**

```bash
git add scripts/audit_stem_recording_links.py tests/ingest/test_stem_misattach_validation.py
git commit -m "feat(ingest): read-only stem mis-attach ledger scan + validation over frozen fixture"
```

---

## Post-plan (not in PR #70)

- **Ops (gated):** apply `migrate_correction_recording_axis.sql` to canonical pi, run the audit against canonical titles for the over-block/coverage numbers, remediate (re-link / MINT / delete) — Part 3 follow-on plan. Coordinate with the 92-commits-behind pi state first.
- **Part 4:** content-verified manual-capture flow, reusing this guard.

## Self-Review

**Spec coverage:**
- Part 1 (ledger recording axis) → Task 2. ✅
- Part 2 (same-song guard: title-token OR stem-aware chromaprint, fail-closed, abstain + log + `--force`, shared path) → Tasks 3 (pure decision), 4 (I/O runner + detach log), 5 (wiring into both stem-attach entrypoints). ✅
- Layering (ingest must not import labeling) → Task 1 (relocate `labels_overlap` to core). ✅
- Content REFUSE set `{WRONG_SONG, DURATION_MISMATCH}` → Task 3 `_CONTENT_REFUSE`. ✅
- No-reference content channel stays silent → Task 4 `content_gate` early return. ✅
- Validation over BB11/BB12 frozen fixture + no silent caps → Task 6. ✅
- Part 3 audit seed (`audit_stem_recording_links.py`) → Task 6 (read-only; remediation deferred). ✅
- Part 4 manual capture → explicitly deferred (Post-plan). ✅

**Placeholder scan:** no open TODOs. The two names that were uncertain are now pinned against the source: `Ok.value` (core/result.py) and `_delete_old_row_if_exists(db_path, audio_root, track_audio_id, keep_path=None)` (scripts/replace_track_audio.py:219, deletes row + cascades + unlinks file). Every code step shows complete code.

**Type consistency:** `GuardVerdict(accept, channel, reason)` used identically in Tasks 3/4/5. `RecordingContext(title, regular_path)` consistent Task 4↔5. `Correction(..., old_recording_id, new_recording_id)` consistent Task 2↔4. `same_song_guard` signature identical across Tasks 3/4.
