# Content-addressed GT identity binding — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bind every GT `.als` clip to its `recording_id` by audio content (not slot/path), abstaining on a miss, so the `slot_id_map` guess-ladder can be deleted and BB12/BB11 GT stop carrying cross-song ids.

**Architecture:** A pull-time `content_catalog.json` sidecar (built from pi `track_audio.sha256` + demucs stems hashed on pi) maps two content keys — full-file sha256 and a tag-invariant mp4 `mdat` payload sha256 — to `recording_id`. The offline exporter resolves each clip's local file against that catalog via the landed `labeling/content_resolver.py`, stamps `id_source = content | abstain`, and no longer consults `slot_id_map`.

**Tech Stack:** Python 3 (stdlib `hashlib`/`struct` for hashing — no new deps), SQLite (pi canonical DB over SSH), existing `labeling/als` codec, PyYAML GT schema, pytest.

## Global Constraints

- Run tests from repo root with `venvs/audio/bin/python`. The worktree has `venvs` symlinked to the main checkout so the pre-commit hook finds `venvs/audio/bin/python`.
- New hashing code is **stdlib-only** (`hashlib`, `struct`) so `build_content_catalog` can run under pi's bare `python3` (no venv).
- `track_audio.sha256` is full-file chunked sha256 (`ingest/adapters/downloader.py::_sha256`, 1 MiB chunks). Any full-file hash MUST match it byte-for-byte (same chunked sha256).
- **Abstain, don't guess.** A null/abstained id is correct; a confidently-wrong cross-song id is the poison. Never derive `recording_id` from a slot number or filename.
- pi-storage is canonical: DB `/mnt/storage/data/db/music_database.db`; do not trust the local `data/db/music_database.db`. `track_audio.path` may carry UTF-8/Latin-1 mojibake.
- Set ids: BB12 = `1fsnxchk`, BB11 = `2nvzlh2k`.
- Never hand-edit `docs/alignment_status.md`; only `/align-checkpoint` writes numbers there.
- Frozen dataclasses, `from __future__ import annotations`, Result-in-core / fail-fast-at-edge (per repo style guide).
- Commit per task. Branch: `crush/depoison-content-binding` (worktree `../tracklist_engine__crush-depoison`).

---

## File structure

- **Create** `labeling/content_hash.py` — stdlib content-hash primitives (`file_sha256`, `mdat_sha256`). Imported by both the catalog builder (pi) and the exporter (Mac).
- **Create** `labeling/build_content_catalog.py` — pure `build_catalog(conn, set_id, ...)` + CLI `main()` that prints `content_catalog.json` to stdout; runs on pi.
- **Modify** `labeling/ground_truth/schema.py` — add `id_source` to `GroundTruthTrack`; serialize/parse.
- **Modify** `labeling/export_als_to_gt.py` — load the sidecar, content-bind in `_clip_row`, stamp `id_source`, delete `slot_id_map`, re-base the coverage gate.
- **Delete** `labeling/fixtures/id_maps/1fsnxchk_slots.json`, `2nvzlh2k_slots.json`.
- **Modify** `labeling/pull_set_for_alignment.py` — emit the sidecar after `manifest.json`.
- **Create** `tests/labeling/test_content_hash.py`, `tests/labeling/test_build_content_catalog.py`, `tests/labeling/test_content_identity_metamorphic.py`.
- **Modify** `tests/labeling/` export tests as noted per task.

---

## Task 1: Content-hash primitives (`labeling/content_hash.py`)

**Files:**
- Create: `labeling/content_hash.py`
- Test: `tests/labeling/test_content_hash.py`

**Interfaces:**
- Produces: `file_sha256(path: str | Path) -> str` (full-file, 1 MiB chunks — identical to `track_audio.sha256`); `mdat_sha256(path: str | Path) -> str | None` (sha256 of the mp4 top-level `mdat` box payload; `None` if no `mdat`, e.g. non-mp4).

- [ ] **Step 1: Write the failing test**

```python
# tests/labeling/test_content_hash.py
"""content_hash: full-file sha256 == track_audio.sha256; mdat hash is tag-invariant."""
from __future__ import annotations

import hashlib
import struct
from pathlib import Path

from labeling.content_hash import file_sha256, mdat_sha256


def _write_min_mp4(path: Path, mdat_payload: bytes, tag_blob: bytes) -> None:
    # ftyp, then a 'moov'->'udta' carrying tag_blob, then 'mdat' with payload.
    def box(typ: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body) + 8) + typ + body

    ftyp = box(b"ftyp", b"isom" + b"\x00\x00\x02\x00" + b"isomiso2")
    udta = box(b"udta", box(b"\xa9nam", tag_blob))
    moov = box(b"moov", udta)
    mdat = box(b"mdat", mdat_payload)
    path.write_bytes(ftyp + moov + mdat)


def test_file_sha256_matches_hashlib(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello world" * 1000)
    assert file_sha256(p) == hashlib.sha256(p.read_bytes()).hexdigest()


def test_mdat_hash_is_tag_invariant(tmp_path: Path) -> None:
    payload = b"AUDIO-PAYLOAD-BYTES" * 500
    a = tmp_path / "a.m4a"
    b = tmp_path / "b.m4a"
    _write_min_mp4(a, payload, tag_blob=b"tags-before")
    _write_min_mp4(b, payload, tag_blob=b"COMPLETELY-DIFFERENT-LONGER-TAGS")
    # Different container metadata -> different full-file hash ...
    assert file_sha256(a) != file_sha256(b)
    # ... but identical audio payload -> identical mdat hash.
    assert mdat_sha256(a) == mdat_sha256(b)
    assert mdat_sha256(a) == hashlib.sha256(payload).hexdigest()


def test_mdat_hash_none_for_non_mp4(tmp_path: Path) -> None:
    p = tmp_path / "not.mp4"
    p.write_bytes(b"fLaC" + b"\x00" * 100)
    assert mdat_sha256(p) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/labeling/test_content_hash.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'labeling.content_hash'`.

- [ ] **Step 3: Write minimal implementation**

```python
# labeling/content_hash.py
"""Content-addressed audio identity primitives (stdlib only — runs on pi's bare python3).

Two keys bind a clip to a recording:
  * file_sha256  — full-file sha256, IDENTICAL to track_audio.sha256
                   (ingest/adapters/downloader.py::_sha256, 1 MiB chunks).
  * mdat_sha256  — sha256 of the mp4 top-level `mdat` box payload. iTunes tag
                   injection (tag_aligning_folder.py) rewrites moov/udta atoms but
                   never the `mdat` audio payload, so this is tag-invariant: a
                   locally-tagged master hashes to the same value as pi's canonical
                   file. Validated 2026-07-21 (Chainsmokers "Honest" mdat fe374e…
                   == pi 2vmxu50p).
"""
from __future__ import annotations

import hashlib
import struct
from pathlib import Path

_CHUNK = 1 << 20


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def mdat_sha256(path: str | Path) -> str | None:
    """sha256 of the first top-level `mdat` box payload, or None if there is none."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                return None
            size = struct.unpack(">I", hdr[:4])[0]
            typ = hdr[4:8]
            if size == 1:  # 64-bit extended size
                size = struct.unpack(">Q", f.read(8))[0]
                hdrlen = 16
            else:
                hdrlen = 8
            payload = size - hdrlen
            if typ == b"mdat":
                remaining = payload
                while remaining > 0:
                    chunk = f.read(min(_CHUNK, remaining))
                    if not chunk:
                        break
                    h.update(chunk)
                    remaining -= len(chunk)
                return h.hexdigest()
            if size == 0:
                return None
            f.seek(payload, 1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/labeling/test_content_hash.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add labeling/content_hash.py tests/labeling/test_content_hash.py
git commit -m "feat(labeling): stdlib content-hash primitives (file + tag-invariant mdat sha256)"
```

---

## Task 2: `id_source` on `GroundTruthTrack`

**Files:**
- Modify: `labeling/ground_truth/schema.py` (dataclass `GroundTruthTrack`; `dump`; `_parse_track`)
- Test: `tests/labeling/test_gt_id_source.py` (create)

**Interfaces:**
- Produces: `GroundTruthTrack.id_source: str` (values `""` legacy | `"content"` | `"abstain"`), round-tripped through `dump`/`load`.

- [ ] **Step 1: Write the failing test**

```python
# tests/labeling/test_gt_id_source.py
from __future__ import annotations

from pathlib import Path

from labeling.ground_truth.schema import (
    GroundTruthSet,
    GroundTruthTrack,
    dump,
    load,
)


def _track(**kw) -> GroundTruthTrack:
    base = dict(label="X", track_id="rec1", claimed_stem="regular",
                set_start_s=1.0, set_end_s=2.0, ref_start_s=0.0)
    base.update(kw)
    return GroundTruthTrack(**base)


def test_id_source_defaults_empty() -> None:
    assert _track().id_source == ""


def test_id_source_round_trips(tmp_path: Path) -> None:
    gt = GroundTruthSet(set_id="s", tracks=(
        _track(id_source="content"),
        _track(track_id=None, id_source="abstain"),
    ))
    p = tmp_path / "gt.yaml"
    p.write_text(dump(gt))
    back = load(p)
    assert back.is_ok()
    assert [t.id_source for t in back.value.tracks] == ["content", "abstain"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/labeling/test_gt_id_source.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'id_source'`.

- [ ] **Step 3: Write minimal implementation**

In `labeling/ground_truth/schema.py`, add the field to `GroundTruthTrack` (after `source_note`):

```python
    id_source: str = ""  # "" legacy | "content" (bound by sha256/mdat) | "abstain"
```

In `dump`, emit it right after the `source_note` block (only when set):

```python
        if t.id_source:
            out.append(f"    id_source:   {t.id_source}")
```

In `_parse_track`, before the `return Ok(GroundTruthTrack(...))`, read it and pass it through:

```python
    id_source = str(t.get("id_source") or "").strip()
```

and add `id_source=id_source,` to the `GroundTruthTrack(...)` constructor call.

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/labeling/test_gt_id_source.py tests/labeling/ -q`
Expected: PASS (new tests green; existing schema tests unaffected).

- [ ] **Step 5: Commit**

```bash
git add labeling/ground_truth/schema.py tests/labeling/test_gt_id_source.py
git commit -m "feat(labeling): id_source (content|abstain) on GroundTruthTrack"
```

---

## Task 3: Catalog loader + two-pass content bind (exporter side)

**Files:**
- Modify: `labeling/export_als_to_gt.py` (add `_load_content_catalog`, `_content_bind`)
- Test: `tests/labeling/test_content_bind.py` (create)

**Interfaces:**
- Consumes: `labeling.content_resolver.{ContentCatalog, CatalogEntry, resolve_clip_identity}`; `labeling.content_hash.{file_sha256, mdat_sha256}`; `labeling.als.models.ParsedClip`.
- Produces:
  - `_load_content_catalog(set_dir: Path) -> ContentCatalog | None` — reads `set_dir/content_catalog.json`; `None` if absent. Registers TWO `CatalogEntry` per catalog row (one keyed `head_hash=content_sha256`, one `head_hash=payload_sha256` when present), both carrying the same identity.
  - `_content_bind(clip: ParsedClip, catalog: ContentCatalog) -> tuple[str | None, str]` — returns `(recording_id, id_source)` where `id_source` is `"content"` on a hit else `"abstain"`. Two passes: `head_hash_of=file_sha256`, then (on Err, for `.m4a`) `head_hash_of=mdat_sha256`. A missing file on disk → abstain (never raises).

- [ ] **Step 1: Write the failing test**

```python
# tests/labeling/test_content_bind.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from labeling.als.models import ParsedClip, WarpMarkers
from labeling.export_als_to_gt import _content_bind, _load_content_catalog


def _clip(path: str) -> ParsedClip:
    return ParsedClip(
        group_name="", track_name="", path=path,
        arr_start=0.0, arr_end=1.0, loop_start=0.0, loop_end=1.0,
        pitch_coarse=0, pitch_fine=0,
        warp=WarpMarkers(points=((0.0, 0.0), (1.0, 1.0))),
    )


def _catalog(set_dir: Path, entries: list[dict]) -> None:
    (set_dir / "content_catalog.json").write_text(
        json.dumps({"set_id": "s", "entries": entries})
    )


def test_binds_by_full_file_sha256(tmp_path: Path) -> None:
    f = tmp_path / "cand.m4a"
    f.write_bytes(b"CANDIDATE-BYTES" * 100)
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    _catalog(tmp_path, [{"content_sha256": sha, "payload_sha256": None,
                         "recording_id": "recX", "track_audio_id": "ta1",
                         "stem": "acappella"}])
    cat = _load_content_catalog(tmp_path)
    assert _content_bind(_clip(str(f)), cat) == ("recX", "content")


def test_abstains_when_bytes_not_in_catalog(tmp_path: Path) -> None:
    f = tmp_path / "unknown.m4a"
    f.write_bytes(b"NOT-CATALOGUED" * 50)
    _catalog(tmp_path, [{"content_sha256": "deadbeef", "payload_sha256": None,
                         "recording_id": "recX", "track_audio_id": "ta1",
                         "stem": "regular"}])
    cat = _load_content_catalog(tmp_path)
    assert _content_bind(_clip(str(f)), cat) == (None, "abstain")


def test_missing_file_abstains(tmp_path: Path) -> None:
    _catalog(tmp_path, [])
    cat = _load_content_catalog(tmp_path)
    assert _content_bind(_clip(str(tmp_path / "gone.m4a")), cat) == (None, "abstain")
```

Add an mdat-payload bind test using `content_hash.mdat_sha256` and the `_write_min_mp4` helper (import it or inline the same helper) to assert a tag-mutated master binds via `payload_sha256`:

```python
def test_binds_tagged_master_by_mdat(tmp_path: Path) -> None:
    import struct
    from labeling.content_hash import mdat_sha256
    def box(t, b): return struct.pack(">I", len(b) + 8) + t + b
    payload = b"MASTER-AUDIO" * 200
    f = tmp_path / "154__A - B [100bpm 5B].m4a"
    f.write_bytes(box(b"ftyp", b"isom") + box(b"moov", box(b"udta", b"tag"))
                  + box(b"mdat", payload))
    _catalog(tmp_path, [{"content_sha256": "not-the-file",
                         "payload_sha256": mdat_sha256(f),
                         "recording_id": "recM", "track_audio_id": "ta9",
                         "stem": "regular"}])
    cat = _load_content_catalog(tmp_path)
    assert _content_bind(_clip(str(f)), cat) == ("recM", "content")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/labeling/test_content_bind.py -q`
Expected: FAIL — `ImportError: cannot import name '_content_bind'`.

- [ ] **Step 3: Write minimal implementation**

Add to `labeling/export_als_to_gt.py` (imports at top, functions near `_clip_row`):

```python
from labeling.content_hash import file_sha256, mdat_sha256
from labeling.content_resolver import CatalogEntry, ContentCatalog, resolve_clip_identity


def _load_content_catalog(set_dir: Path) -> ContentCatalog | None:
    p = set_dir / "content_catalog.json"
    if not p.is_file():
        return None
    payload = json.loads(p.read_text())
    entries: list[CatalogEntry] = []
    for e in payload.get("entries") or []:
        rid = e.get("recording_id")
        taid = str(e.get("track_audio_id") or "")
        stem = str(e.get("stem") or "regular")
        for key in (e.get("content_sha256"), e.get("payload_sha256")):
            if key:
                entries.append(CatalogEntry(
                    track_audio_id=taid, recording_id=rid, stem=stem,
                    head_hash=str(key),
                ))
    return ContentCatalog.from_entries(entries)


def _content_bind(clip, catalog: ContentCatalog | None) -> tuple[str | None, str]:
    """(recording_id, id_source): bind by content or abstain. Never raises."""
    if catalog is None:
        return None, "abstain"

    def _safe(hasher):
        def _inner(path: str) -> str | None:
            try:
                return hasher(path)
            except OSError:
                return None
        return _inner

    r = resolve_clip_identity(clip, catalog, head_hash_of=_safe(file_sha256))
    if not r.is_ok() and clip.path.lower().endswith((".m4a", ".mp4", ".m4b")):
        r = resolve_clip_identity(clip, catalog, head_hash_of=_safe(mdat_sha256))
    if r.is_ok():
        return r.value.recording_id, "content"
    return None, "abstain"
```

Note: `resolve_clip_identity` first tries `by_size_crc` (empty here) then `by_head_hash` via `head_hash_of` — so the two-pass swap of `head_hash_of` is the mechanism. `json` is already imported in the module.

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/labeling/test_content_bind.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add labeling/export_als_to_gt.py tests/labeling/test_content_bind.py
git commit -m "feat(labeling): content-catalog loader + two-pass content bind (sha256 -> mdat)"
```

---

## Task 4: Catalog builder (`labeling/build_content_catalog.py`, runs on pi)

**Files:**
- Create: `labeling/build_content_catalog.py`
- Test: `tests/labeling/test_build_content_catalog.py`

**Interfaces:**
- Consumes: `sqlite3.Connection` to the canonical DB; `labeling.content_hash.{file_sha256, mdat_sha256}`.
- Produces:
  - `build_catalog(conn, set_id, *, file_sha256=..., mdat_sha256=...) -> dict` returning `{"set_id": set_id, "entries": [ {content_sha256, payload_sha256, recording_id, track_audio_id, stem} ... ]}`. `track_audio` rows contribute `content_sha256` from the DB column and `payload_sha256` from `mdat_sha256(path)` when the path is `.m4a`/`.mp4` and readable (else `None`). Demucs stems (`track_stems` joined to `track_audio` for the set's recordings) contribute `content_sha256 = file_sha256(stem_path)`, `payload_sha256=None`, `stem = acappella` (vocals) / `instrumental`, `recording_id` from the parent `track_audio`.
  - `main(argv=None) -> int` — CLI: `python3 -m labeling.build_content_catalog <set_id>` prints the JSON to stdout.

- [ ] **Step 1: Write the failing test**

```python
# tests/labeling/test_build_content_catalog.py
from __future__ import annotations

import sqlite3
from pathlib import Path

from labeling.build_content_catalog import build_catalog


def _db(tmp_path: Path) -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.executescript(
        """
        CREATE TABLE set_track_slots(set_id TEXT, row_index INTEGER,
            recording_id TEXT, track_id TEXT);
        CREATE TABLE track_audio(track_audio_id INTEGER PRIMARY KEY,
            recording_id TEXT, stem TEXT, sha256 TEXT, path TEXT);
        CREATE TABLE track_stems(track_audio_id INTEGER, stem_name TEXT, path TEXT);
        """
    )
    return c


def test_build_catalog_covers_track_audio_and_stems(tmp_path: Path) -> None:
    conn = _db(tmp_path)
    vocals = tmp_path / "vocals.flac"
    vocals.write_bytes(b"VOCALS-STEM-BYTES" * 100)
    conn.executemany(
        "INSERT INTO set_track_slots VALUES(?,?,?,?)",
        [("s", 0, "recA", "recA"), ("s", 1, "recB", "recB")],
    )
    conn.executemany(
        "INSERT INTO track_audio VALUES(?,?,?,?,?)",
        [(1, "recA", "regular", "shaA", "/x/a.m4a"),
         (2, "recB", "acappella", "shaB", "/x/b.m4a")],
    )
    conn.execute("INSERT INTO track_stems VALUES(1, 'vocals', ?)", (str(vocals),))

    out = build_catalog(
        conn, "s",
        file_sha256=lambda p: "STEMHASH" if p == str(vocals) else "?",
        mdat_sha256=lambda p: None,  # skip real mp4 parsing in unit test
    )
    got = {(e["recording_id"], e["stem"], e["content_sha256"]) for e in out["entries"]}
    assert ("recA", "regular", "shaA") in got
    assert ("recB", "acappella", "shaB") in got
    assert ("recA", "acappella", "STEMHASH") in got  # demucs vocals -> acappella
    assert out["set_id"] == "s"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/labeling/test_build_content_catalog.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'labeling.build_content_catalog'`.

- [ ] **Step 3: Write minimal implementation**

```python
# labeling/build_content_catalog.py
"""Build a set's content_catalog.json from the canonical DB (runs on pi).

Emits {content_sha256, payload_sha256, recording_id, track_audio_id, stem} per
audio artifact a GT clip can reference: every track_audio row for the set's
recordings (content_sha256 from the DB; payload_sha256 = mdat hash for m4a), plus
demucs vocals/instrumental stems (hashed here — track_stems has no stored hash).

stdlib only; run under pi's bare python3:
    python3 -m labeling.build_content_catalog <set_id>   # prints JSON to stdout
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from labeling.content_hash import file_sha256 as _file_sha256
from labeling.content_hash import mdat_sha256 as _mdat_sha256

_DB = "/mnt/storage/data/db/music_database.db"
_M4A_EXT = (".m4a", ".mp4", ".m4b")
_STEM_TO_AXIS = {"vocals": "acappella", "instrumental": "instrumental"}


def build_catalog(conn, set_id, *, file_sha256=_file_sha256, mdat_sha256=_mdat_sha256):
    recs = [r[0] for r in conn.execute(
        "SELECT DISTINCT recording_id FROM set_track_slots "
        "WHERE set_id=? AND recording_id IS NOT NULL", (set_id,))]
    entries: list[dict] = []
    if not recs:
        return {"set_id": set_id, "entries": entries}
    qmarks = ",".join("?" * len(recs))

    for taid, rid, stem, sha, path in conn.execute(
        f"SELECT track_audio_id, recording_id, stem, sha256, path FROM track_audio "
        f"WHERE recording_id IN ({qmarks})", recs):
        payload = None
        p = str(path or "")
        if p.lower().endswith(_M4A_EXT):
            try:
                payload = mdat_sha256(p)
            except OSError:
                payload = None
        entries.append({
            "content_sha256": sha, "payload_sha256": payload,
            "recording_id": rid, "track_audio_id": str(taid),
            "stem": stem or "regular",
        })

    for taid, rid, stem_name, spath in conn.execute(
        f"SELECT ts.track_audio_id, ta.recording_id, ts.stem_name, ts.path "
        f"FROM track_stems ts JOIN track_audio ta ON ta.track_audio_id=ts.track_audio_id "
        f"WHERE ta.recording_id IN ({qmarks}) AND ts.stem_name IN ('vocals','instrumental')",
        recs):
        try:
            csha = file_sha256(str(spath))
        except OSError:
            continue
        entries.append({
            "content_sha256": csha, "payload_sha256": None,
            "recording_id": rid, "track_audio_id": str(taid),
            "stem": _STEM_TO_AXIS.get(stem_name, stem_name),
        })
    return {"set_id": set_id, "entries": entries}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: build_content_catalog <set_id> [db_path]", file=sys.stderr)
        return 2
    set_id = argv[0]
    db = argv[1] if len(argv) > 1 else _DB
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        print(json.dumps(build_catalog(conn, set_id)))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/labeling/test_build_content_catalog.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add labeling/build_content_catalog.py tests/labeling/test_build_content_catalog.py
git commit -m "feat(labeling): build_content_catalog (track_audio sha + mdat + demucs stems)"
```

---

## Task 5: Wire the exporter to content-bind; delete `slot_id_map`

**Files:**
- Modify: `labeling/export_als_to_gt.py` (`_clip_row`, `ClipRow`, `collect_kept_clip_rows`, `_to_gt_track`; remove `_load_slot_id_map`)
- Delete: `labeling/fixtures/id_maps/1fsnxchk_slots.json`, `labeling/fixtures/id_maps/2nvzlh2k_slots.json`
- Test: `tests/labeling/test_export_content_identity.py` (create)

**Interfaces:**
- Consumes: `_load_content_catalog`, `_content_bind` (Task 3); `ClipRow` gains `id_source: str = ""`.
- Produces: `_clip_row(clip, mapper, manifest, catalog)` — `recording_id` and `id_source` come from `_content_bind`; slot/display/stem still from `resolve_identity`. `_to_gt_track` passes `id_source` through.

- [ ] **Step 1: Write the failing test**

```python
# tests/labeling/test_export_content_identity.py
"""Content binding replaces the slot_id_map guess: a clip whose slot 'looks like'
a different recording must bind to its own audio content or abstain — never the
slot's id."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from labeling.als.models import ParsedClip, WarpMarkers
from labeling.export_als_to_gt import _clip_row, _load_content_catalog
from labeling.als import build_manifest_index


def _mapper():
    class M:  # arr==set seconds
        def arr_to_set_sec(self, a): return a
    return M()


def _clip(path):
    return ParsedClip(group_name="g", track_name="t", path=path,
        arr_start=0.0, arr_end=5.0, loop_start=0.0, loop_end=5.0,
        pitch_coarse=0, pitch_fine=0,
        warp=WarpMarkers(points=((0.0, 0.0), (5.0, 5.0))))


def test_binds_to_own_content_not_slot(tmp_path: Path) -> None:
    # File lives under slot '028' but its bytes belong to recording 'beatles'.
    d = tmp_path / "stems" / "028__X"; d.mkdir(parents=True)
    f = d / "vocals.flac"; f.write_bytes(b"BEATLES-VOCALS" * 100)
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps({"set_id": "s", "tracks": []}))
    (tmp_path / "content_catalog.json").write_text(json.dumps({"set_id": "s",
        "entries": [{"content_sha256": sha, "payload_sha256": None,
                     "recording_id": "beatles", "track_audio_id": "ta", "stem": "acappella"}]}))
    manifest = build_manifest_index(tmp_path / "manifest.json")
    catalog = _load_content_catalog(tmp_path)
    row = _clip_row(_clip(str(f)), _mapper(), manifest, catalog)
    assert row.recording_id == "beatles"
    assert row.id_source == "content"


def test_abstains_when_no_content_match(tmp_path: Path) -> None:
    d = tmp_path / "stems" / "031__Y"; d.mkdir(parents=True)
    f = d / "vocals.flac"; f.write_bytes(b"CCR-VOCALS" * 100)
    (tmp_path / "manifest.json").write_text(json.dumps({"set_id": "s", "tracks": []}))
    (tmp_path / "content_catalog.json").write_text(json.dumps({"set_id": "s", "entries": []}))
    manifest = build_manifest_index(tmp_path / "manifest.json")
    catalog = _load_content_catalog(tmp_path)
    row = _clip_row(_clip(str(f)), _mapper(), manifest, catalog)
    assert row.recording_id is None
    assert row.id_source == "abstain"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/labeling/test_export_content_identity.py -q`
Expected: FAIL — `_clip_row` signature mismatch / `ClipRow` has no `id_source`.

- [ ] **Step 3: Write minimal implementation**

In `labeling/export_als_to_gt.py`:

1. Add `id_source: str = ""` to the `ClipRow` dataclass (after `skip_training`).
2. Replace `_clip_row`'s `slot_id_map` parameter and fallback block:

```python
def _clip_row(
    clip: ParsedClip,
    mapper: ArrangementMapper,
    manifest,
    catalog=None,
) -> ClipRow | None:
    set_start = mapper.arr_to_set_sec(clip.arr_start)
    set_end = mapper.arr_to_set_sec(clip.arr_end)
    if set_start is None or set_end is None:
        return None
    _rid_unused, slot_label, display, claimed_stem = resolve_identity(clip, manifest)
    recording_id, id_source = _content_bind(clip, catalog)
    ...
```

   Remove the `if recording_id is None and slot_id_map and slot_label:` block entirely. Pass `id_source=id_source` into the `ClipRow(...)` constructor.

3. In `collect_kept_clip_rows`: replace `slot_id_map = _load_slot_id_map(set_id)` with `catalog = _load_content_catalog(set_dir)`, and the call `_clip_row(part, mapper, manifest, slot_id_map)` with `_clip_row(part, mapper, manifest, catalog)`.
4. Delete the `_load_slot_id_map` function.
5. In `_to_gt_track`, add `id_source=row.id_source,` to the `GroundTruthTrack(...)` call.

Then delete the fixtures:

```bash
git rm labeling/fixtures/id_maps/1fsnxchk_slots.json labeling/fixtures/id_maps/2nvzlh2k_slots.json
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/labeling/test_export_content_identity.py tests/labeling/ -q`
Expected: PASS. Then verify the poison carrier is gone:

Run: `grep -rn 'slot_id_map' --include='*.py' labeling/ | grep -v attic/`
Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add labeling/export_als_to_gt.py tests/labeling/test_export_content_identity.py
git add -A labeling/fixtures/id_maps/
git commit -m "feat(labeling): content-bind clip identity in export; delete slot_id_map poison"
```

---

## Task 6: C1b renumber-metamorphic gate

**Files:**
- Create: `tests/labeling/test_content_identity_metamorphic.py`

(Note: `tests/test_alignment_metamorphic.py` is the DSP fingerprint probe's metamorphic
suite — the wrong home for a GT-identity property. This co-locates with the content
resolver / export tests it exercises. Divergence from the handoff's literal "extend
test_alignment_metamorphic.py" is intentional.)

**Interfaces:**
- Consumes: `_content_bind`, `_load_content_catalog` (Task 3).

- [ ] **Step 1: Write the failing test**

```python
# tests/labeling/test_content_identity_metamorphic.py
"""C1b: renumbering a clip's slot must NOT change its content-bound identity.

This is the property slot_id_map violated by construction — identity followed the
slot number, so a renumber silently rebound the row to a different song. Content
binding makes identity a function of the audio bytes alone.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from labeling.als.models import ParsedClip, WarpMarkers
from labeling.export_als_to_gt import _content_bind, _load_content_catalog


def _clip(path: str) -> ParsedClip:
    return ParsedClip(group_name="", track_name="", path=path,
        arr_start=0.0, arr_end=1.0, loop_start=0.0, loop_end=1.0,
        pitch_coarse=0, pitch_fine=0,
        warp=WarpMarkers(points=((0.0, 0.0), (1.0, 1.0))))


def test_renumber_preserves_content_identity(tmp_path: Path) -> None:
    payload = b"SAME-AUDIO-BYTES" * 200
    a = tmp_path / "028__Beatles" / "vocals.flac"
    b = tmp_path / "144__Beatles" / "vocals.flac"
    for f in (a, b):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    (tmp_path / "content_catalog.json").write_text(json.dumps({"set_id": "s",
        "entries": [{"content_sha256": sha, "payload_sha256": None,
                     "recording_id": "beatles", "track_audio_id": "ta", "stem": "acappella"}]}))
    cat = _load_content_catalog(tmp_path)
    # Identical bytes under two different slot numbers -> identical identity.
    assert _content_bind(_clip(str(a)), cat) == _content_bind(_clip(str(b)), cat)
    assert _content_bind(_clip(str(a)), cat) == ("beatles", "content")
```

- [ ] **Step 2: Run test to verify it fails, then passes**

Run: `venvs/audio/bin/python -m pytest tests/labeling/test_content_identity_metamorphic.py -q`
Expected: PASS (the mechanism from Tasks 3/5 already satisfies it). If it errors on import, the prior tasks are incomplete — fix there, not here.

- [ ] **Step 3: Commit**

```bash
git add tests/labeling/test_content_identity_metamorphic.py
git commit -m "test(labeling): C1b renumber-metamorphic — content identity is slot-independent"
```

---

## Task 7: Re-base the export coverage gate on `id_source`

**Files:**
- Modify: `labeling/export_als_to_gt.py` (`id_coverage`, the `main()` gate block, `print_review`)
- Test: `tests/labeling/test_id_coverage_gate.py` (create)

**Interfaces:**
- Produces: `id_coverage(tracks) -> tuple[int, int, float]` now counts tracks with `id_source == "content"` (not merely a non-null `track_id`). The refusal message lists abstained slots.

- [ ] **Step 1: Write the failing test**

```python
# tests/labeling/test_id_coverage_gate.py
from __future__ import annotations

from labeling.export_als_to_gt import id_coverage
from labeling.ground_truth.schema import GroundTruthTrack


def _t(id_source: str, rid: str | None) -> GroundTruthTrack:
    return GroundTruthTrack(label="x", track_id=rid, claimed_stem="regular",
        set_start_s=0.0, set_end_s=1.0, ref_start_s=0.0, id_source=id_source)


def test_coverage_counts_content_only() -> None:
    tracks = [_t("content", "r1"), _t("abstain", None), _t("content", "r2")]
    resolved, total, frac = id_coverage(tracks)
    assert (resolved, total) == (2, 3)
    assert abs(frac - 2 / 3) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/labeling/test_id_coverage_gate.py -q`
Expected: FAIL — current `id_coverage` counts `track_id`, so an abstain row with a stale `track_id` would be miscounted (or the count differs).

- [ ] **Step 3: Write minimal implementation**

Change `id_coverage` in `labeling/export_als_to_gt.py`:

```python
def id_coverage(tracks) -> tuple[int, int, float]:
    """(content-bound, total, fraction) of GT tracks whose id came from content."""
    total = len(tracks)
    resolved = sum(1 for t in tracks if getattr(t, "id_source", "") == "content")
    return resolved, total, (resolved / total if total else 1.0)
```

In `main()`'s refusal block, keep the `ID_COVERAGE_MIN` check but update the message to point at content binding, and list abstains:

```python
    resolved, total, coverage = id_coverage(gt.tracks)
    if total and coverage < ID_COVERAGE_MIN and not args.allow_invalid:
        abstained = [t.slot_label or t.label for t in gt.tracks
                     if getattr(t, "id_source", "") != "content"]
        print(
            f"REFUSING to export: only {resolved}/{total} tracks ({coverage:.0%}) "
            f"content-bound (min {ID_COVERAGE_MIN:.0%}). Abstained: "
            f"{', '.join(abstained[:30])}. Rebuild content_catalog.json (re-pull), "
            "or pass --allow-invalid to override.",
            file=sys.stderr,
        )
        return 1
```

Update the `print_review` `unresolved` line to key on the row's binding where a
`ClipRow` is available (leave the ReviewRow-based summary as-is; it already reports
`recording_id` nullness).

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest tests/labeling/test_id_coverage_gate.py tests/labeling/ -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add labeling/export_als_to_gt.py tests/labeling/test_id_coverage_gate.py
git commit -m "feat(labeling): coverage gate counts content-bound ids, lists abstains"
```

---

## Task 8: Emit the sidecar from the pull

**Files:**
- Modify: `labeling/pull_set_for_alignment.py` (after the `manifest.json` write near line 888)
- Test: manual/ops (documented) + a smoke unit if a seam allows.

**Interfaces:**
- Consumes: `build_content_catalog` on pi via SSH (repo deployed via `make deploy`).
- Produces: `<dest_root>/content_catalog.json` alongside `manifest.json`.

- [ ] **Step 1: Add the emit step**

After `(dest_root / "manifest.json").write_text(json.dumps(manifest, indent=2))`, add:

```python
        # Content catalog sidecar (Operation Crush): sha256/mdat -> recording_id
        # for content-addressed GT identity. Built on pi (canonical DB + files).
        try:
            import subprocess
            cat = subprocess.run(
                ["ssh", "pi-storage",
                 "cd ~/tracklist_engine && python3 -m labeling.build_content_catalog "
                 + args.set_id],
                capture_output=True, text=True, timeout=1800, check=True,
            ).stdout
            (dest_root / "content_catalog.json").write_text(cat)
            n = len(json.loads(cat).get("entries", []))
            print(f"wrote content_catalog.json ({n} entries)")
        except (subprocess.SubprocessError, OSError, ValueError) as e:
            print(f"WARNING: content_catalog.json not written: {e}", file=sys.stderr)
```

(Confirm the pi repo path — adjust `~/tracklist_engine` to the deployed checkout root used by `make deploy`.)

- [ ] **Step 2: Verify import + help still work**

Run: `venvs/audio/bin/python -c "import labeling.pull_set_for_alignment"`
Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add labeling/pull_set_for_alignment.py
git commit -m "feat(labeling): pull emits content_catalog.json sidecar (built on pi)"
```

---

## Task 9: Guardrail + `make check`, then push & PR

**Files:**
- Modify: `scripts/guardrails.py` (add a fence) OR rely on the grep in Task 5 — pick one.
- Test: `make check`.

- [ ] **Step 1: Add a stale-name fence** (optional but recommended) — a guardrail assertion that `slot_id_map` and `labeling/fixtures/id_maps/` do not reappear:

```python
# in scripts/guardrails.py, alongside existing stale-name checks:
_forbid("slot_id_map", roots=["labeling"], allow=["attic"],
        why="slot_id_map is the deleted GT id-poison (Operation Crush)")
```

(Match the existing helper's real signature; if `guardrails.py` uses a different
pattern, follow it. If adding a fence is awkward, skip this file and rely on the
Task 5 grep in CI.)

- [ ] **Step 2: Run the full gate**

Run: `make check`
Expected: PASS (guardrails + fast pytest). Fix any fallout in the owning task.

- [ ] **Step 3: Push and open the PR**

```bash
git push -u origin crush/depoison-content-binding
gh auth switch --user jca225   # PRs need the write account (see memory)
gh pr create --title "Operation Crush: content-addressed GT identity (kill slot_id_map)" \
  --body "See docs/superpowers/specs/2026-07-21-content-addressed-gt-identity-design.md"
```

---

## Task 10 (OPS — needs pi + the annotator's folder): re-pull, re-measure, Crush exit

Not a code task; sequence after Tasks 1–9 merge. Coordinate with the user (mutates `~/aligning/1fsnxchk/…`).

- [ ] `make deploy` so `labeling.build_content_catalog` exists on pi.
- [ ] Snapshot first (#51): back up `set_ground_truth` for `1fsnxchk` + the current `labeling/fixtures/bb12_ground_truth.yaml`; confirm tag `wip/bb12-enrichment-backup` exists.
- [ ] Re-pull BB12 to refresh the truncated manifest + emit the sidecar: `/alignment-pull 1fsnxchk` (NO `--prune`; annotator tags are user territory). Confirm `content_catalog.json` written with ~300+ entries.
- [ ] Re-export: `venvs/audio/bin/python -m labeling.export_als_to_gt` (defaults to the BB12 `.als`/set-dir). Verify: slots 028/031/144 bind to the CORRECT recording or `id_source: abstain` — **none carry `2p25k23p`/`1q8nc02p`/`2uq9800f`**; every row carries `id_source`. Record the real abstain rate; if it trips `ID_COVERAGE_MIN`, investigate the abstained files (ad-hoc downloads absent from `track_audio`) before relaxing the gate.
- [ ] Confirm `gt_als_gate` still green (yaml == export(.als)).
- [ ] Repeat for BB11 (`2nvzlh2k`).
- [ ] `/align-checkpoint` to regenerate `docs/alignment_status.md` on the de-poisoned GT — the first honest post-Crush numbers. **This is Crush exit.**

---

## Self-review notes

- **Spec coverage:** A→content_hash+catalog build (T1,T4,T8); B→loader+bind+wiring (T3,T5); C→id_source (T2); D→delete slot_id_map (T5) + guardrail (T9); E→gate (T7); F→metamorphic (T6); ops re-measure (T10). Step 3 (audio round-trip) is explicitly out of scope per the spec.
- **Deferred/there-is-judgment:** `match_manifest_for_path` weak-tier trimming (spec Component D) is **intentionally NOT a hard task** — after Task 5 identity no longer flows through it (the exporter ignores `resolve_identity`'s `recording_id`), so the poison is already dead; trimming the tiers would break `tests/labeling/test_als_io.py` stem-folder assertions and `enrich_gt_track_ids`/`als_path_audit` behavior for zero identity gain. If a reviewer insists, do it as a follow-up with those tests updated.
- **Type consistency:** `_content_bind -> (recording_id, id_source)` used identically in T3/T5/T6; `id_coverage` counts `id_source=="content"` in T7 matching the field added in T2; catalog entry keys (`content_sha256`, `payload_sha256`, `recording_id`, `track_audio_id`, `stem`) identical across T4 (producer) and T3 (consumer).
