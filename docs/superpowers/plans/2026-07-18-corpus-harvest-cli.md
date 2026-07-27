# Corpus-harvest CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a thin batch CLI that runs the co-training flywheel's write-side over the pi-storage corpus — pi DB queries → build cases (cue-time anchored) → certified-probe scorer → harvest ledger — plus a `--census` mode that quantifies eligibility before the GPU stem pass.

**Architecture:** One new module `pws_aligner/corpus_harvest.py` of small pure functions with a thin `main()`. It adds only a **corpus case-builder** (DB → cases) and a **batch loop**; all alignment logic delegates to existing, tested machinery (`harvest.harvest` / `harvest.write_ledger` / `cotrain_seam.real_probe_scorer` / `cotrain_seam.corpus_mix_resolver`). Zero canonical-DB mutation (inherits the seam invariant). CPU-only for the certified axes (regular/instrumental skip HuBERT), so it runs on pi-storage.

**Tech Stack:** Python 3, stdlib `sqlite3` + `argparse` + `dataclasses`, pytest. Repo "Rust-flavoured functional" style: `from __future__ import annotations`, frozen dataclasses, full type hints, pure core + fail-fast edge.

## Global Constraints

- Module + tests + spec live on branch `cotrain-corpus-harvest` (already created, stacked off `worktree-cotrain-accept-precision`). Work in the worktree at `/Users/johnnycabrahams/Desktop/tracklist_engine/.claude/worktrees/cotrain-accept-precision`.
- Run tests with the repo venv from repo root: `venvs/audio/bin/python -m pytest pws_aligner/tests/test_corpus_harvest.py -v`. Imports are absolute from repo root (e.g. `pws_aligner.corpus_harvest`).
- **Only the certified axes are harvestable.** Reuse `harvest.CERTIFIED_POLICY` verbatim (`regular` @ default 2-channel band, `instrumental` @ `min_agreeing=3`). Never add `acappella` — it must produce no cases and no ledger rows.
- **Positive-only** cases (no decoys — decoys were only for the precision *gate*).
- **Zero canonical mutation.** The CLI writes ONLY the harvest-ledger JSONL (via `harvest.write_ledger`, idempotent by `span_key`). It never writes the DB or the correction ledger.
- **Disk is truth** for audio existence (RoFormer writes `vocals.flac`/`instrumental.flac`; the `set_stems` table is not consulted). The census disk-checks.
- Canonical path defaults: `DEFAULT_DB = /mnt/storage/data/db/music_database.db`, `DEFAULT_STEMS_ROOT = /mnt/storage/stems/set`, `DEFAULT_SPAN_S = 40.0`.
- Commit each task with `--no-verify` **only if** the pre-commit hook fails solely on `mypy` not installed (known env gap; guardrails must still print `OK`). Otherwise commit normally.
- Spec: `docs/superpowers/specs/2026-07-18-corpus-harvest-cli-design.md`.

---

## File Structure

- **Create** `pws_aligner/corpus_harvest.py` — the whole feature: `CorpusSlot`, `query_corpus_slots`, `build_corpus_cases`, `HarvestSummary`, `run_corpus_harvest`, `CensusReport`, `census_rows`, `census`, `main`, and helpers `_resolve` / `_default_scorer_factory`.
- **Create** `pws_aligner/tests/test_corpus_harvest.py` — fixture-DB + fake-scorer tests. No pi access, no real audio.

Reference shapes (from the existing code — do not redefine, import them):
- `RefCandidate(recording_id, source_url, source_path=None, display_name="", version="original", stem="regular", variant="regular", track_audio_id=None)` — `cotrain_seam`.
- `MixSpan(set_id, slot_label, set_start_s, span_dur_s)` — `cotrain_seam`.
- `BandThresholds(min_probes=2, min_agreeing=2, accept_tol_s=1.0, accept_conf=0.55, review_tol_s=3.0, review_conf=0.70)` — `cotrain_seam`.
- `AlignmentResult(recording_id, offset_s, ref_end_s=None, segments=(), tempo_ratio=None, confidence=0.0, abstain=False, source="")` — `alignment.harness.contract`. **`confidence` must be in `[0,1]`** (enforced in `__post_init__`).
- `harvest(cases, scorer, *, policy=CERTIFIED_POLICY) -> list[HarvestRecord]` where `cases: Iterable[tuple[RefCandidate, MixSpan, dict|None]]` and `scorer: Callable[[RefCandidate, MixSpan], Sequence[AlignmentResult]]` — `harvest`.
- `write_ledger(records, ledger: Path) -> int` (idempotent append, returns n written) — `harvest`.
- `corpus_mix_resolver(mix_full_path: Path, mix_stem_dir: Path) -> (stem:str)->Path|None` and `real_probe_scorer(*, mix_resolver=...) -> RefMixScorer` — `cotrain_seam`.

---

## Task 1: `CorpusSlot` + `query_corpus_slots` (pi DB → eligible slots)

**Files:**
- Create: `pws_aligner/corpus_harvest.py`
- Test: `pws_aligner/tests/test_corpus_harvest.py`

**Interfaces:**
- Consumes: nothing (entry).
- Produces:
  - `@dataclass(frozen=True) class CorpusSlot` with fields `set_id: str, set_audio_id: int, slot_label: str, recording_id: str, ref_path: str, claimed_version: str, claimed_stem: str, claimed_variant: str, cue_time_s: float, duration_s: float | None, mix_full_path: str`.
  - `query_corpus_slots(conn: sqlite3.Connection, *, policy_stems: Iterable[str], limit: int | None = None) -> list[CorpusSlot]` — strict inner-join eligibility; requires `conn.row_factory = sqlite3.Row`.
  - Module constants `DEFAULT_DB: Path`, `DEFAULT_STEMS_ROOT: Path`, `DEFAULT_SPAN_S: float`.

- [ ] **Step 1: Write the failing test**

Create `pws_aligner/tests/test_corpus_harvest.py`:

```python
"""Corpus-harvest CLI: DB→cases→ledger glue + eligibility census.

Tests the thin flywheel-step-2 runner against an in-memory fixture DB and a
FAKE scorer — no pi access, no real audio. All alignment logic is delegated to
the already-tested harvest/seam machinery; here we test the corpus glue: which
slots are eligible, how cases are built, that banding gates per axis, that the
ledger is idempotent, and that the census classifies blockers correctly.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from pws_aligner.corpus_harvest import query_corpus_slots

# NOTE: imports are added incrementally per task, so each task's test file
# collects with only the names defined so far. Do NOT import later-task names
# here (no stubs) — Task 2 adds CorpusSlot/DEFAULT_SPAN_S/build_corpus_cases,
# Task 3 adds AlignmentResult/run_corpus_harvest, Task 4 adds census/census_rows,
# Task 5 adds main.


def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE set_track_slots (
            set_id TEXT, row_index INTEGER, recording_id TEXT, slot_label TEXT,
            cue_seconds INTEGER, cue_time_seconds INTEGER,
            claimed_version TEXT, claimed_stem TEXT, claimed_variant TEXT,
            duration_seconds INTEGER
        );
        CREATE TABLE set_audio (
            set_audio_id INTEGER, set_id TEXT, path TEXT, sha256 TEXT,
            is_reference INTEGER
        );
        CREATE TABLE track_audio (
            track_audio_id INTEGER, recording_id TEXT, stem TEXT,
            path TEXT, is_reference INTEGER
        );
        """
    )
    return conn


def _add_slot(
    conn, *, set_id, row_index, recording_id, slot_label="001",
    cue_time_seconds=100, claimed_stem="regular", duration_seconds=42,
    claimed_version="original", claimed_variant="regular",
):
    conn.execute(
        "INSERT INTO set_track_slots VALUES (?,?,?,?,?,?,?,?,?,?)",
        (set_id, row_index, recording_id, slot_label, None, cue_time_seconds,
         claimed_version, claimed_stem, claimed_variant, duration_seconds),
    )


def _add_set_audio(conn, *, set_audio_id, set_id, path, is_reference=1):
    conn.execute(
        "INSERT INTO set_audio VALUES (?,?,?,?,?)",
        (set_audio_id, set_id, path, None, is_reference),  # sha256=None
    )


def _add_track_audio(conn, *, recording_id, stem="regular", path="/ref.flac", is_reference=1):
    conn.execute(
        "INSERT INTO track_audio VALUES (?,?,?,?,?)",
        (None, recording_id, stem, path, is_reference),
    )


def test_query_returns_only_eligible_slots():
    conn = _make_db()
    # eligible regular slot
    _add_slot(conn, set_id="S1", row_index=0, recording_id="R1")
    _add_set_audio(conn, set_audio_id=10, set_id="S1", path="/mix1.m4a")
    _add_track_audio(conn, recording_id="R1", stem="regular", path="/r1.flac")
    # ineligible: acappella (uncertified axis, excluded by policy_stems)
    _add_slot(conn, set_id="S1", row_index=1, recording_id="R2", slot_label="002",
              claimed_stem="acappella")
    _add_track_audio(conn, recording_id="R2", stem="acappella", path="/r2.flac")
    # ineligible: no cue time
    _add_slot(conn, set_id="S1", row_index=2, recording_id="R3", slot_label="003",
              cue_time_seconds=None)
    _add_track_audio(conn, recording_id="R3", stem="regular", path="/r3.flac")
    # ineligible: ref audio wrong stem (claimed regular, only acappella ref exists)
    _add_slot(conn, set_id="S1", row_index=3, recording_id="R4", slot_label="004")
    _add_track_audio(conn, recording_id="R4", stem="acappella", path="/r4.flac")

    slots = query_corpus_slots(conn, policy_stems=("regular", "instrumental"))

    assert [s.recording_id for s in slots] == ["R1"]
    s = slots[0]
    assert s.set_audio_id == 10
    assert s.cue_time_s == 100.0
    assert s.duration_s == 42.0
    assert s.ref_path == "/r1.flac"
    assert s.mix_full_path == "/mix1.m4a"


def test_query_excludes_non_reference_mix_and_ref():
    conn = _make_db()
    _add_slot(conn, set_id="S2", row_index=0, recording_id="R1")
    _add_set_audio(conn, set_audio_id=20, set_id="S2", path="/mix.m4a", is_reference=0)
    _add_track_audio(conn, recording_id="R1", stem="regular", is_reference=1)
    _add_slot(conn, set_id="S2", row_index=1, recording_id="R2", slot_label="002")
    _add_set_audio(conn, set_audio_id=21, set_id="S2", path="/mix.m4a", is_reference=1)
    _add_track_audio(conn, recording_id="R2", stem="regular", is_reference=0)

    slots = query_corpus_slots(conn, policy_stems=("regular", "instrumental"))
    # S2 has a reference mix (id 21) but R1's row_index-0 slot: R1 ref is_reference=1
    # yet its set's reference mix is id 21 → R1 IS eligible; R2 ref is non-reference → excluded.
    assert [s.recording_id for s in slots] == ["R1"]
    assert slots[0].set_audio_id == 21


def test_query_respects_limit_and_order():
    conn = _make_db()
    _add_set_audio(conn, set_audio_id=30, set_id="S3", path="/m.m4a")
    for i in range(3):
        _add_slot(conn, set_id="S3", row_index=i, recording_id=f"R{i}", slot_label=f"00{i}")
        _add_track_audio(conn, recording_id=f"R{i}", stem="regular")
    slots = query_corpus_slots(conn, policy_stems=("regular",), limit=2)
    assert [s.recording_id for s in slots] == ["R0", "R1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest pws_aligner/tests/test_corpus_harvest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pws_aligner.corpus_harvest'` (or ImportError on the names).

- [ ] **Step 3: Write minimal implementation**

Create `pws_aligner/corpus_harvest.py` with the header + Task-1 pieces:

```python
"""Corpus-harvest CLI — co-training flywheel step 2 (batch runner over the corpus).

Turns already-downloaded+analyzed pi-storage corpus sets into a harvest-ledger of
pseudo-labelled (ref ↔ mix-span) training pairs. It is GLUE: a corpus case-builder
(pi DB → cases) + a batch loop, delegating all alignment logic to the tested
machinery (``harvest.harvest`` / ``harvest.write_ledger`` /
``cotrain_seam.real_probe_scorer`` / ``cotrain_seam.corpus_mix_resolver``).

Placement anchor = the scraped 1001TL cue time
(``set_track_slots.cue_time_seconds`` / ``cue_seconds``) — the corpus has no GT
window. A noisy window costs RECALL not PRECISION: bad window → certified probes
disagree → ABSTAIN → not harvested, so the 2026-07-18 ACCEPT-precision
certification (regular @2-channel, instrumental @3-channel-unanimity) transfers.
Only the certified axes are harvestable (CERTIFIED_POLICY); acappella is never
harvested and needs no HuBERT, so this runs CPU-only on pi-storage.

Invariant (inherited from cotrain_seam/harvest): ZERO canonical mutation — writes
ONLY the harvest-ledger JSONL, idempotent by span_key.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from pws_aligner.cotrain_seam import (
    BandThresholds,
    MixSpan,
    RefCandidate,
    RefMixScorer,
    corpus_mix_resolver,
    real_probe_scorer,
)
from pws_aligner.harvest import (
    CERTIFIED_POLICY,
    harvest,
    write_ledger,
)

# Canonical pi-storage defaults (all overridable via CLI args for tests/other hosts).
DEFAULT_DB = Path("/mnt/storage/data/db/music_database.db")
DEFAULT_STEMS_ROOT = Path("/mnt/storage/stems/set")
DEFAULT_SPAN_S = 40.0


@dataclass(frozen=True)
class CorpusSlot:
    """A DB-projected harvest-eligible slot (decouples SQL from case-building)."""

    set_id: str
    set_audio_id: int
    slot_label: str
    recording_id: str
    ref_path: str
    claimed_version: str
    claimed_stem: str
    claimed_variant: str
    cue_time_s: float
    duration_s: float | None
    mix_full_path: str


def query_corpus_slots(
    conn: sqlite3.Connection,
    *,
    policy_stems: Iterable[str],
    limit: int | None = None,
) -> list[CorpusSlot]:
    """Strict inner-join eligibility: reference mix + reference ref @ claimed_stem
    + a scraped cue time + a certified axis. ``conn.row_factory`` must be
    ``sqlite3.Row``. On-disk audio existence is NOT checked here (disk is truth;
    the scorer abstains when absent) — see ``census`` for the disk funnel.
    """
    stems = tuple(policy_stems)
    if not stems:
        return []
    placeholders = ",".join("?" for _ in stems)
    sql = f"""
        SELECT s.set_id AS set_id, sa.set_audio_id AS set_audio_id,
               s.slot_label AS slot_label, s.recording_id AS recording_id,
               ta.path AS ref_path, s.claimed_version AS claimed_version,
               s.claimed_stem AS claimed_stem, s.claimed_variant AS claimed_variant,
               COALESCE(s.cue_time_seconds, s.cue_seconds) AS cue_time_s,
               s.duration_seconds AS duration_s, sa.path AS mix_full_path
        FROM set_track_slots s
        JOIN set_audio sa ON sa.set_id = s.set_id AND sa.is_reference = 1
        JOIN track_audio ta ON ta.recording_id = s.recording_id
                            AND ta.stem = s.claimed_stem
                            AND ta.is_reference = 1
        WHERE s.claimed_stem IN ({placeholders})
          AND s.recording_id IS NOT NULL
          AND COALESCE(s.cue_time_seconds, s.cue_seconds) IS NOT NULL
        ORDER BY s.set_id, s.row_index
    """
    params: list[object] = list(stems)
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    out: list[CorpusSlot] = []
    for r in conn.execute(sql, params).fetchall():
        out.append(
            CorpusSlot(
                set_id=r["set_id"],
                set_audio_id=int(r["set_audio_id"]),
                slot_label=r["slot_label"] or "",
                recording_id=r["recording_id"],
                ref_path=r["ref_path"],
                claimed_version=r["claimed_version"] or "original",
                claimed_stem=r["claimed_stem"],
                claimed_variant=r["claimed_variant"] or "regular",
                cue_time_s=float(r["cue_time_s"]),
                duration_s=float(r["duration_s"]) if r["duration_s"] is not None else None,
                mix_full_path=r["mix_full_path"],
            )
        )
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest pws_aligner/tests/test_corpus_harvest.py -v`
Expected: PASS — 3 passed (the file contains only the `test_query_*` functions at this task; later tasks append more). Do NOT add stubs for later-task names — imports are incremental per task.

- [ ] **Step 5: Commit**

```bash
cd /Users/johnnycabrahams/Desktop/tracklist_engine/.claude/worktrees/cotrain-accept-precision
git add pws_aligner/corpus_harvest.py pws_aligner/tests/test_corpus_harvest.py
git commit -m "feat(cotrain): corpus_harvest CorpusSlot + query_corpus_slots (DB eligibility)"
```

---

## Task 2: `build_corpus_cases` (slots → positive-only cases)

**Files:**
- Modify: `pws_aligner/corpus_harvest.py`
- Test: `pws_aligner/tests/test_corpus_harvest.py`

**Interfaces:**
- Consumes: `CorpusSlot` (Task 1); `RefCandidate`, `MixSpan` (cotrain_seam).
- Produces:
  - `_resolve(path: str, root: Path | None) -> Path` — join `root` only when `path` is relative.
  - `build_corpus_cases(slots: Sequence[CorpusSlot], *, ref_audio_root: Path | None = None) -> list[tuple[RefCandidate, MixSpan, dict[str, str]]]`.

- [ ] **Step 1: Write the failing test**

Append to `test_corpus_harvest.py` (the import goes with the other imports near the top of the file; the rest is appended at the end):

```python
from pws_aligner.corpus_harvest import (  # noqa: E402
    DEFAULT_SPAN_S,
    CorpusSlot,
    build_corpus_cases,
)


def _slot(**kw) -> CorpusSlot:
    base = dict(
        set_id="S1", set_audio_id=10, slot_label="001", recording_id="R1",
        ref_path="/ref/r1.flac", claimed_version="original", claimed_stem="regular",
        claimed_variant="regular", cue_time_s=120.0, duration_s=55.0,
        mix_full_path="/mix/s1.m4a",
    )
    base.update(kw)
    return CorpusSlot(**base)


def test_build_cases_maps_slot_to_candidate_span_axes():
    cases = build_corpus_cases([_slot()])
    assert len(cases) == 1
    cand, span, axes = cases[0]
    assert cand.recording_id == "R1"
    assert cand.source_path == "/ref/r1.flac"
    assert cand.stem == "regular"
    assert cand.version == "original"
    assert cand.variant == "regular"
    assert cand.source_url.startswith("corpus://")
    assert span.set_id == "S1"
    assert span.slot_label == "001"
    assert span.set_start_s == 120.0
    assert span.span_dur_s == 55.0
    assert axes == {"version": "original", "stem": "regular", "variant": "regular"}


def test_build_cases_defaults_span_when_no_duration():
    cases = build_corpus_cases([_slot(duration_s=None)])
    _, span, _ = cases[0]
    assert span.span_dur_s == DEFAULT_SPAN_S


def test_build_cases_applies_ref_audio_root_to_relative_paths():
    cases = build_corpus_cases([_slot(ref_path="rel/r1.flac")], ref_audio_root=Path("/root"))
    cand, _, _ = cases[0]
    assert cand.source_path == "/root/rel/r1.flac"
    # absolute paths untouched
    cases2 = build_corpus_cases([_slot(ref_path="/abs/r1.flac")], ref_audio_root=Path("/root"))
    assert cases2[0][0].source_path == "/abs/r1.flac"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest pws_aligner/tests/test_corpus_harvest.py -k build_cases -v`
Expected: FAIL — `ImportError: cannot import name 'build_corpus_cases'`.

- [ ] **Step 3: Write minimal implementation**

Append to `corpus_harvest.py` (after `query_corpus_slots`):

```python
def _resolve(path: str, root: Path | None) -> Path:
    """Join ``root`` to ``path`` only when ``path`` is relative; else pass through."""
    p = Path(path)
    if root is not None and not p.is_absolute():
        return Path(root) / p
    return p


def build_corpus_cases(
    slots: Sequence[CorpusSlot],
    *,
    ref_audio_root: Path | None = None,
) -> list[tuple[RefCandidate, MixSpan, dict[str, str]]]:
    """One positive case per slot (no decoys — harvesting keeps confident
    agreements; decoys were only for the precision gate). ``claim_axes`` mirrors
    the slot claim so ``cotrain_seam`` can propose a correction if the accepted
    candidate ever differs (here they match by construction → correction None).
    """
    cases: list[tuple[RefCandidate, MixSpan, dict[str, str]]] = []
    for s in slots:
        candidate = RefCandidate(
            recording_id=s.recording_id,
            source_url=f"corpus://{s.set_id}/{s.slot_label}",
            source_path=str(_resolve(s.ref_path, ref_audio_root)),
            version=s.claimed_version,
            stem=s.claimed_stem,
            variant=s.claimed_variant,
        )
        span = MixSpan(
            set_id=s.set_id,
            slot_label=s.slot_label,
            set_start_s=s.cue_time_s,
            span_dur_s=s.duration_s if s.duration_s else DEFAULT_SPAN_S,
        )
        claim_axes = {
            "version": s.claimed_version,
            "stem": s.claimed_stem,
            "variant": s.claimed_variant,
        }
        cases.append((candidate, span, claim_axes))
    return cases
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest pws_aligner/tests/test_corpus_harvest.py -k build_cases -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add pws_aligner/corpus_harvest.py pws_aligner/tests/test_corpus_harvest.py
git commit -m "feat(cotrain): build_corpus_cases (cue-anchored positive-only cases)"
```

---

## Task 3: `run_corpus_harvest` (batch loop, per-set scorer, idempotent ledger)

**Files:**
- Modify: `pws_aligner/corpus_harvest.py`
- Test: `pws_aligner/tests/test_corpus_harvest.py`

**Interfaces:**
- Consumes: `CorpusSlot`, `build_corpus_cases`, `_resolve` (Tasks 1–2); `harvest`, `write_ledger`, `CERTIFIED_POLICY`, `corpus_mix_resolver`, `real_probe_scorer`, `BandThresholds`, `RefMixScorer`.
- Produces:
  - `@dataclass(frozen=True) class HarvestSummary` with `n_sets: int, n_cases: int, n_harvested: int, n_written: int` and `to_json() -> dict`.
  - `ScorerFactory = Callable[[Path, Path], RefMixScorer]` (args: `mix_full_path`, `mix_stem_dir`).
  - `_default_scorer_factory(mix_full_path: Path, mix_stem_dir: Path) -> RefMixScorer`.
  - `run_corpus_harvest(slots, *, stems_root: Path, out: Path, policy: dict[str, BandThresholds] = CERTIFIED_POLICY, set_audio_root: Path | None = None, ref_audio_root: Path | None = None, scorer_factory: ScorerFactory = _default_scorer_factory) -> HarvestSummary`.

- [ ] **Step 1: Write the failing test**

Append to `test_corpus_harvest.py` (imports with the others near the top; helpers/tests at the end):

```python
from alignment.harness.contract import AlignmentResult  # noqa: E402
from pws_aligner.corpus_harvest import run_corpus_harvest  # noqa: E402


def _agree(rec_id, sources, *, offset=12.0, conf=0.8):
    return [
        AlignmentResult(recording_id=rec_id, offset_s=offset, confidence=conf, source=src)
        for src in sources
    ]


def test_run_harvest_writes_only_accepts(tmp_path):
    # two regular slots in one set; scorer agrees (2 channels) → both ACCEPT.
    slots = [
        _slot(recording_id="R1", slot_label="001", cue_time_s=100.0),
        _slot(recording_id="R2", slot_label="002", cue_time_s=200.0),
    ]

    def factory(mix_full_path, mix_stem_dir):
        def scorer(cand, span):
            return _agree(cand.recording_id, ("fp", "chroma"))
        return scorer

    out = tmp_path / "ledger.jsonl"
    summary = run_corpus_harvest(
        slots, stems_root=tmp_path, out=out, scorer_factory=factory
    )
    assert summary.n_sets == 1
    assert summary.n_cases == 2
    assert summary.n_harvested == 2
    assert summary.n_written == 2
    lines = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    assert {r["recording_id"] for r in lines} == {"R1", "R2"}
    assert all(r["stem"] == "regular" for r in lines)


def test_run_harvest_instrumental_needs_three_channels(tmp_path):
    slots = [_slot(recording_id="RI", slot_label="003", claimed_stem="instrumental")]

    def two_channel(mix_full_path, mix_stem_dir):
        def scorer(cand, span):
            return _agree(cand.recording_id, ("fp", "chroma"))  # only 2 agree
        return scorer

    out = tmp_path / "ledger.jsonl"
    summary = run_corpus_harvest(slots, stems_root=tmp_path, out=out, scorer_factory=two_channel)
    assert summary.n_harvested == 0  # instrumental banded < ACCEPT at 2 channels
    assert not out.exists() or out.read_text().strip() == ""

    def three_channel(mix_full_path, mix_stem_dir):
        def scorer(cand, span):
            return _agree(cand.recording_id, ("fp", "chroma", "continuity"))
        return scorer

    out2 = tmp_path / "ledger2.jsonl"
    summary2 = run_corpus_harvest(slots, stems_root=tmp_path, out=out2, scorer_factory=three_channel)
    assert summary2.n_harvested == 1


def test_run_harvest_is_idempotent(tmp_path):
    slots = [_slot(recording_id="R1", slot_label="001", cue_time_s=100.0)]

    def factory(mix_full_path, mix_stem_dir):
        return lambda cand, span: _agree(cand.recording_id, ("fp", "chroma"))

    out = tmp_path / "ledger.jsonl"
    first = run_corpus_harvest(slots, stems_root=tmp_path, out=out, scorer_factory=factory)
    second = run_corpus_harvest(slots, stems_root=tmp_path, out=out, scorer_factory=factory)
    assert first.n_written == 1
    assert second.n_written == 0  # span_key already present
    assert len([x for x in out.read_text().splitlines() if x.strip()]) == 1


def test_run_harvest_builds_one_scorer_per_set(tmp_path):
    # two slots share set_audio_id → factory called once; stem dir routed by id.
    slots = [
        _slot(set_id="S1", set_audio_id=77, recording_id="R1", slot_label="001", cue_time_s=100.0),
        _slot(set_id="S1", set_audio_id=77, recording_id="R2", slot_label="002", cue_time_s=200.0),
    ]
    calls = []

    def factory(mix_full_path, mix_stem_dir):
        calls.append((Path(mix_full_path), Path(mix_stem_dir)))
        return lambda cand, span: _agree(cand.recording_id, ("fp", "chroma"))

    out = tmp_path / "ledger.jsonl"
    run_corpus_harvest(slots, stems_root=tmp_path / "stems", out=out, scorer_factory=factory)
    assert len(calls) == 1
    assert calls[0][1] == tmp_path / "stems" / "77"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest pws_aligner/tests/test_corpus_harvest.py -k run_harvest -v`
Expected: FAIL — `ImportError: cannot import name 'run_corpus_harvest'`.

- [ ] **Step 3: Write minimal implementation**

Append to `corpus_harvest.py` (after `build_corpus_cases`):

```python
@dataclass(frozen=True)
class HarvestSummary:
    """Counts from one batch harvest run."""

    n_sets: int
    n_cases: int
    n_harvested: int
    n_written: int

    def to_json(self) -> dict:
        return {
            "n_sets": self.n_sets,
            "n_cases": self.n_cases,
            "n_harvested": self.n_harvested,
            "n_written": self.n_written,
        }


# A ScorerFactory builds a per-set scorer from (mix_full_path, mix_stem_dir).
# Injected so the batch loop is testable with a fake scorer offline.
ScorerFactory = Callable[[Path, Path], RefMixScorer]


def _default_scorer_factory(mix_full_path: Path, mix_stem_dir: Path) -> RefMixScorer:
    """Real corpus scorer: certified probes over the pi-storage layout."""
    return real_probe_scorer(
        mix_resolver=corpus_mix_resolver(mix_full_path, mix_stem_dir)
    )


def run_corpus_harvest(
    slots: Sequence[CorpusSlot],
    *,
    stems_root: Path,
    out: Path,
    policy: dict[str, BandThresholds] = CERTIFIED_POLICY,
    set_audio_root: Path | None = None,
    ref_audio_root: Path | None = None,
    scorer_factory: ScorerFactory = _default_scorer_factory,
) -> HarvestSummary:
    """Group slots by ``set_audio_id``, build ONE scorer per set (so the mix
    feature cache is reused across the set's slots), harvest under ``policy``, and
    append incrementally to the idempotent ledger (crash-safe + resumable).
    """
    stems_root = Path(stems_root)
    by_set: dict[int, list[CorpusSlot]] = {}
    for s in slots:
        by_set.setdefault(s.set_audio_id, []).append(s)

    n_cases = n_harvested = n_written = 0
    for set_audio_id, set_slots in by_set.items():
        mix_full = _resolve(set_slots[0].mix_full_path, set_audio_root)
        mix_stem_dir = stems_root / str(set_audio_id)
        scorer = scorer_factory(mix_full, mix_stem_dir)
        cases = build_corpus_cases(set_slots, ref_audio_root=ref_audio_root)
        n_cases += len(cases)
        records = harvest(cases, scorer, policy=policy)
        n_harvested += len(records)
        n_written += write_ledger(records, out)
    return HarvestSummary(len(by_set), n_cases, n_harvested, n_written)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest pws_aligner/tests/test_corpus_harvest.py -k run_harvest -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add pws_aligner/corpus_harvest.py pws_aligner/tests/test_corpus_harvest.py
git commit -m "feat(cotrain): run_corpus_harvest batch loop (per-set scorer, idempotent ledger)"
```

---

## Task 4: `census` (eligibility funnel, disk-checked)

**Files:**
- Modify: `pws_aligner/corpus_harvest.py`
- Test: `pws_aligner/tests/test_corpus_harvest.py`

**Interfaces:**
- Consumes: `_resolve` (Task 2); `sqlite3.Row` connection.
- Produces:
  - `_CENSUS_CATEGORIES: tuple[str, ...] = ("eligible-now", "no-cue-time", "no-ref-audio", "no-mix-audio", "no-mix-stem")`.
  - `@dataclass(frozen=True) class CensusReport` with `by_axis: dict[str, dict[str, int]]`, methods `total() -> int`, `to_json() -> dict`, `render() -> str`.
  - `census_rows(conn, *, policy_stems: Iterable[str]) -> list[sqlite3.Row]` (LEFT-join funnel).
  - `_classify(row, *, stems_root: Path, set_audio_root: Path | None) -> str`.
  - `census(conn, *, stems_root: Path, policy_stems: Iterable[str], set_audio_root: Path | None = None) -> CensusReport`.

- [ ] **Step 1: Write the failing test**

Append to `test_corpus_harvest.py` (imports with the others near the top; tests at the end):

```python
from pws_aligner.corpus_harvest import census, census_rows  # noqa: E402


def test_census_classifies_blockers_per_axis(tmp_path):
    conn = _make_db()
    stems_root = tmp_path / "stems"
    # 1) eligible-now regular: mix file present, ref present, cue present
    mix1 = tmp_path / "mix1.m4a"
    mix1.write_bytes(b"x")
    _add_slot(conn, set_id="A", row_index=0, recording_id="R1", claimed_stem="regular")
    _add_set_audio(conn, set_audio_id=1, set_id="A", path=str(mix1))
    _add_track_audio(conn, recording_id="R1", stem="regular")
    # 2) no-cue-time regular
    _add_slot(conn, set_id="A", row_index=1, recording_id="R2", slot_label="002",
              claimed_stem="regular", cue_time_seconds=None)
    _add_track_audio(conn, recording_id="R2", stem="regular")
    # 3) no-ref-audio regular (no track_audio row)
    _add_slot(conn, set_id="A", row_index=2, recording_id="R3", slot_label="003",
              claimed_stem="regular")
    # 4) no-mix-audio regular (set B has no set_audio row)
    _add_slot(conn, set_id="B", row_index=0, recording_id="R4", claimed_stem="regular")
    _add_track_audio(conn, recording_id="R4", stem="regular")
    # 5) instrumental, mix present but no instrumental.flac on disk → no-mix-stem
    mix2 = tmp_path / "mix2.m4a"
    mix2.write_bytes(b"x")
    _add_slot(conn, set_id="C", row_index=0, recording_id="R5", claimed_stem="instrumental")
    _add_set_audio(conn, set_audio_id=5, set_id="C", path=str(mix2))
    _add_track_audio(conn, recording_id="R5", stem="instrumental")
    # 6) instrumental eligible-now: instrumental.flac present
    mix3 = tmp_path / "mix3.m4a"
    mix3.write_bytes(b"x")
    (stems_root / "6").mkdir(parents=True)
    (stems_root / "6" / "instrumental.flac").write_bytes(b"x")
    _add_slot(conn, set_id="D", row_index=0, recording_id="R6", claimed_stem="instrumental")
    _add_set_audio(conn, set_audio_id=6, set_id="D", path=str(mix3))
    _add_track_audio(conn, recording_id="R6", stem="instrumental")
    # excluded entirely: acappella (uncertified) — must not appear in any axis bucket
    _add_slot(conn, set_id="D", row_index=1, recording_id="R7", slot_label="002",
              claimed_stem="acappella")
    _add_track_audio(conn, recording_id="R7", stem="acappella")

    report = census(conn, stems_root=stems_root, policy_stems=("regular", "instrumental"))

    assert set(report.by_axis) == {"regular", "instrumental"}
    assert report.by_axis["regular"]["eligible-now"] == 1
    assert report.by_axis["regular"]["no-cue-time"] == 1
    assert report.by_axis["regular"]["no-ref-audio"] == 1
    assert report.by_axis["regular"]["no-mix-audio"] == 1
    assert report.by_axis["instrumental"]["no-mix-stem"] == 1
    assert report.by_axis["instrumental"]["eligible-now"] == 1
    assert report.total() == 6
    # to_json + render are well-formed
    assert report.to_json()["total"] == 6
    assert "eligible-now" in report.render()


def test_census_rows_excludes_uncertified_axis(tmp_path):
    conn = _make_db()
    _add_slot(conn, set_id="A", row_index=0, recording_id="R1", claimed_stem="acappella")
    rows = census_rows(conn, policy_stems=("regular", "instrumental"))
    assert rows == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest pws_aligner/tests/test_corpus_harvest.py -k census -v`
Expected: FAIL — `ImportError: cannot import name 'census'` / `census_rows`.

- [ ] **Step 3: Write minimal implementation**

Append to `corpus_harvest.py` (after `run_corpus_harvest`):

```python
_CENSUS_CATEGORIES: tuple[str, ...] = (
    "eligible-now",
    "no-cue-time",
    "no-ref-audio",
    "no-mix-audio",
    "no-mix-stem",
)


@dataclass(frozen=True)
class CensusReport:
    """Per-axis eligibility funnel — the flywheel's recall ceiling today."""

    by_axis: dict[str, dict[str, int]]

    def total(self) -> int:
        return sum(sum(cats.values()) for cats in self.by_axis.values())

    def to_json(self) -> dict:
        return {"by_axis": self.by_axis, "total": self.total()}

    def render(self) -> str:
        lines = ["=== corpus-harvest eligibility census ==="]
        for axis in sorted(self.by_axis):
            cats = self.by_axis[axis]
            lines.append(f"[{axis}] total={sum(cats.values())}")
            for cat in _CENSUS_CATEGORIES:
                lines.append(f"    {cat:<14} {cats.get(cat, 0)}")
        lines.append(f"TOTAL slots (certified axes): {self.total()}")
        return "\n".join(lines)


def census_rows(
    conn: sqlite3.Connection, *, policy_stems: Iterable[str]
) -> list[sqlite3.Row]:
    """LEFT-join funnel over certified-axis slots: every slot with its DB pieces
    (cue / ref audio / reference mix), so the classifier can name what blocks it.
    """
    stems = tuple(policy_stems)
    if not stems:
        return []
    placeholders = ",".join("?" for _ in stems)
    sql = f"""
        SELECT s.set_id AS set_id, s.slot_label AS slot_label,
               s.claimed_stem AS claimed_stem,
               COALESCE(s.cue_time_seconds, s.cue_seconds) AS cue_time_s,
               sa.set_audio_id AS set_audio_id, sa.path AS mix_full_path,
               ta.path AS ref_path
        FROM set_track_slots s
        LEFT JOIN set_audio sa ON sa.set_id = s.set_id AND sa.is_reference = 1
        LEFT JOIN track_audio ta ON ta.recording_id = s.recording_id
                                 AND ta.stem = s.claimed_stem
                                 AND ta.is_reference = 1
        WHERE s.claimed_stem IN ({placeholders})
          AND s.recording_id IS NOT NULL
        ORDER BY s.set_id, s.row_index
    """
    return list(conn.execute(sql, list(stems)).fetchall())


def _classify(row: sqlite3.Row, *, stems_root: Path, set_audio_root: Path | None) -> str:
    """First-missing wins: cue → ref → mix(row/file) → mix-stem(instrumental) → ok."""
    if row["cue_time_s"] is None:
        return "no-cue-time"
    if row["ref_path"] is None:
        return "no-ref-audio"
    if row["mix_full_path"] is None:
        return "no-mix-audio"
    mix = _resolve(row["mix_full_path"], set_audio_root)
    if not mix.is_file():
        return "no-mix-audio"
    if row["claimed_stem"] == "instrumental":
        stem_file = Path(stems_root) / str(row["set_audio_id"]) / "instrumental.flac"
        if not stem_file.is_file():
            return "no-mix-stem"
    return "eligible-now"


def census(
    conn: sqlite3.Connection,
    *,
    stems_root: Path,
    policy_stems: Iterable[str],
    set_audio_root: Path | None = None,
) -> CensusReport:
    """Classify every certified-axis slot by what blocks harvest, disk-checked."""
    by_axis: dict[str, dict[str, int]] = {}
    for row in census_rows(conn, policy_stems=policy_stems):
        axis = row["claimed_stem"]
        cat = _classify(row, stems_root=stems_root, set_audio_root=set_audio_root)
        bucket = by_axis.setdefault(axis, {c: 0 for c in _CENSUS_CATEGORIES})
        bucket[cat] += 1
    return CensusReport(by_axis)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `venvs/audio/bin/python -m pytest pws_aligner/tests/test_corpus_harvest.py -k census -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add pws_aligner/corpus_harvest.py pws_aligner/tests/test_corpus_harvest.py
git commit -m "feat(cotrain): corpus-harvest eligibility census (disk-checked funnel)"
```

---

## Task 5: `main` CLI wiring (both modes) + full-suite green

**Files:**
- Modify: `pws_aligner/corpus_harvest.py`
- Test: `pws_aligner/tests/test_corpus_harvest.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `main(argv: list[str] | None = None) -> int`; `if __name__ == "__main__": sys.exit(main())`.

- [ ] **Step 1: Write the failing test**

Append to `test_corpus_harvest.py`:

```python
from pws_aligner.corpus_harvest import main  # noqa: E402


def _write_fixture_db(path: Path) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE set_track_slots (
            set_id TEXT, row_index INTEGER, recording_id TEXT, slot_label TEXT,
            cue_seconds INTEGER, cue_time_seconds INTEGER,
            claimed_version TEXT, claimed_stem TEXT, claimed_variant TEXT,
            duration_seconds INTEGER
        );
        CREATE TABLE set_audio (
            set_audio_id INTEGER, set_id TEXT, path TEXT, sha256 TEXT,
            is_reference INTEGER
        );
        CREATE TABLE track_audio (
            track_audio_id INTEGER, recording_id TEXT, stem TEXT,
            path TEXT, is_reference INTEGER
        );
        INSERT INTO set_track_slots VALUES
            ('A',0,'R1','001',NULL,100,'original','regular','regular',42),
            ('A',1,'R2','002',NULL,NULL,'original','regular','regular',42);
        INSERT INTO set_audio VALUES (1,'A','/nope/mix.m4a',NULL,1);
        INSERT INTO track_audio VALUES (NULL,'R1','regular','/nope/r1.flac',1);
        INSERT INTO track_audio VALUES (NULL,'R2','regular','/nope/r2.flac',1);
        """
    )
    conn.commit()
    conn.close()


def test_main_census_mode_runs_on_fixture_db(tmp_path, capsys):
    db = tmp_path / "fix.db"
    _write_fixture_db(db)
    rc = main(["--db", str(db), "--stems-root", str(tmp_path / "stems"), "--census"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "eligibility census" in out
    # R1 has cue+ref but mix file absent → no-mix-audio; R2 no cue → no-cue-time
    payload = json.loads(out.strip().splitlines()[-1])
    assert payload["by_axis"]["regular"]["no-mix-audio"] == 1
    assert payload["by_axis"]["regular"]["no-cue-time"] == 1


def test_main_rejects_uncertified_stem(tmp_path):
    db = tmp_path / "fix.db"
    _write_fixture_db(db)
    try:
        main(["--db", str(db), "--stem", "acappella", "--census"])
        assert False, "expected SystemExit on uncertified stem"
    except SystemExit as e:
        assert e.code != 0


def test_main_harvest_mode_requires_out(tmp_path):
    db = tmp_path / "fix.db"
    _write_fixture_db(db)
    try:
        main(["--db", str(db)])  # no --out, no --census
        assert False, "expected SystemExit when --out missing"
    except SystemExit as e:
        assert e.code != 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest pws_aligner/tests/test_corpus_harvest.py -k main -v`
Expected: FAIL — `ImportError: cannot import name 'main'`.

- [ ] **Step 3: Write minimal implementation**

Append to `corpus_harvest.py`:

```python
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Corpus-harvest CLI — co-training flywheel step 2."
    )
    ap.add_argument("--db", default=str(DEFAULT_DB), help="canonical DB path")
    ap.add_argument("--stems-root", default=str(DEFAULT_STEMS_ROOT),
                    help="mix-side stems root: <root>/<set_audio_id>/instrumental.flac")
    ap.add_argument("--set-audio-root", default=None,
                    help="prefix for relative set_audio.path values (optional)")
    ap.add_argument("--ref-audio-root", default=None,
                    help="prefix for relative track_audio.path values (optional)")
    ap.add_argument("--out", default=None, help="harvest-ledger JSONL (required unless --census)")
    ap.add_argument("--limit", type=int, default=None, help="cap eligible slots (harvest mode)")
    ap.add_argument("--stem", default=None, help="restrict to one certified axis")
    ap.add_argument("--census", action="store_true",
                    help="report eligibility without running probes (no --out needed)")
    args = ap.parse_args(argv)

    policy: dict[str, BandThresholds] = CERTIFIED_POLICY
    if args.stem:
        if args.stem not in CERTIFIED_POLICY:
            ap.error(f"uncertified stem {args.stem!r} — not in CERTIFIED_POLICY")
        policy = {args.stem: CERTIFIED_POLICY[args.stem]}
    policy_stems = tuple(policy)

    set_audio_root = Path(args.set_audio_root) if args.set_audio_root else None
    ref_audio_root = Path(args.ref_audio_root) if args.ref_audio_root else None
    stems_root = Path(args.stems_root)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    try:
        if args.census:
            report = census(
                conn, stems_root=stems_root, policy_stems=policy_stems,
                set_audio_root=set_audio_root,
            )
            print(report.render())
            print(json.dumps(report.to_json()))
            return 0

        if not args.out:
            ap.error("--out is required unless --census")
        slots = query_corpus_slots(conn, policy_stems=policy_stems, limit=args.limit)
        summary = run_corpus_harvest(
            slots, stems_root=stems_root, out=Path(args.out), policy=policy,
            set_audio_root=set_audio_root, ref_audio_root=ref_audio_root,
        )
        print(json.dumps(summary.to_json()))
        print(f"ledger={args.out}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the full test file + the module's existing suite**

Run: `venvs/audio/bin/python -m pytest pws_aligner/tests/test_corpus_harvest.py -v`
Expected: PASS (all tasks' tests green).

Run: `venvs/audio/bin/python -m pytest pws_aligner/tests/test_harvest.py pws_aligner/tests/test_corpus_scorer.py -v`
Expected: PASS (no regression in the machinery this glue depends on).

- [ ] **Step 5: Commit**

```bash
git add pws_aligner/corpus_harvest.py pws_aligner/tests/test_corpus_harvest.py
git commit -m "feat(cotrain): corpus_harvest main() CLI (harvest + --census modes)"
```

---

## Task 6: Guardrails + docstring polish + module CLAUDE.md pointer

**Files:**
- Modify: `pws_aligner/CLAUDE.md` (add a one-line pointer to the new CLI).
- Verify: `pws_aligner/corpus_harvest.py` module docstring is accurate.

**Interfaces:**
- Consumes: the finished module.
- Produces: no code — docs + guardrail pass.

- [ ] **Step 1: Add a pointer to `pws_aligner/CLAUDE.md`**

Under the "What's reusable" list, add:

```markdown
- `corpus_harvest.py` — flywheel step-2 batch CLI: pi DB → cue-anchored
  positive-only cases → certified-probe scorer (`corpus_mix_resolver`) → harvest
  ledger; `--census` reports eligibility (recall ceiling) before the GPU stem
  pass. CPU-only for the certified axes (regular/instrumental); runs on pi-storage.
```

- [ ] **Step 2: Run guardrails**

Run: `venvs/audio/bin/python scripts/guardrails.py`
Expected: prints `guardrails: OK` (WARNs about state-of-record staleness / docs-gc are pre-existing and unrelated; a `mypy` "No module named mypy" line is the known env gap, not a failure).

- [ ] **Step 3: Full module suite once more**

Run: `venvs/audio/bin/python -m pytest pws_aligner/tests/ -q`
Expected: PASS (whole pws_aligner suite green).

- [ ] **Step 4: Commit**

```bash
git add pws_aligner/CLAUDE.md
git commit -m "docs(cotrain): point pws_aligner/CLAUDE.md at corpus_harvest CLI"
```

---

## Self-Review

**Spec coverage:**
- §5.1 `CorpusSlot` → Task 1. §5.2 `query_corpus_slots` → Task 1. §5.3 `build_corpus_cases` (positive-only, cue anchor, DEFAULT_SPAN_S, claim_axes) → Task 2. §5.4 `run_corpus_harvest` (per-set scorer, feature-cache reuse via one scorer/set, incremental idempotent ledger, HarvestSummary) → Task 3. §5.5 `census` + `--census` (disk-checked, per-axis, blocker categories) → Task 4. §5.6 `main` argparse (all flags, `--stem` cert guard, `--out` unless `--census`) → Task 5. §4 path defaults → Task 1 constants + Task 5 args. §7 error handling (missing audio → abstain via scorer; bad `--db`/`--stem` → fail-fast `ap.error`/`SystemExit`) → Tasks 3–5. §8 testing (fixture DB + fake scorer, all 5 listed cases incl. certified-policy guard) → Tasks 1–5. §9 out-of-scope respected (no GPU, no acappella harvest, no DB writes). §10 branch/CLAUDE pointer → Task 6.
- Note: `--ref-audio-root` (spec §4 table) is wired through `build_corpus_cases`/`run_corpus_harvest` and exposed in `main` (Tasks 2/3/5), tested in Task 2.

**Placeholder scan:** none — every code step shows complete code; every run step shows the exact command + expected result.

**Type consistency:** `CorpusSlot` fields identical across Tasks 1–4. `build_corpus_cases(slots, *, ref_audio_root=None)` signature identical in Tasks 2/3. `ScorerFactory = Callable[[Path, Path], RefMixScorer]` matches `_default_scorer_factory` and the test `factory(mix_full_path, mix_stem_dir)`. `harvest(cases, scorer, *, policy=...)` call matches the real signature (verified in `harvest.py`). `AlignmentResult(...)` kwargs (`recording_id/offset_s/confidence/source`) match `contract.py`; test confidences are in `[0,1]` (post-init guard). `write_ledger(records, out)` returns n-written, used for `n_written`. `CERTIFIED_POLICY` keys (`regular`, `instrumental`) drive both the SQL `policy_stems` and the banding — acappella absent everywhere.
