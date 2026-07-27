# Acquisition Data Engine — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the ad-hoc wrong-version/missing-track remediation flow into a persistent, deduped, queryable **worklist** — cases opened at *detection* time from two co-equal sources (the aligner residual + the manual wrong-version scan), plus a deferred "is it matchable yet?" gate-1 check.

**Architecture:** Reuse the existing `core/acquisition_case.py` state machine (JSONL store, `CaseStatus`/`ProblemClass` enums, `record_attempt` seam). Add: (a) an `impact_score` field + an `open_case()` find-or-create-in-OPEN seam + a global `open_worklist()` query in core; (b) a serializer that writes the scorer's never-matched GT recordings to JSON; (c) two case-source adapters in `ingest/`; (d) a DB-read matchability predicate + verify pass in `ingest/`. No DB migration this phase — the JSONL store stays; a pure query layer makes it globally queryable.

**Tech Stack:** Python 3.12+ (stdlib `dataclasses`/`enum`/`json`/`sqlite3`), pytest. Run everything from repo root with `venvs/audio/bin/python`.

## Global Constraints

- **Style:** `from __future__ import annotations` at top of every module; full type hints; frozen dataclasses for records; pure functions with file/DB IO only at the edges. (Copied from repo CLAUDE.md "Rust-flavoured functional Python".)
- **Substrate rule:** `core/` imports nothing upward. DB-touching code lives in a stage module (`ingest/`), never in `core/`. `core/acquisition_case.py` stays DB-free (it declares "No DB, schema soft").
- **Errors:** library/core returns values; CLI scripts fail-fast with `sys.exit`. Read-only DB predicates may use `sqlite3` directly (precedent: `scripts/scan_wrong_versions.py:126`).
- **Case identity:** a case is keyed by `(set_id, slot_label, recording_id)` via `find_case_index`; `case_id` property is `f"{set_id}:{slot_label}:{layer_role}"`. **Dedup guarantee:** opening the same key twice must never produce two cases.
- **Authoritative, do not touch:** `track_audio_correction` ledger and GT YAML/`set_ground_truth` stay authoritative; a case is a trace/worklist entry, never an override.
- **Enums are open-vocab:** reuse existing `ProblemClass` members; do not rename. New failure modes get new members only when a real case demands.
- **Set ids:** BB11 = `2nvzlh2k`, BB12 = `1fsnxchk`.
- **Out of scope this phase:** the metric-close re-score arm (Phase 2), any DB table/migration, cross-set slippage (a model issue).
- **Tests:** pytest, mirroring `tests/test_acquisition_case.py`. Run from repo root: `venvs/audio/bin/python -m pytest <path> -v`.

---

### Task 1: Add `impact_score` field to `AcquisitionCase`

Lets the worklist rank cases by estimated metric cost (higher = worse). Free of behavior change otherwise.

**Files:**
- Modify: `core/acquisition_case.py` (the `AcquisitionCase` dataclass ~line 146; `case_to_dict` ~line 223; `case_from_dict` ~line 262)
- Test: `tests/test_acquisition_case.py`

**Interfaces:**
- Produces: `AcquisitionCase.impact_score: int` (default `0`); round-trips through `case_to_dict`/`case_from_dict`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_acquisition_case.py`:

```python
def test_impact_score_round_trips():
    from core.acquisition_case import (
        AcquisitionCase, CaseClaim, case_to_dict, case_from_dict,
    )
    case = AcquisitionCase(
        set_id="1fsnxchk",
        slot_label="097",
        layer_role="solo",
        claim=CaseClaim(recording_id="1jz334x5"),
        impact_score=3,
    )
    back = case_from_dict(case_to_dict(case))
    assert back.impact_score == 3

def test_impact_score_defaults_zero_when_absent():
    from core.acquisition_case import case_from_dict
    # legacy dict with no impact_score key
    back = case_from_dict({"set_id": "x", "slot_label": "1", "claim": {"recording_id": "r"}})
    assert back.impact_score == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/test_acquisition_case.py::test_impact_score_round_trips tests/test_acquisition_case.py::test_impact_score_defaults_zero_when_absent -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'impact_score'`.

- [ ] **Step 3: Add the field and wire serialization**

In `core/acquisition_case.py`, in the `AcquisitionCase` dataclass, add the field after `notes`:

```python
    notes: str = ""
    impact_score: int = 0  # estimated metric cost (affected spans); worklist rank
```

In `case_to_dict`, add to the returned dict (next to `"notes": case.notes,`):

```python
        "impact_score": case.impact_score,
```

In `case_from_dict`, add to the `AcquisitionCase(...)` constructor call (next to `notes=...`):

```python
        impact_score=int(d.get("impact_score") or 0),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venvs/audio/bin/python -m pytest tests/test_acquisition_case.py -v`
Expected: PASS (all existing tests + the two new ones).

- [ ] **Step 5: Commit**

```bash
git add core/acquisition_case.py tests/test_acquisition_case.py
git commit -m "feat(acquisition): add impact_score to AcquisitionCase for worklist ranking

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `open_case()` — find-or-create in OPEN at detection time

Today a case is only created *after* an attempt (`record_attempt`). This adds the missing seam: open a case the moment a problem is *detected*, with no attempt, deduped by key.

**Files:**
- Modify: `core/acquisition_case.py` (add function after `record_attempt`, ~line 410)
- Test: `tests/test_acquisition_case.py`

**Interfaces:**
- Consumes: `load_cases`, `save_cases`, `find_case_index`, `with_problem_classes`, `default_path`, `CaseClaim`, `CaseStatus`, `ProblemClass`, `derive_layer_role`.
- Produces:
  ```python
  def open_case(*, set_id: str, slot_label: str, recording_id: str,
                problem_classes: Iterable[ProblemClass] = (), impact_score: int = 0,
                claim: CaseClaim | None = None, layer_role: LayerRole | None = None,
                notes: str = "", root: str | Path = "data/acquisition_cases",
                ) -> AcquisitionCase
  ```

- [ ] **Step 1: Write the failing test**

Add to `tests/test_acquisition_case.py`:

```python
def test_open_case_creates_open_case(tmp_path):
    from core.acquisition_case import open_case, load_cases, default_path, CaseStatus, ProblemClass
    c = open_case(
        set_id="1fsnxchk", slot_label="097", recording_id="1jz334x5",
        problem_classes=(ProblemClass.MISSING_ASSET,), impact_score=1,
        notes="never matched", root=tmp_path,
    )
    assert c.status is CaseStatus.OPEN
    assert ProblemClass.MISSING_ASSET in c.problem_classes
    assert c.impact_score == 1
    on_disk = load_cases(default_path("1fsnxchk", root=tmp_path))
    assert len(on_disk) == 1

def test_open_case_is_idempotent_and_merges(tmp_path):
    from core.acquisition_case import open_case, load_cases, default_path, ProblemClass
    open_case(set_id="s", slot_label="1", recording_id="r",
              problem_classes=(ProblemClass.MISSING_ASSET,), impact_score=1, root=tmp_path)
    open_case(set_id="s", slot_label="1", recording_id="r",
              problem_classes=(ProblemClass.WRONG_VERSION,), impact_score=3, root=tmp_path)
    cases = load_cases(default_path("s", root=tmp_path))
    assert len(cases) == 1  # dedup guarantee
    assert ProblemClass.MISSING_ASSET in cases[0].problem_classes
    assert ProblemClass.WRONG_VERSION in cases[0].problem_classes
    assert cases[0].impact_score == 3  # takes the max
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/test_acquisition_case.py::test_open_case_creates_open_case -v`
Expected: FAIL — `ImportError: cannot import name 'open_case'`.

- [ ] **Step 3: Implement `open_case`**

In `core/acquisition_case.py`, after `record_attempt` (end of file):

```python
def open_case(
    *,
    set_id: str,
    slot_label: str,
    recording_id: str,
    problem_classes: Iterable[ProblemClass] = (),
    impact_score: int = 0,
    claim: CaseClaim | None = None,
    layer_role: LayerRole | None = None,
    notes: str = "",
    root: str | Path = "data/acquisition_cases",
) -> AcquisitionCase:
    """Find-or-create the ``(set_id, slot_label, recording_id)`` case in OPEN.

    The detection-time counterpart of ``record_attempt``: opens a case with *no*
    attempt when a problem is first spotted. Idempotent — an existing case (any
    status) is merged (problem_classes union, ``impact_score`` max, ``notes``
    appended) and returned; a case is never duplicated.
    """
    from core.slot_inventory import derive_layer_role

    path = default_path(set_id, root=root)
    cases = load_cases(path)
    idx = find_case_index(cases, slot_label, recording_id)
    if idx is None:
        seed_claim = claim or CaseClaim(recording_id=recording_id)
        role = layer_role or derive_layer_role(slot_label, claimed_stem=seed_claim.stem)
        cases.append(
            AcquisitionCase(
                set_id=set_id,
                slot_label=slot_label,
                layer_role=role,
                claim=seed_claim,
                status=CaseStatus.OPEN,
            )
        )
        idx = len(cases) - 1

    case = cases[idx]
    if problem_classes:
        case = with_problem_classes(case, *problem_classes)
    if impact_score > case.impact_score:
        case = replace(case, impact_score=impact_score)
    if notes and notes not in case.notes:
        case = replace(case, notes=(f"{case.notes}\n{notes}".strip() if case.notes else notes))
    cases[idx] = case
    save_cases(path, cases)
    return case
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venvs/audio/bin/python -m pytest tests/test_acquisition_case.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/acquisition_case.py tests/test_acquisition_case.py
git commit -m "feat(acquisition): open_case() find-or-create seam (detection-time, deduped)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Global worklist query — `load_all_cases()` + `open_worklist()`

Makes "what's still open across the corpus?" a single call, ranked by impact — the cure for re-scanning to rediscover the same suspects.

**Files:**
- Modify: `core/acquisition_case.py` (add after `open_case`)
- Test: `tests/test_acquisition_case.py`

**Interfaces:**
- Produces:
  ```python
  def load_all_cases(root: str | Path = "data/acquisition_cases") -> list[AcquisitionCase]
  def open_worklist(root: str | Path = "data/acquisition_cases") -> list[AcquisitionCase]
  ```
  `open_worklist` returns only `CaseStatus.OPEN` cases, sorted by `impact_score` descending, then `(set_id, slot_label)` ascending for stability.

- [ ] **Step 1: Write the failing test**

```python
def test_open_worklist_spans_sets_and_ranks_by_impact(tmp_path):
    from core.acquisition_case import open_case, open_worklist, ProblemClass
    open_case(set_id="A", slot_label="1", recording_id="r1", impact_score=1, root=tmp_path)
    open_case(set_id="B", slot_label="2", recording_id="r2", impact_score=5, root=tmp_path)
    wl = open_worklist(root=tmp_path)
    assert [c.set_id for c in wl] == ["B", "A"]  # higher impact first

def test_open_worklist_excludes_non_open(tmp_path):
    from core.acquisition_case import open_case, open_worklist, load_cases, save_cases, default_path, CaseStatus
    from dataclasses import replace
    open_case(set_id="A", slot_label="1", recording_id="r1", root=tmp_path)
    cases = load_cases(default_path("A", root=tmp_path))
    save_cases(default_path("A", root=tmp_path), [replace(cases[0], status=CaseStatus.RESOLVED)])
    assert open_worklist(root=tmp_path) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/test_acquisition_case.py::test_open_worklist_spans_sets_and_ranks_by_impact -v`
Expected: FAIL — `ImportError: cannot import name 'open_worklist'`.

- [ ] **Step 3: Implement the queries**

In `core/acquisition_case.py`, after `open_case`:

```python
def load_all_cases(root: str | Path = "data/acquisition_cases") -> list[AcquisitionCase]:
    """Load every case across all ``{set_id}.jsonl`` files under ``root``."""
    r = Path(root)
    if not r.exists():
        return []
    out: list[AcquisitionCase] = []
    for p in sorted(r.glob("*.jsonl")):
        out.extend(load_cases(p))
    return out


def open_worklist(root: str | Path = "data/acquisition_cases") -> list[AcquisitionCase]:
    """All OPEN cases corpus-wide, highest ``impact_score`` first."""
    opens = [c for c in load_all_cases(root) if c.status == CaseStatus.OPEN]
    return sorted(opens, key=lambda c: (-c.impact_score, c.set_id, c.slot_label))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venvs/audio/bin/python -m pytest tests/test_acquisition_case.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/acquisition_case.py tests/test_acquisition_case.py
git commit -m "feat(acquisition): load_all_cases + open_worklist global query

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Serialize the scorer's never-matched GT recordings

The residual source's fuel. The scorer *computes* never-matched recordings but only `print()`s a count and only for acappellas. This extracts a pure, generalized (all-stems) helper and writes it to JSON.

**Files:**
- Create: `alignment/never_matched.py`
- Test: `tests/test_never_matched.py`
- (Wiring, untested glue) Modify: `alignment/score_timeline_vs_gt.py` — add a `--emit-never-matched PATH` flag in the `--decompose` block (~lines 622–633) that calls the helper with the in-scope `gt_rows` and `spans` and writes the JSON.

**Interfaces:**
- Produces:
  ```python
  def never_matched_recordings(gt_rows: list[dict], spans: list[dict]) -> list[dict]
  def write_never_matched(set_id: str, gt_rows: list[dict], spans: list[dict], out_path: Path) -> dict
  ```
  Each entry: `{"recording_id": str, "slot_label": str, "claimed_stem": str, "reason": str}`.
  JSON file shape: `{"set_id": str, "never_matched_recordings": [entry, ...]}`.
  A GT row is "matched" iff its `track_id` appears as any span's `recording_id`. Rows with no `track_id` (mix-only/phantom) are skipped (nothing to acquire).

- [ ] **Step 1: Write the failing test**

Create `tests/test_never_matched.py`:

```python
from pathlib import Path
import json
from alignment.never_matched import (
    never_matched_recordings, write_never_matched,
)

_GT = [
    {"track_id": "matched1", "slot_label": "010", "claimed_stem": "regular"},
    {"track_id": "gone1", "slot_label": "097", "claimed_stem": "acappella"},
    {"track_id": "gone2", "slot_label": "121", "claimed_stem": "instrumental"},
    {"track_id": None, "slot_label": "200", "claimed_stem": "regular"},  # mix-only, skip
]
_SPANS = [{"recording_id": "matched1", "slot_label": "010"}]

def test_never_matched_covers_all_stems_and_skips_trackless():
    out = never_matched_recordings(_GT, _SPANS)
    ids = {e["recording_id"] for e in out}
    assert ids == {"gone1", "gone2"}  # instrumental included, not acappella-only
    assert all("slot_label" in e and "claimed_stem" in e for e in out)

def test_write_never_matched_json_shape(tmp_path):
    p = tmp_path / "nm.json"
    doc = write_never_matched("1fsnxchk", _GT, _SPANS, p)
    assert doc["set_id"] == "1fsnxchk"
    on_disk = json.loads(p.read_text())
    assert on_disk == doc
    assert len(on_disk["never_matched_recordings"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/test_never_matched.py -v`
Expected: FAIL — `ModuleNotFoundError: alignment.never_matched`.

- [ ] **Step 3: Implement the helper**

Create `alignment/never_matched.py`:

```python
"""Serialize the GT recordings that no predicted span matched — the residual
source's fuel. Pure over ``gt_rows`` / ``spans`` dicts (see score_timeline_vs_gt).
"""
from __future__ import annotations

import json
from pathlib import Path


def never_matched_recordings(gt_rows: list[dict], spans: list[dict]) -> list[dict]:
    """GT rows whose ``track_id`` matched no span's ``recording_id``.

    Covers every stem (not just acappella). Rows without a ``track_id``
    (mix-only / phantom hosts) are skipped — there is nothing to acquire.
    """
    matched = {str(s.get("recording_id")) for s in spans if s.get("recording_id")}
    out: list[dict] = []
    for r in gt_rows:
        tid = r.get("track_id")
        if not tid:
            continue
        if str(tid) in matched:
            continue
        out.append(
            {
                "recording_id": str(tid),
                "slot_label": str(r.get("slot_label") or ""),
                "claimed_stem": str(r.get("claimed_stem") or ""),
                "reason": "never matched by any predicted span",
            }
        )
    return out


def write_never_matched(
    set_id: str, gt_rows: list[dict], spans: list[dict], out_path: Path
) -> dict:
    """Write ``{set_id, never_matched_recordings}`` JSON; return the doc."""
    doc = {
        "set_id": set_id,
        "never_matched_recordings": never_matched_recordings(gt_rows, spans),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venvs/audio/bin/python -m pytest tests/test_never_matched.py -v`
Expected: PASS.

- [ ] **Step 5: Wire the flag into the scorer (glue — no new test)**

In `alignment/score_timeline_vs_gt.py`, in the `--decompose` block where `gt_rows`, `spans`, and `matched_tids` are already in scope (~lines 622–633), add after the existing never-matched print:

```python
        if getattr(args, "emit_never_matched", None):
            from alignment.never_matched import write_never_matched
            from pathlib import Path as _Path
            write_never_matched(set_id, gt_rows, spans, _Path(args.emit_never_matched))
```

And register the flag in the argparse setup (next to the `--decompose` flag):

```python
    ap.add_argument("--emit-never-matched", type=str, default=None,
                    help="write never-matched GT recordings to this JSON path")
```

- [ ] **Step 6: Sanity-run against BB12 (manual verification)**

Run: `venvs/audio/bin/python -m alignment.score_timeline_vs_gt --set-id 1fsnxchk --decompose --emit-never-matched out/1fsnxchk_never_matched.json` (adjust to the script's actual timeline arg).
Expected: `out/1fsnxchk_never_matched.json` exists and lists ~11 recordings. If the arg surface differs, match the script's existing `--decompose` invocation; do not invent flags.

- [ ] **Step 7: Commit**

```bash
git add alignment/never_matched.py tests/test_never_matched.py alignment/score_timeline_vs_gt.py
git commit -m "feat(align): serialize never-matched GT recordings to JSON (all stems)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Residual source — open cases from `never_matched.json`

**Files:**
- Create: `ingest/acquisition_sources.py`
- Test: `tests/test_acquisition_sources.py`

**Interfaces:**
- Consumes: `core.acquisition_case.open_case`, `ProblemClass`.
- Produces:
  ```python
  def open_cases_from_never_matched(json_path: str | Path,
                                    root: str | Path = "data/acquisition_cases",
                                    ) -> list[AcquisitionCase]
  ```

- [ ] **Step 1: Write the failing test**

Create `tests/test_acquisition_sources.py`:

```python
import json
from pathlib import Path
from core.acquisition_case import open_worklist, ProblemClass
from ingest.acquisition_sources import open_cases_from_never_matched

def test_residual_source_opens_one_case_per_entry(tmp_path):
    nm = tmp_path / "1fsnxchk_never_matched.json"
    nm.write_text(json.dumps({
        "set_id": "1fsnxchk",
        "never_matched_recordings": [
            {"recording_id": "1jz334x5", "slot_label": "097", "claimed_stem": "acappella", "reason": "x"},
            {"recording_id": "1q9r2r8x", "slot_label": "121", "claimed_stem": "instrumental", "reason": "y"},
        ],
    }))
    root = tmp_path / "cases"
    opened = open_cases_from_never_matched(nm, root=root)
    assert len(opened) == 2
    wl = open_worklist(root=root)
    assert len(wl) == 2
    assert all(ProblemClass.MISSING_ASSET in c.problem_classes for c in wl)

def test_residual_source_is_idempotent(tmp_path):
    nm = tmp_path / "s_never_matched.json"
    nm.write_text(json.dumps({"set_id": "s", "never_matched_recordings": [
        {"recording_id": "r", "slot_label": "1", "claimed_stem": "regular", "reason": "x"}]}))
    root = tmp_path / "cases"
    open_cases_from_never_matched(nm, root=root)
    open_cases_from_never_matched(nm, root=root)
    assert len(open_worklist(root=root)) == 1  # no duplicate
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/test_acquisition_sources.py -v`
Expected: FAIL — `ModuleNotFoundError: ingest.acquisition_sources`.

- [ ] **Step 3: Implement the residual source**

Create `ingest/acquisition_sources.py`:

```python
"""Case-source adapters: turn detection signals into OPEN acquisition cases.

Imports ``core.acquisition_case`` (downward only). The residual source consumes
the scorer's ``never_matched.json``; the manual-scan source (Task 6) consumes
wrong-version suspects.
"""
from __future__ import annotations

import json
from pathlib import Path

from core.acquisition_case import AcquisitionCase, ProblemClass, open_case


def open_cases_from_never_matched(
    json_path: str | Path,
    root: str | Path = "data/acquisition_cases",
) -> list[AcquisitionCase]:
    """Open one OPEN case per never-matched GT recording (deduped)."""
    doc = json.loads(Path(json_path).read_text(encoding="utf-8"))
    set_id = str(doc.get("set_id") or "")
    opened: list[AcquisitionCase] = []
    for entry in doc.get("never_matched_recordings", []):
        opened.append(
            open_case(
                set_id=set_id,
                slot_label=str(entry.get("slot_label") or ""),
                recording_id=str(entry.get("recording_id") or ""),
                problem_classes=(ProblemClass.MISSING_ASSET,),
                impact_score=1,
                notes=str(entry.get("reason") or "never matched by any predicted span"),
                root=root,
            )
        )
    return opened
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venvs/audio/bin/python -m pytest tests/test_acquisition_sources.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add ingest/acquisition_sources.py tests/test_acquisition_sources.py
git commit -m "feat(ingest): residual source opens cases from never_matched.json

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Manual-scan source — `scan_wrong_versions.py --open-cases`

Second co-equal source. A wrong-version suspect is corpus-wide (keyed by track), so it maps to a case per *placement* — every `(set_id, slot_label)` where that track is claimed in `set_track_slots`. The mapping is a pure function (testable without a DB); the DB join is thin glue.

**Files:**
- Create: `ingest/scan_source.py` (pure mapping + a DB-join helper)
- Modify: `scripts/scan_wrong_versions.py` (add `--open-cases` flag calling the helper)
- Test: `tests/test_scan_source.py`

**Interfaces:**
- Consumes: `core.acquisition_case.open_case`, `ProblemClass`; the `Suspect` dataclass from `scripts/scan_wrong_versions.py`.
- Produces:
  ```python
  # pure: suspect + its placements -> opened cases
  def open_cases_for_suspect(track_id: str, klass: str, detail: str,
                             placements: list[tuple[str, str]],  # (set_id, slot_label)
                             root: str | Path = "data/acquisition_cases",
                             ) -> list[AcquisitionCase]
  # DB glue: find where a track is claimed
  def placements_for_track(db_path: Path, track_id: str) -> list[tuple[str, str]]
  ```
- `klass` → `ProblemClass`: `topic_original`/`wrong_remix`/`live_suspect` → `WRONG_VERSION`; `mashup_metadata_gap` → `STRUCTURE`; anything else → `WRONG_VERSION`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_scan_source.py`:

```python
from core.acquisition_case import open_worklist, ProblemClass
from ingest.scan_source import open_cases_for_suspect, _klass_to_problem

def test_klass_mapping():
    assert _klass_to_problem("topic_original") is ProblemClass.WRONG_VERSION
    assert _klass_to_problem("mashup_metadata_gap") is ProblemClass.STRUCTURE
    assert _klass_to_problem("anything_else") is ProblemClass.WRONG_VERSION

def test_suspect_opens_case_per_placement(tmp_path):
    opened = open_cases_for_suspect(
        track_id="r1", klass="wrong_remix", detail="oEmbed lacks remixer",
        placements=[("1fsnxchk", "030"), ("2nvzlh2k", "045")], root=tmp_path,
    )
    assert len(opened) == 2
    wl = open_worklist(root=tmp_path)
    assert {c.set_id for c in wl} == {"1fsnxchk", "2nvzlh2k"}
    assert all(ProblemClass.WRONG_VERSION in c.problem_classes for c in wl)

def test_suspect_with_no_placements_opens_nothing(tmp_path):
    assert open_cases_for_suspect("r", "wrong_remix", "d", [], root=tmp_path) == []
    assert open_worklist(root=tmp_path) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/test_scan_source.py -v`
Expected: FAIL — `ModuleNotFoundError: ingest.scan_source`.

- [ ] **Step 3: Implement the source**

Create `ingest/scan_source.py`:

```python
"""Manual wrong-version scan → acquisition cases.

A suspect is keyed by track (corpus-wide); it maps to one case per placement in
``set_track_slots``. Pure mapping (``open_cases_for_suspect``) + a DB-join helper
(``placements_for_track``).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from core.acquisition_case import AcquisitionCase, ProblemClass, open_case

_KLASS: dict[str, ProblemClass] = {
    "topic_original": ProblemClass.WRONG_VERSION,
    "wrong_remix": ProblemClass.WRONG_VERSION,
    "live_suspect": ProblemClass.WRONG_VERSION,
    "mashup_metadata_gap": ProblemClass.STRUCTURE,
}


def _klass_to_problem(klass: str) -> ProblemClass:
    return _KLASS.get(klass, ProblemClass.WRONG_VERSION)


def open_cases_for_suspect(
    track_id: str,
    klass: str,
    detail: str,
    placements: list[tuple[str, str]],
    root: str | Path = "data/acquisition_cases",
) -> list[AcquisitionCase]:
    """Open one case per ``(set_id, slot_label)`` placement of a suspect track."""
    problem = _klass_to_problem(klass)
    opened: list[AcquisitionCase] = []
    for set_id, slot_label in placements:
        opened.append(
            open_case(
                set_id=set_id,
                slot_label=slot_label,
                recording_id=track_id,
                problem_classes=(problem,),
                impact_score=1,
                notes=f"wrong-version scan ({klass}): {detail}",
                root=root,
            )
        )
    return opened


def placements_for_track(db_path: Path, track_id: str) -> list[tuple[str, str]]:
    """Every ``(set_id, slot_label)`` where ``track_id`` is claimed."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT set_id, slot_label FROM set_track_slots WHERE recording_id = ?",
            (track_id,),
        ).fetchall()
    finally:
        conn.close()
    return [(str(s), str(sl)) for s, sl in rows]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venvs/audio/bin/python -m pytest tests/test_scan_source.py -v`
Expected: PASS.

- [ ] **Step 5: Wire `--open-cases` into the scan CLI (glue — no new test)**

In `scripts/scan_wrong_versions.py`, add the flag in `main()` (after `--limit`):

```python
    ap.add_argument("--open-cases", action="store_true",
                    help="open acquisition cases for each suspect (per placement)")
    ap.add_argument("--cases-root", type=Path, default=Path("data/acquisition_cases"))
```

Then after the suspects are computed (`suspects = scan(...)`), before/after the CSV write:

```python
    if args.open_cases:
        from ingest.scan_source import open_cases_for_suspect, placements_for_track
        n = 0
        for s in suspects:
            placements = placements_for_track(args.db, s.track_id)
            n += len(open_cases_for_suspect(
                s.track_id, s.klass, s.detail, placements, root=args.cases_root))
        print(f"Opened/updated {n} cases from {len(suspects)} suspects")
```

- [ ] **Step 6: Commit**

```bash
git add ingest/scan_source.py tests/test_scan_source.py scripts/scan_wrong_versions.py
git commit -m "feat(ingest): scan_wrong_versions --open-cases (case per placement)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Gate-1 matchability predicate + verify pass

Because `replace_track_audio.py` does not recompute features (the analyze loop does, later), "is the fix matchable by the aligner yet?" is a *deferred* check. This adds the DB predicate and a CLI verify pass that reports which resolved cases are matchable vs still awaiting features. (Actual close-on-metric is Phase 2.)

**Files:**
- Create: `ingest/matchability.py`
- Test: `tests/test_matchability.py`

**Interfaces:**
- Produces:
  ```python
  def has_matchable_features(db_path: Path, track_audio_id: int) -> bool
  def verify_worklist(db_path: Path, root: str | Path = "data/acquisition_cases",
                      ) -> list[tuple[str, bool]]  # (case_id, matchable)
  ```
  `has_matchable_features` is True iff the row's `(recording_id, stem)` has a `track_fingerprints` row AND the `track_audio_id` has ≥1 `track_mert_measures` row.

- [ ] **Step 1: Write the failing test**

Create `tests/test_matchability.py`:

```python
import sqlite3
from pathlib import Path
from ingest.matchability import has_matchable_features

def _fixture_db(tmp_path: Path) -> Path:
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE track_audio (track_audio_id INTEGER PRIMARY KEY, recording_id TEXT, stem TEXT);
        CREATE TABLE track_fingerprints (recording_id TEXT, stem TEXT);
        CREATE TABLE track_mert_measures (track_audio_id INTEGER);
        INSERT INTO track_audio VALUES (1, 'recA', 'regular');   -- fully matchable
        INSERT INTO track_audio VALUES (2, 'recB', 'regular');   -- fp only, no mert
        INSERT INTO track_audio VALUES (3, 'recC', 'regular');   -- nothing
        INSERT INTO track_fingerprints VALUES ('recA', 'regular');
        INSERT INTO track_fingerprints VALUES ('recB', 'regular');
        INSERT INTO track_mert_measures VALUES (1);
    """)
    conn.commit(); conn.close()
    return db

def test_matchable_true_when_fp_and_mert_present(tmp_path):
    assert has_matchable_features(_fixture_db(tmp_path), 1) is True

def test_not_matchable_when_mert_missing(tmp_path):
    assert has_matchable_features(_fixture_db(tmp_path), 2) is False

def test_not_matchable_when_nothing(tmp_path):
    assert has_matchable_features(_fixture_db(tmp_path), 3) is False

def test_not_matchable_when_row_absent(tmp_path):
    assert has_matchable_features(_fixture_db(tmp_path), 999) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest tests/test_matchability.py -v`
Expected: FAIL — `ModuleNotFoundError: ingest.matchability`.

- [ ] **Step 3: Implement the predicate + verify pass**

Create `ingest/matchability.py`:

```python
"""Gate-1: is a replaced track_audio row *matchable* by the aligner yet?

A row is matchable once the identity channels exist for it: a landmark
fingerprint (``track_fingerprints`` by ``recording_id``+``stem``) and at least
one MERT measure (``track_mert_measures`` by ``track_audio_id``). These are
computed asynchronously by the analyze loop after a replace, so this is a
deferred check, not a synchronous gate.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from core.acquisition_case import open_worklist


def has_matchable_features(db_path: Path, track_audio_id: int) -> bool:
    conn = sqlite3.connect(db_path)
    try:
        ta = conn.execute(
            "SELECT recording_id, stem FROM track_audio WHERE track_audio_id = ?",
            (track_audio_id,),
        ).fetchone()
        if ta is None:
            return False
        recording_id, stem = ta
        fp = conn.execute(
            "SELECT 1 FROM track_fingerprints WHERE recording_id = ? AND stem = ?",
            (recording_id, stem),
        ).fetchone()
        if fp is None:
            return False
        mert = conn.execute(
            "SELECT 1 FROM track_mert_measures WHERE track_audio_id = ? LIMIT 1",
            (track_audio_id,),
        ).fetchone()
        return mert is not None
    finally:
        conn.close()


def verify_worklist(
    db_path: Path, root: str | Path = "data/acquisition_cases"
) -> list[tuple[str, bool]]:
    """For each OPEN case that already picked a winning asset, report whether that
    asset is matchable yet. Cases with no resolution asset are skipped.
    """
    out: list[tuple[str, bool]] = []
    for case in open_worklist(root=root):
        taid = case.resolution.track_audio_id if case.resolution else None
        if taid is None:
            continue
        out.append((case.case_id, has_matchable_features(db_path, int(taid))))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venvs/audio/bin/python -m pytest tests/test_matchability.py -v`
Expected: PASS.

- [ ] **Step 5: Add a CLI entry (glue — no new test)**

Append to `ingest/matchability.py`:

```python
def _main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="report matchability of resolved cases")
    ap.add_argument("--db", type=Path,
                    default=Path("/mnt/storage/data/db/music_database.db"))
    ap.add_argument("--root", type=Path, default=Path("data/acquisition_cases"))
    args = ap.parse_args()
    for case_id, matchable in verify_worklist(args.db, root=args.root):
        print(f"{'MATCHABLE ' if matchable else 'PENDING   '} {case_id}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main())
```

- [ ] **Step 6: Run the full new test set + guardrails**

Run: `venvs/audio/bin/python -m pytest tests/test_acquisition_case.py tests/test_never_matched.py tests/test_acquisition_sources.py tests/test_scan_source.py tests/test_matchability.py -v`
Expected: PASS.
Run: `make check`
Expected: guardrails OK (pre-existing WARNs about state-of-record/docs-gc are unrelated).

- [ ] **Step 7: Commit**

```bash
git add ingest/matchability.py tests/test_matchability.py
git commit -m "feat(ingest): gate-1 matchability predicate + verify pass

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Phase 1 done — what exists after

- A **persistent, deduped, corpus-wide worklist** (`open_worklist()`), opened at detection time from **two co-equal sources**: the aligner residual (`never_matched.json` → `open_cases_from_never_matched`) and the manual wrong-version scan (`scan_wrong_versions.py --open-cases`).
- Every case tagged with a root-cause `ProblemClass` and an `impact_score` for ranking.
- A **gate-1 matchability check** (`has_matchable_features` / `verify_worklist`) that answers "can the aligner actually see this fix yet?" — the honest, deferred version given async feature compute.

## Deferred to Phase 2 (separate plan)

- **Metric-close arm:** re-run the scorer after a fix and flip OPEN → RESOLVED only when the GT span moves unmatched → matched. Requires the `never_matched.json` diff (Task 4 output) as the before/after baseline — already emitted here.
- **DB-table worklist:** promote the JSONL store to a queried table once the schema freezes / set-count grows past what per-set JSONL globbing handles.
- **Autonomy:** the self-running loop.
