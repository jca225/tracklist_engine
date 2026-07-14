# streaming_mir WS1 Prefetch + WS2 Vast Validation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hide the input download (and actually hide the output persist, which today is
accidentally serial) behind GPU analysis in `scripts/vast_loop.py`, instrument per-stage
timings, and validate both WS1 throughput and the WS2 seam fix on one Vast rental.

**Architecture:** A one-slot prefetch thread (`scripts/loop_prefetch.py`, pure/injectable,
torch-free) mirrors the existing persist thread; correctness comes from in-flight tid
exclusion passed to `next_task`. The persist thread's join moves from loop-top (where it
serializes everything — a latent bug) to just-before-next-handoff. WS2 validation uses a
shifted-grid pseudo-reference render, scored by boundary-local SDR.

**Tech Stack:** Python threading (no asyncio), subprocess ssh/rsync, numpy/soundfile for
SDR, Vast.ai via curl API.

**Spec:** [2026-07-14-ws1-prefetch-vast-validation-design.md](../specs/2026-07-14-ws1-prefetch-vast-validation-design.md)

## Global Constraints

- Never write causal/streaming estimates to the canonical store (streaming_mir design law).
- No new top-level directories; everything lands in `scripts/`, `analysis/`, `workspaces/streaming_mir/`, `tests/`.
- `scripts/loop_prefetch.py` must import NO heavy deps (no torch, no analysis.*) — tests import it directly.
- Do NOT set `os.environ["TRACKLIST_DISABLE_FK"]` in any module tests import (see render_set_stems.py:41-43 for why).
- Instrumentation is log-only: no changes to `TrackAnalysisResult`, persistence, or DB schema.
- `--no-prefetch` must reproduce today's exact serial behavior (it is the A/B baseline).
- Style: `from __future__ import annotations`, full type hints, frozen dataclasses, pure functions with I/O injected.
- Run tests as `venvs/audio/bin/pytest` from repo root. `make check` before push.
- Vast: list-before-create, destroy only instances you created (multi-agent coordination rule).

---

### Task 1: `scripts/loop_prefetch.py` — one-slot prefetch primitive

**Files:**
- Create: `scripts/loop_prefetch.py`
- Test: `tests/test_loop_prefetch.py`

**Interfaces:**
- Produces (Task 4 consumes):
  - `PrefetchItem(tid: int, local_audio: Path, asset: Any, pull_s: float)` (frozen dataclass)
  - `PrefetchFailure(tid: int, detail: str)` (frozen dataclass)
  - `PrefetchSlot(pick, pull, hydrate)` with `.pending: bool`, `.start(skip_tids: frozenset[int]) -> None`, `.take() -> PrefetchItem | PrefetchFailure | None`
  - `pick(skip: frozenset[int]) -> tuple[int, str] | None`; `pull(tid: int, remote_path: str) -> Path`; `hydrate(tid: int, local: Path) -> Any`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_loop_prefetch.py
"""Unit tests for the WS1 one-slot input-prefetch primitive.

PrefetchSlot is pure orchestration over three injected callables (pick /
pull / hydrate), so it is tested here with plain lambdas — no torch, no
network, no vast_loop import (importing vast_loop would set
TRACKLIST_DISABLE_FK and drag the GPU stack in).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.loop_prefetch import PrefetchFailure, PrefetchItem, PrefetchSlot


def _slot(pick=None, pull=None, hydrate=None) -> PrefetchSlot:
    return PrefetchSlot(
        pick=pick or (lambda skip: (7, "/remote/7.m4a")),
        pull=pull or (lambda tid, remote: Path(f"/local/{tid}.m4a")),
        hydrate=hydrate or (lambda tid, local: {"tid": tid}),
    )


def test_take_returns_item_on_success() -> None:
    slot = _slot()
    slot.start(frozenset())
    item = slot.take()
    assert isinstance(item, PrefetchItem)
    assert item.tid == 7
    assert item.local_audio == Path("/local/7.m4a")
    assert item.asset == {"tid": 7}
    assert item.pull_s >= 0.0


def test_take_returns_none_when_drained() -> None:
    slot = _slot(pick=lambda skip: None)
    slot.start(frozenset())
    assert slot.take() is None


def test_skip_tids_forwarded_to_pick() -> None:
    seen: list[frozenset[int]] = []

    def pick(skip: frozenset[int]):
        seen.append(skip)
        return None

    slot = _slot(pick=pick)
    slot.start(frozenset({3, 9}))
    slot.take()
    assert seen == [frozenset({3, 9})]


def test_pull_error_becomes_failure_with_tid() -> None:
    def pull(tid: int, remote: str) -> Path:
        raise RuntimeError("rsync exploded")

    slot = _slot(pull=pull)
    slot.start(frozenset())
    item = slot.take()
    assert isinstance(item, PrefetchFailure)
    assert item.tid == 7
    assert "rsync exploded" in item.detail


def test_hydrate_error_becomes_failure_with_tid() -> None:
    def hydrate(tid: int, local: Path):
        raise RuntimeError("ssh exploded")

    slot = _slot(hydrate=hydrate)
    slot.start(frozenset())
    item = slot.take()
    assert isinstance(item, PrefetchFailure)
    assert item.tid == 7


def test_single_slot_enforced() -> None:
    slot = _slot()
    slot.start(frozenset())
    with pytest.raises(AssertionError):
        slot.start(frozenset())
    slot.take()
    slot.start(frozenset())  # legal again after take()
    assert isinstance(slot.take(), PrefetchItem)


def test_take_without_start_raises() -> None:
    with pytest.raises(AssertionError):
        _slot().take()


def test_pending_reflects_slot_state() -> None:
    slot = _slot()
    assert not slot.pending
    slot.start(frozenset())
    assert slot.pending
    slot.take()
    assert not slot.pending
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venvs/audio/bin/pytest tests/test_loop_prefetch.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.loop_prefetch'`

- [ ] **Step 3: Write the implementation**

```python
# scripts/loop_prefetch.py
"""One-slot input prefetch for the analysis driver loops (streaming_mir WS1).

While the GPU analyzes track N, a single background thread picks and pulls
track N+1 so its audio is already local when the main loop needs it. This is
the input-side mirror of vast_loop's single-slot persist thread.

Imported, not run (like rescue_common.py). Deliberately torch-free and
I/O-free: the three I/O actions are injected callables, so vast_loop wires
ssh/rsync in and tests wire lambdas in. Keep it that way — tests import this
module directly.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class PrefetchItem:
    """A track whose audio is already local, ready to analyze."""

    tid: int
    local_audio: Path
    asset: Any  # core.models.AudioAsset — typed Any to stay import-light
    pull_s: float


@dataclass(frozen=True)
class PrefetchFailure:
    """pull/hydrate raised for this tid; caller treats it like today's
    per-track subprocess failure (log + failed_tids.add + continue)."""

    tid: int
    detail: str


class PrefetchSlot:
    """Single-slot prefetch thread: start() spawns, take() joins + returns.

    take() returns:
      - PrefetchItem     — picked, pulled, hydrated; ready to analyze
      - PrefetchFailure  — picked, but pull/hydrate raised
      - None             — pick() found nothing (queue drained for the given
                           skip set; the caller decides whether that's final)

    The single-slot join discipline (take() before the next start()) is the
    same happens-before contract the persist thread uses, so the plain
    attribute write in _run is safe to read after join.
    """

    def __init__(
        self,
        pick: Callable[[frozenset[int]], tuple[int, str] | None],
        pull: Callable[[int, str], Path],
        hydrate: Callable[[int, Path], Any],
    ) -> None:
        self._pick = pick
        self._pull = pull
        self._hydrate = hydrate
        self._thread: threading.Thread | None = None
        self._result: PrefetchItem | PrefetchFailure | None = None

    @property
    def pending(self) -> bool:
        return self._thread is not None

    def start(self, skip_tids: frozenset[int]) -> None:
        assert self._thread is None, "single slot: take() before next start()"

        def _run() -> None:
            picked = self._pick(skip_tids)
            if picked is None:
                self._result = None
                return
            tid, remote_path = picked
            try:
                t0 = time.monotonic()
                local = self._pull(tid, remote_path)
                pull_s = time.monotonic() - t0
                asset = self._hydrate(tid, local)
            except Exception as exc:  # rsync/ssh CalledProcessError et al.
                self._result = PrefetchFailure(tid=tid, detail=str(exc))
                return
            self._result = PrefetchItem(
                tid=tid, local_audio=local, asset=asset, pull_s=pull_s
            )

        self._result = None
        self._thread = threading.Thread(target=_run, daemon=False)
        self._thread.start()

    def take(self) -> PrefetchItem | PrefetchFailure | None:
        assert self._thread is not None, "take() without start()"
        self._thread.join()
        self._thread = None
        result = self._result
        self._result = None
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venvs/audio/bin/pytest tests/test_loop_prefetch.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/loop_prefetch.py tests/test_loop_prefetch.py
git commit -m "feat(streaming_mir): one-slot input-prefetch primitive (WS1)"
```

---

### Task 2: `render_set_stems.py` — `--grid-offset-sec` + `--out-tag` (WS2 pseudo-reference support)

**Files:**
- Modify: `scripts/render_set_stems.py:112-125` (`plan_windows`), `:299-346` (`main` args + work/out dirs)
- Test: `tests/test_render_set_stems_windows.py` (append)

**Interfaces:**
- Produces: `plan_windows(duration, core_sec, overlap_sec, grid_offset_sec: float = 0.0)` — new optional kwarg; default reproduces current behavior exactly. CLI flags `--grid-offset-sec` (float, default 0) and `--out-tag` (str, default "") consumed by Task 6's run protocol.
- `--out-tag X` suffixes both the work dir and the output dir (`<sid>__X`), so three renders of the same set don't collide with the resumable skip-if-parts-exist logic.

- [ ] **Step 1: Write the failing tests (append to existing file)**

```python
# append to tests/test_render_set_stems_windows.py

def test_grid_offset_zero_matches_legacy() -> None:
    legacy = plan_windows(1000.0, 360.0, 10.0)
    offset0 = plan_windows(1000.0, 360.0, 10.0, grid_offset_sec=0.0)
    assert offset0 == legacy


def test_grid_offset_shifts_interior_boundaries() -> None:
    ws = plan_windows(1000.0, 360.0, 10.0, grid_offset_sec=180.0)
    cores = [(w.core_start, w.core_end) for w in ws]
    # first core is the short offset stub, then regular tiling from 180
    assert cores == [(0.0, 180.0), (180.0, 540.0), (540.0, 900.0), (900.0, 1000.0)]
    # cores still tile [0, duration] exactly
    for prev, nxt in zip(ws, ws[1:]):
        assert prev.core_end == nxt.core_start
    assert ws[0].core_start == 0.0 and ws[-1].core_end == 1000.0


def test_grid_offset_windows_padded_and_clamped() -> None:
    ws = plan_windows(1000.0, 360.0, 10.0, grid_offset_sec=180.0)
    assert ws[0].win_start == 0.0  # clamped at mix start
    assert ws[1].win_start == 170.0 and ws[1].win_end == 550.0
    assert ws[-1].win_end == 1000.0  # clamped at mix end
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `venvs/audio/bin/pytest tests/test_render_set_stems_windows.py -v`
Expected: existing tests PASS; the three new ones FAIL with `TypeError: plan_windows() got an unexpected keyword argument 'grid_offset_sec'`

- [ ] **Step 3: Implement `plan_windows` offset + wire the CLI flags**

Replace `plan_windows` (render_set_stems.py:112-125) with:

```python
def plan_windows(
    duration: float,
    core_sec: float,
    overlap_sec: float,
    grid_offset_sec: float = 0.0,
) -> list[Window]:
    """Tile [0, duration] into `core_sec` cores, each padded by `overlap_sec` of
    two-sided context (clamped at the mix edges). overlap_sec=0 reproduces the
    legacy hard-cut behaviour (window == core).

    grid_offset_sec > 0 shifts the tiling grid: the first core is a short
    [0, grid_offset_sec) stub, then regular core_sec tiling. Used by the WS2
    seam validation to build a pseudo-reference whose chunk *interiors* span
    another render's join points (a full-file offline reference is
    VRAM-impossible on a 60-90 min set).
    """
    boundaries: list[float] = [0.0]
    t = grid_offset_sec if grid_offset_sec > 0.0 else core_sec
    while t < duration - 1e-6:
        boundaries.append(t)
        t += core_sec
    boundaries.append(duration)
    return [
        Window(s, e, max(0.0, s - overlap_sec), min(duration, e + overlap_sec))
        for s, e in zip(boundaries, boundaries[1:])
        if e > s
    ]
```

In `main()` add the two args after `--overlap-sec` (render_set_stems.py:312):

```python
    ap.add_argument(
        "--grid-offset-sec",
        type=float,
        default=0.0,
        help="shift the chunk grid by this many seconds (first core is a short "
        "stub). WS2 seam validation only: builds a pseudo-reference whose chunk "
        "interiors cover another render's joins.",
    )
    ap.add_argument(
        "--out-tag",
        default="",
        help="suffix for work+output dirs so multiple renders of one set "
        "(e.g. overlap A/B + pseudo-ref) don't collide with resume logic",
    )
```

Change the dir derivation (render_set_stems.py:343) and the `plan_windows` call (:350):

```python
    dir_key = f"{sid}__{args.out_tag}" if args.out_tag else str(sid)
    work = RENDER_ROOT / dir_key
    ...
    windows = plan_windows(
        total, float(args.chunk_sec), args.overlap_sec, args.grid_offset_sec
    )
```

and the output dir (:411): `out_dir = OUT_ROOT / dir_key`.

Guard: `--out-tag` is required whenever `--grid-offset-sec > 0` unless `--no-push` is set — a pseudo-reference must never overwrite the canonical render or be pushed. Add right after `args = ap.parse_args()`:

```python
    if args.grid_offset_sec > 0 and not args.no_push:
        ap.error("--grid-offset-sec is a WS2 validation mode; it requires --no-push")
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `venvs/audio/bin/pytest tests/test_render_set_stems_windows.py -v`
Expected: all PASS (legacy + 3 new)

- [ ] **Step 5: Commit**

```bash
git add scripts/render_set_stems.py tests/test_render_set_stems_windows.py
git commit -m "feat(render_set_stems): grid-offset pseudo-reference + out-tag renders (WS2 validation)"
```

---

### Task 3: `analysis/pipeline.py` — per-stage timing log line

**Files:**
- Modify: `analysis/pipeline.py:176-309` (`analyze_track`)

**Interfaces:**
- Produces one log line per track, parsed by Task 6's aggregation:
  `TIMING-STAGES track_audio_id=<tid> {"separation": 42.1, "beats": 3.0, "cues": 1.2, "load_lufs": 2.4, "mert": 18.9, "essentia": 6.7}`
- Log-only. No signature, return-type, or persistence changes.

- [ ] **Step 1: Add stage timers**

`analysis/pipeline.py` already imports `numpy`; add at the top imports: `import json` and `import time` (check first — add only what's missing). Then in `analyze_track`, thread a `stage_s: dict[str, float] = {}` through the existing sequential stages by bracketing each with `time.monotonic()`:

```python
    stage_s: dict[str, float] = {}
    _t = time.monotonic()
    stems_r = run_separation(a, audio_path, stems_dir, asset.track_audio_id)
    stage_s["separation"] = time.monotonic() - _t
    if not stems_r.is_ok():
        return stems_r

    _t = time.monotonic()
    beats_r = beat_this_adapter.predict(a.beats, audio_path)
```

…and so on for each stage, closing each bracket right before the next stage's timer opens (grid repair + bpm/measures fold into `"beats"`; `load_mono` + `integrated_lufs` fold into `"load_lufs"`; the essentia `match` block gets `"essentia"` only when it runs). Immediately before the final `return Ok(...)`:

```python
    _log.info(
        "TIMING-STAGES track_audio_id=%s %s",
        asset.track_audio_id,
        json.dumps({k: round(v, 1) for k, v in stage_s.items()}),
    )
```

Bracket placement (exact stage → lines in the current file): separation 197-199, beats 202-222, cues 227-230, load_lufs 233-240, mert 246-254, essentia 271-283.

- [ ] **Step 2: Run the existing analysis tests**

Run: `venvs/audio/bin/pytest tests/test_audio_pipeline_analysis.py -v`
Expected: all PASS (nothing in them touches `analyze_track` end-to-end; this guards the helper functions and imports)

- [ ] **Step 3: Commit**

```bash
git add analysis/pipeline.py
git commit -m "feat(analysis): per-stage TIMING-STAGES log line (WS1 batching go/no-go data)"
```

---

### Task 4: `scripts/vast_loop.py` — prefetch integration + persist-join fix

**Files:**
- Modify: `scripts/vast_loop.py:293-464` (`main`)

**Interfaces:**
- Consumes from Task 1: `PrefetchSlot`, `PrefetchItem`, `PrefetchFailure` via `from scripts.loop_prefetch import ...` — but note vast_loop runs on the Vast box with `sys.path` rooted at `/workspace/tracklist_engine`, so import as `from scripts.loop_prefetch import PrefetchFailure, PrefetchItem, PrefetchSlot` (works both there and on Mac since repo root is on `sys.path` in both).
- New CLI: `--no-prefetch` (A/B baseline, exact legacy behavior), `--max-tracks N` (stop after N analyzed+failed; A/B run sizing).
- New log lines parsed by Task 6: `TIMING tid=<t> prefetched=<0|1> pull_s=<f> analyze_s=<f>` and `TIMING-BG tid=<t> rsync_out_s=<f> push_s=<f>`.

**The two behavior changes (from the spec, plus the bug found during planning):**

1. **Input prefetch** — while track N analyzes, a `PrefetchSlot` picks+pulls N+1.
2. **Persist-join fix** — the current code starts the persist thread at iteration end and joins it at the next iteration's *top*, milliseconds later ([vast_loop.py:398-402](../../scripts/vast_loop.py) vs :443-448), so the documented "~30% rsync-hiding" never actually happens — the loop is serial. The join moves to just before the next hand-off, so persist(N-1) overlaps analyze(N). This is only sound together with in-flight exclusion (below).
3. **In-flight exclusion** — `next_task` picks from canonical, where an in-progress track's rows haven't landed. Every prefetch pick passes `failed_tids ∪ {analyzing tid} ∪ {persisting tid}`. A track whose *push* fails is re-picked once it leaves the in-flight set — same eventual retry as today, one iteration deferred.

- [ ] **Step 1: Add the CLI args**

After the `--shard` arg (vast_loop.py:310-315):

```python
    p.add_argument(
        "--no-prefetch",
        action="store_true",
        help="disable the WS1 input-prefetch thread (serial legacy path; "
        "used as the A/B throughput baseline)",
    )
    p.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="stop after this many tracks (analyzed+failed); A/B run sizing",
    )
```

- [ ] **Step 2: Replace the main loop**

Import at the top of the file (with the other repo imports, after `sys.path.insert`):

```python
from scripts.loop_prefetch import PrefetchFailure, PrefetchItem, PrefetchSlot
```

Replace the block from the WS1 comment (vast_loop.py:370) through the end of `main` with:

```python
    # --- streaming_mir WS1: two single-slot overlaps around the GPU stage ---
    # input:  PrefetchSlot pulls track N+1's audio while N analyzes.
    # output: _persist_in_bg pushes N's stems/rows while N+1 analyzes. (The
    #         pre-WS1 code started this thread at iteration end and joined it
    #         at the next iteration's TOP — milliseconds later — so the
    #         intended rsync-hiding never happened; the join now sits just
    #         before the next hand-off.)
    # Correctness: next_task picks from canonical, where an in-progress
    # track's rows haven't landed, so every prefetch pick skips
    # failed_tids ∪ {analyzing tid} ∪ {persisting tid}. A track whose PUSH
    # fails is re-picked once it leaves the in-flight set — same eventual
    # retry as before, one iteration later.
    bg: threading.Thread | None = None
    persist_tid: int | None = None

    def _persist_in_bg(tid: int, stem_local: Path) -> None:
        """Owns the cleanup of stem_local once handed off."""
        try:
            t_r = time.monotonic()
            if stem_local.exists():
                log.info("[%d] (bg) pushing stems", tid)
                rsync_stems_out(stem_local, tid)
            rsync_out_s = time.monotonic() - t_r
            log.info("[%d] (bg) pushing DB rows to canonical", tid)
            t_p = time.monotonic()
            push_track_rows(tid)
            log.info(
                "TIMING-BG tid=%d rsync_out_s=%.1f push_s=%.1f",
                tid,
                rsync_out_s,
                time.monotonic() - t_p,
            )
            log.info("[%d] (bg) DONE", tid)
        except subprocess.CalledProcessError as e:
            # Canonical never received track_analysis, so next_task re-picks
            # this tid once it leaves the in-flight set. Annoying, but safe.
            log.error("[%d] (bg) push failed: %s", tid, e)
        finally:
            if stem_local.exists():
                shutil.rmtree(stem_local, ignore_errors=True)

    def _pick(skip: frozenset[int]) -> tuple[int, str] | None:
        return next_task(skip, sets=active_sets, shard=shard)

    def _pull(tid: int, remote_path: str) -> Path:
        local = LOCAL_AUDIO / f"{tid}.m4a"
        rsync_in(remote_path, local)
        return local

    prefetch = (
        None
        if args.no_prefetch
        else PrefetchSlot(pick=_pick, pull=_pull, hydrate=fetch_asset)
    )

    def _sync_item(
        skip: frozenset[int],
    ) -> PrefetchItem | PrefetchFailure | None:
        """Serial pick+pull+hydrate for --no-prefetch, the first iteration,
        and post-failure refills. Same shape as PrefetchSlot.take()."""
        picked = _pick(skip)
        if picked is None:
            return None
        tid, remote_path = picked
        try:
            t0 = time.monotonic()
            local = _pull(tid, remote_path)
            pull_s = time.monotonic() - t0
            asset = fetch_asset(tid, local)
        except (subprocess.CalledProcessError, Exception) as exc:
            return PrefetchFailure(tid=tid, detail=str(exc))
        return PrefetchItem(tid=tid, local_audio=local, asset=asset, pull_s=pull_s)

    while True:
        if args.max_tracks is not None and n_done + n_failed >= args.max_tracks:
            log.info(
                "--max-tracks %d reached — analyzed %d, failed %d",
                args.max_tracks,
                n_done,
                n_failed,
            )
            if bg is not None:
                bg.join()
            return 0

        # ---- obtain this iteration's item ----
        inflight = frozenset(t for t in (persist_tid,) if t is not None)
        if prefetch is None:
            # Legacy serial path (A/B baseline): join the persist tail FIRST,
            # exactly like the pre-WS1 loop, then pick+pull inline.
            if bg is not None:
                bg.join()
                bg = None
                persist_tid = None
            item = _sync_item(frozenset(failed_tids))
        elif prefetch.pending:
            item = prefetch.take()
        else:
            # First iteration / refill after a failure path.
            item = _sync_item(frozenset(failed_tids) | inflight)

        if item is None:
            # Drained *for the current skip set*. A persisting track whose
            # push failed still needs a re-pick: join the tail, then re-check
            # once with only failed_tids excluded.
            if bg is not None:
                bg.join()
                bg = None
                persist_tid = None
            item = _sync_item(frozenset(failed_tids))
            if item is None:
                log.info("queue drained — analyzed %d, failed %d", n_done, n_failed)
                return 0

        if isinstance(item, PrefetchFailure):
            log.error("[%d] input pull failed: %s", item.tid, item.detail)
            n_failed += 1
            failed_tids.add(item.tid)
            continue

        tid = item.tid
        local_audio = item.local_audio
        handed_off = False

        # Kick off the NEXT prefetch now — it downloads behind this track's
        # GPU work. Skip set: failures + this tid + the persisting tid.
        if prefetch is not None:
            prefetch.start(
                frozenset(failed_tids)
                | frozenset({tid})
                | frozenset(t for t in (persist_tid,) if t is not None)
            )

        try:
            log.info("[%d] analyzing %s (%s)", tid, item.asset.track_id, item.asset.platform)
            t1 = time.monotonic()
            r = analyze_track(a, item.asset, stems_dir=LOCAL_STEMS)
            analyze_s = time.monotonic() - t1
            if not r.is_ok():
                log.warning(
                    "[%d] analyze_track failed: %s — %s",
                    tid,
                    r.error.kind,
                    r.error.detail,
                )
                n_failed += 1
                failed_tids.add(tid)
                continue
            log.info(
                "TIMING tid=%d prefetched=%d pull_s=%.1f analyze_s=%.1f",
                tid,
                int(prefetch is not None),
                item.pull_s,
                analyze_s,
            )

            p_r = persistence.persist_analysis(SCRATCH_DB, r.value)
            if not p_r.is_ok():
                log.warning("[%d] persist failed: %s", tid, p_r.error.detail)
                n_failed += 1
                failed_tids.add(tid)
                continue

            # Hand off rsync + push_track_rows. Single slot: join the
            # PREVIOUS persist here (not at loop top) so it overlapped this
            # track's analyze.
            if bg is not None:
                bg.join()
            stem_local = LOCAL_STEMS / str(tid)
            bg = threading.Thread(
                target=_persist_in_bg, args=(tid, stem_local), daemon=False
            )
            persist_tid = tid
            bg.start()
            handed_off = True
            n_done += 1
            log.info("[%d] handed off (n_done=%d, n_failed=%d)", tid, n_done, n_failed)
        except subprocess.CalledProcessError as e:
            log.error("[%d] subprocess failed: %s", tid, e)
            n_failed += 1
            failed_tids.add(tid)
        finally:
            if local_audio.exists():
                local_audio.unlink()
            # If hand-off succeeded, the bg thread owns stem_local cleanup.
            if not handed_off:
                stem_local = LOCAL_STEMS / str(tid)
                if stem_local.exists():
                    shutil.rmtree(stem_local, ignore_errors=True)
```

Notes for the implementer:

- The old `main` body's `while True:` block (vast_loop.py:398-464) and the old
  `_persist_in_bg` + `bg` declaration (:370-396) are replaced wholesale by the above.
  Everything before (argparse, `load_analyzers`, orphan wipe, `failed_tids`) is unchanged
  except renaming the persist result variable `p` → `p_r` (the argparse parser already
  owns the name `p` at :296).
- `fetch_asset(tid, local)` already has the exact `(int, Path) -> AudioAsset` shape
  `hydrate` needs — pass it directly.
- The rename of the analyze log line drops the old separate "pulling" log for the
  prefetched case (the pull happens inside the slot); `_pull` keeps no log — the TIMING
  line carries `pull_s`.

- [ ] **Step 3: Sanity-check compile + full test suite**

Run: `venvs/audio/bin/python -m py_compile scripts/vast_loop.py && venvs/audio/bin/pytest -q`
Expected: compiles; full suite green (vast_loop has no direct tests; the prefetch semantics are covered by Task 1's tests)

- [ ] **Step 4: Run `make check`**

Run: `make check`
Expected: guardrails OK, mypy OK, pytest subset green

- [ ] **Step 5: Commit**

```bash
git add scripts/vast_loop.py
git commit -m "feat(vast_loop): WS1 input prefetch + fix persist overlap (join moved off loop top)"
```

---

### Task 5: `workspaces/streaming_mir/seam_check.py` — boundary SDR vs pseudo-reference

**Files:**
- Create: `workspaces/streaming_mir/seam_check.py`

**Interfaces:**
- CLI: `seam_check.py --render-a DIR --render-b DIR --pseudo-ref DIR [--chunk-sec 360] [--win-sec 2.0] [--snippets-out DIR]` where each DIR contains `vocals.flac` + `instrumental.flac` (render_set_stems output dirs).
- Prints a per-stem table: join-region SDR and interior-control SDR for renders A and B, each vs the pseudo-reference; optional worst-join snippet export for ear checks.

- [ ] **Step 1: Write the script**

```python
# workspaces/streaming_mir/seam_check.py
"""WS2 real-set seam validation: score two set-stem renders against a
shifted-grid pseudo-reference.

A 60-90 min set has no full-file offline reference (VRAM-impossible), so the
reference is a THIRD render whose chunk grid is offset by half a chunk
(render_set_stems.py --grid-offset-sec): its chunk interiors span the other
renders' join points, so near a join it plays the role full-file offline
played in block_overlap_sweep.py. Metric: SDR restricted to ±win_sec of each
join (boundary-local), with chunk-midpoint windows as the interior control.

Success criterion (spec): render B (overlap 10) boundary SDR within 0.5 dB of
its interior control; render A (overlap 0) shows the seam gap.

sdr() is copied from block_overlap_sweep.py rather than imported — importing
that module drags the separator adapters in; this script needs only
numpy/soundfile.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

STEMS = ("vocals", "instrumental")


def sdr(ref: np.ndarray, est: np.ndarray) -> float:
    """Global SDR in dB. Trims to common length; flattens channels."""
    n = min(len(ref), len(est))
    r = ref[:n].reshape(-1).astype(np.float64)
    e = est[:n].reshape(-1).astype(np.float64)
    num = float(np.sum(r * r)) + 1e-12
    den = float(np.sum((r - e) ** 2)) + 1e-12
    return 10.0 * np.log10(num / den)


def sdr_windows(
    ref: np.ndarray, est: np.ndarray, centers: list[int], half_win: int
) -> float:
    """SDR restricted to ±half_win samples around each center."""
    n = min(len(ref), len(est))
    mask = np.zeros(n, dtype=bool)
    for c in centers:
        if 0 <= c < n:
            mask[max(0, c - half_win) : min(n, c + half_win)] = True
    if not mask.any():
        return float("nan")
    return sdr(ref[:n][mask], est[:n][mask])


def per_join_sdr(
    ref: np.ndarray, est: np.ndarray, centers: list[int], half_win: int
) -> list[tuple[int, float]]:
    return [(c, sdr_windows(ref, est, [c], half_win)) for c in centers]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-a", type=Path, required=True, help="overlap-0 render dir")
    ap.add_argument("--render-b", type=Path, required=True, help="overlap-10 render dir")
    ap.add_argument("--pseudo-ref", type=Path, required=True, help="grid-offset render dir")
    ap.add_argument("--chunk-sec", type=float, default=360.0)
    ap.add_argument("--win-sec", type=float, default=2.0)
    ap.add_argument(
        "--snippets-out",
        type=Path,
        default=None,
        help="export 10s wavs around each render's worst join for ear checks",
    )
    args = ap.parse_args()

    for stem in STEMS:
        ref, sr = sf.read(args.pseudo_ref / f"{stem}.flac", always_2d=True)
        a, sr_a = sf.read(args.render_a / f"{stem}.flac", always_2d=True)
        b, sr_b = sf.read(args.render_b / f"{stem}.flac", always_2d=True)
        assert sr == sr_a == sr_b, f"sample-rate mismatch on {stem}"

        hop = int(args.chunk_sec * sr)
        n = min(len(ref), len(a), len(b))
        joins = list(range(hop, n, hop))
        mids = [j - hop // 2 for j in joins]  # interior control windows
        half = int(args.win_sec * sr)

        print(f"\n== {stem} ({len(joins)} joins, ±{args.win_sec}s windows) ==")
        for name, est in (("A(ovl=0)", a), ("B(ovl=10)", b)):
            j_sdr = sdr_windows(ref, est, joins, half)
            i_sdr = sdr_windows(ref, est, mids, half)
            print(
                f"  {name:10s} join={j_sdr:7.2f} dB  interior={i_sdr:7.2f} dB  "
                f"gap={i_sdr - j_sdr:+.2f} dB"
            )
            per = per_join_sdr(ref, est, joins, half)
            worst = min(per, key=lambda p: p[1])
            print(f"             worst join @ {worst[0] / sr:7.1f}s = {worst[1]:.2f} dB")
            if args.snippets_out is not None:
                args.snippets_out.mkdir(parents=True, exist_ok=True)
                c = worst[0]
                snip = est[max(0, c - 5 * sr) : c + 5 * sr]
                sf.write(
                    args.snippets_out / f"{stem}_{name.split('(')[0]}_worst_join.wav",
                    snip,
                    sr,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke-test on synthetic audio (no GPU needed)**

```bash
venvs/audio/bin/python - <<'EOF'
import numpy as np, soundfile as sf
from pathlib import Path
sr = 44100
rng = np.random.default_rng(0)
base = rng.standard_normal((sr * 30, 2)) * 0.1
for d in ("ra", "rb", "pr"):
    Path(f"/private/tmp/claude-501/-Users-johnnycabrahams-Desktop-tracklist-engine/114707a2-f1b5-490e-aa75-203f6c039603/scratchpad/seam/{d}").mkdir(parents=True, exist_ok=True)
root = Path("/private/tmp/claude-501/-Users-johnnycabrahams-Desktop-tracklist-engine/114707a2-f1b5-490e-aa75-203f6c039603/scratchpad/seam")
a = base.copy(); a[sr*10-100:sr*10+100] += 0.05  # seam damage at the 10s join
for stem in ("vocals", "instrumental"):
    sf.write(root/"ra"/f"{stem}.flac", a, sr)
    sf.write(root/"rb"/f"{stem}.flac", base, sr)
    sf.write(root/"pr"/f"{stem}.flac", base, sr)
EOF
venvs/audio/bin/python workspaces/streaming_mir/seam_check.py \
  --render-a /private/tmp/claude-501/-Users-johnnycabrahams-Desktop-tracklist-engine/114707a2-f1b5-490e-aa75-203f6c039603/scratchpad/seam/ra \
  --render-b /private/tmp/claude-501/-Users-johnnycabrahams-Desktop-tracklist-engine/114707a2-f1b5-490e-aa75-203f6c039603/scratchpad/seam/rb \
  --pseudo-ref /private/tmp/claude-501/-Users-johnnycabrahams-Desktop-tracklist-engine/114707a2-f1b5-490e-aa75-203f6c039603/scratchpad/seam/pr \
  --chunk-sec 10 --win-sec 1
```

Expected: render A shows join SDR ≈ low double digits with interior ≫ join; render B shows join ≈ interior (both huge, identical signals).

- [ ] **Step 3: Commit**

```bash
git add workspaces/streaming_mir/seam_check.py
git commit -m "feat(streaming_mir): seam_check scorer — boundary SDR vs shifted-grid pseudo-ref (WS2)"
```

---

### Task 6: Vast rental — WS1 A/B + WS2 before/after, results into the brief

No code; operational protocol. All numbers land in `workspaces/streaming_mir/RESEARCH_BRIEF.md`.

- [ ] **Step 1: Pick the A/B workload (pi-storage query)**

```bash
ssh pi-storage 'sqlite3 -separator "|" /mnt/storage/data/db/music_database.db "
SELECT sts.set_id, COUNT(DISTINCT ta.track_audio_id)
FROM track_audio ta
JOIN set_track_slots sts ON sts.track_id = ta.track_id
LEFT JOIN track_analysis tan ON tan.track_audio_id = ta.track_audio_id
WHERE tan.track_audio_id IS NULL
GROUP BY sts.set_id ORDER BY 2 DESC LIMIT 8"'
```

Pick a set (or two) with ≥70 pending tracks → `--set-ids` value. Also fetch the WS2 mix id: `SELECT set_audio_id FROM set_audio WHERE set_id IN ('2nvzlh2k','1fsnxchk')` and check `set_stems` — if canonical set stems already exist, all three WS2 renders use `--no-push` + `--out-tag`.

- [ ] **Step 2: Rent + bootstrap the box**

Follow the memory recipes (`project_vastai_instance_choice` filter + curl API from `project_vast_access`; coordination: list instances first, destroy only what you create). Rent a 4090-class spot. Since the WS1 branch isn't on `main`, rsync the working tree instead of cloning:

```bash
rsync -az --exclude venvs --exclude data --exclude _mac_scratch --exclude .git \
  ~/Desktop/tracklist_engine/ root@<vast>:/workspace/tracklist_engine/
ssh root@<vast> 'SKIP_CLONE=1 bash /workspace/tracklist_engine/scripts/vast_bootstrap.sh'
```

(`vast_bootstrap.sh` supports `SKIP_CLONE=1` for exactly this.) Verify pi-storage reachability from the box (`ssh pi-storage true`) before starting.

- [ ] **Step 3: WS1 A/B runs**

```bash
# baseline (legacy serial semantics)
tmux new -d -s ab_base '/venv/main/bin/python /workspace/tracklist_engine/scripts/vast_loop.py \
  --separator roformer --set-ids <SET_IDS> --no-prefetch --max-tracks 30 \
  2>&1 | tee /workspace/ab_baseline.log'
# after it exits:
tmux new -d -s ab_pref '/venv/main/bin/python /workspace/tracklist_engine/scripts/vast_loop.py \
  --separator roformer --set-ids <SET_IDS> --max-tracks 30 \
  2>&1 | tee /workspace/ab_prefetch.log'
```

Aggregate (run on the box or after scp'ing logs):

```bash
for f in /workspace/ab_baseline.log /workspace/ab_prefetch.log; do
  echo "== $f"
  grep -o 'TIMING tid=.*' "$f" | awk '{for(i=1;i<=NF;i++){split($i,kv,"=");v[kv[1]]+=kv[2];n[kv[1]]++}}
    END{for(k in v) if (k!="TIMING") printf "%s mean=%.1f n=%d\n", k, v[k]/n[k], n[k]}'
  # wall per track: timestamps of successive "handed off" lines
  grep 'handed off' "$f" | awk '{print $1" "$2}'
done
```

Report: mean per-track wall (baseline vs prefetch), mean `pull_s`, `analyze_s`, `TIMING-BG` rsync/push means, and the `TIMING-STAGES` breakdown table (mean seconds per stage over ≥30 tracks).

- [ ] **Step 4: WS2 renders + seam check (same box)**

```bash
V=/venv/main/bin/python; R=/workspace/tracklist_engine/scripts/render_set_stems.py
$V $R --set-audio-id <SID> --separator roformer --device cuda --no-push --out-tag ovl0  --overlap-sec 0
$V $R --set-audio-id <SID> --separator roformer --device cuda --no-push --out-tag ovl10 --overlap-sec 10
$V $R --set-audio-id <SID> --separator roformer --device cuda --no-push --out-tag pref  --overlap-sec 10 --grid-offset-sec 180
$V /workspace/tracklist_engine/workspaces/streaming_mir/seam_check.py \
  --render-a  /workspace/tracklist_engine/_mac_scratch/set_stems/set/<SID>__ovl0 \
  --render-b  /workspace/tracklist_engine/_mac_scratch/set_stems/set/<SID>__ovl10 \
  --pseudo-ref /workspace/tracklist_engine/_mac_scratch/set_stems/set/<SID>__pref \
  --snippets-out /workspace/seam_snippets
```

scp `/workspace/seam_snippets/*.wav` to the Mac and ear-check the worst joins.

- [ ] **Step 5: Record results + close out**

- Append a dated "WS1 A/B RESULT" + "WS2 REAL-SET RESULT" section to
  `workspaces/streaming_mir/RESEARCH_BRIEF.md`: the wall-clock table, the per-stage
  table, the explicit **batching go/no-go verdict** (go if separation+MERT dominate and
  per-forward utilization is low; state the number), and the WS2 join/interior SDRs +
  ear verdict. Mark WS2 **CLOSED** if the success criterion holds
  (B join ≥ B interior − 0.5 dB, no audible seam).
- **Destroy the Vast instance** (only the one created here).
- Commit: `git add workspaces/streaming_mir/RESEARCH_BRIEF.md && git commit -m "results(streaming_mir): WS1 A/B + WS2 real-set validation on Vast"`
- Push the branch; if all green, merge the WS1 commits to `main` so pi/Vast deploys pick them up.

---

## Self-review notes

- **Spec coverage:** prefetch (T1+T4), `--no-prefetch` A/B flag (T4), in-flight exclusion (T1+T4), instrumentation/batching-data (T3+T6), WS2 pseudo-reference + before/after (T2+T5+T6), success criteria + brief write-up (T6). The persist-join fix (T4) exceeds spec — it was found during planning and is required for the prefetch's in-flight machinery anyway.
- **Type consistency:** `PrefetchSlot(pick, pull, hydrate)` signatures match `_pick`/`_pull`/`fetch_asset` usage in T4; `plan_windows(..., grid_offset_sec=0.0)` default keeps the existing tests and `block_overlap_sweep` unaffected (it has its own tiling).
- **Placeholder scan:** none — every code step carries complete code; T6 placeholders (`<SET_IDS>`, `<SID>`, `<vast>`) are runtime values produced by T6 Step 1/2 themselves.
