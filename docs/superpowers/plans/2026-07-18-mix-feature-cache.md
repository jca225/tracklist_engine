# Streaming Mix-Feature (Fingerprint) Cache — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Precompute + persist the per-mix landmark fingerprint via memory-bounded streaming, so the corpus harvest reads a compact cached fingerprint instead of recomputing the full-mix STFT (removing the RAM wall + the ~10 min/set recompute).

**Architecture:** A streaming fingerprint builder chunks the mix and merges the `{hash: times}` dicts (near-lossless because a fingerprint IS a hash→times map). A tiny file-cache store persists each mix's fingerprint blob. The harvest's per-set `MixFeatureCache` gets a cache-backed `compute_mix_fp` injected through `real_probe_scorer`. A batch driver warms the cache ahead of time.

**Tech Stack:** Python 3.13, numpy, librosa, soundfile; pytest. Run from repo root with `venvs/audio/bin/python`. Work in the worktree `/Users/johnnycabrahams/Desktop/tracklist_engine/.claude/worktrees/cotrain-accept-precision` on branch `cotrain-corpus-harvest`.

## Global Constraints

- **Style:** `from __future__ import annotations`; full type hints; frozen dataclasses for records; pure functions with IO at the edges. Match the surrounding files.
- **No DSP reimplementation:** reuse `constellation` + `hashes` + `LandmarkFingerprint` from `landmark_fp.py` unchanged. This is a caching/perf layer, NOT a new probe (does not touch the alignment_prototype "sensor phase is closed" freeze).
- **Fingerprint internals (verbatim):** `SR = 22050`, `FHOP = 512`, `FPS = SR / FHOP ≈ 43.07`, `hashes(tf, fb, fan=8, dt_max=80)` — the pairing window is `dt_max = 80` frames ≈ 1.86 s, so chunk **overlap must be ≥ `dt_max` frames**; default `overlap_s = 3.0`.
- **`LandmarkFingerprint`** = `dataclass(frozen=True)` with `fps: float`, `duration_s: float`, `hashes: dict[tuple[int,int,int], tuple[int,...]]`; serialize with `.to_blob() -> bytes` / `LandmarkFingerprint.from_blob(bytes)`.
- **Cache key** = `str(set_audio_id)`; **cache file** = `{cache_root}/{key}.fp` (raw `to_blob()` bytes). Default cache_root `/mnt/storage/data/mix_fp_cache/`.
- **Backward compatibility:** with no `--mix-fp-cache`, `corpus_harvest` behavior is byte-unchanged (live per-run fingerprint).
- **Commit:** use `git commit --no-verify` ONLY for the known gitignored `workspaces/msst_webui/` entropy noise; end messages with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- **Tests:** pytest, in `workspaces/pws_aligner/tests/` and `workspaces/alignment_prototype/` test locations mirroring existing tests. Run: `venvs/audio/bin/python -m pytest <path> -v`.

## File structure

- `workspaces/alignment_prototype/landmark_fp.py` — **add** `fingerprint_from_file_streaming` (Task 1).
- `workspaces/pws_aligner/mix_fp_store.py` — **new**, `load_or_build` file cache (Task 2).
- `workspaces/pws_aligner/cotrain_seam.py` + `workspaces/pws_aligner/corpus_harvest.py` — **modify** to thread a cache-backed `compute_mix_fp` (Task 3).
- `scripts/cache_mix_fingerprints.py` — **new** precompute driver (Task 4).
- Tests: `workspaces/alignment_prototype/tests/test_fingerprint_streaming.py`, `workspaces/pws_aligner/tests/test_mix_fp_store.py`, extend `workspaces/pws_aligner/tests/test_corpus_harvest.py`, `tests/test_cache_mix_fingerprints.py` (or alongside).

---

### Task 1: `fingerprint_from_file_streaming` — memory-bounded fingerprint

**Files:**
- Modify: `workspaces/alignment_prototype/landmark_fp.py` (add function after `fingerprint_from_audio`)
- Test: `workspaces/alignment_prototype/tests/test_fingerprint_streaming.py` (create)

**Interfaces:**
- Consumes: `constellation`, `hashes`, `LandmarkFingerprint`, `SR`, `FPS` (same module).
- Produces:
  ```python
  def fingerprint_from_file_streaming(
      path: str | Path, *, chunk_s: float = 120.0, overlap_s: float = 3.0
  ) -> LandmarkFingerprint
  ```

- [ ] **Step 1: Write the failing tests**

Create `workspaces/alignment_prototype/tests/test_fingerprint_streaming.py`:

```python
from __future__ import annotations

import numpy as np
import soundfile as sf

from workspaces.alignment_prototype.landmark_fp import (
    SR,
    fingerprint_from_audio,
    fingerprint_from_file_streaming,
)


def _synth(seconds: float = 40.0) -> np.ndarray:
    # A few stepping tones so constellation finds many landmarks across time.
    t = np.arange(int(seconds * SR)) / SR
    y = np.zeros_like(t)
    for i, f in enumerate([220.0, 440.0, 660.0, 880.0, 330.0, 550.0]):
        seg = (t >= i * seconds / 6) & (t < (i + 1) * seconds / 6)
        y[seg] += np.sin(2 * np.pi * f * t[seg])
    y += 0.01 * np.sin(2 * np.pi * 1000.0 * t)  # broadband-ish content
    return y.astype(np.float32)


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def test_streaming_matches_full_signal_fingerprint(tmp_path):
    y = _synth(40.0)
    wav = tmp_path / "m.wav"
    sf.write(wav, y, SR)
    full = fingerprint_from_audio(y)
    # chunk_s=15 + overlap 3 → ~3 chunks over the 40s signal
    streamed = fingerprint_from_file_streaming(wav, chunk_s=15.0, overlap_s=3.0)
    # near-lossless: the interior of each chunk reproduces full-signal hashes; only
    # a few STFT-edge frames per boundary differ. Require high key overlap.
    j = _jaccard(set(full.hashes), set(streamed.hashes))
    assert j > 0.9, f"streaming keys Jaccard {j:.3f} too low"
    assert abs(streamed.duration_s - full.duration_s) < 0.5


def test_overlap_recovers_boundary_pairs(tmp_path):
    y = _synth(40.0)
    wav = tmp_path / "m.wav"
    sf.write(wav, y, SR)
    full_keys = set(fingerprint_from_audio(y).hashes)
    with_ovl = set(fingerprint_from_file_streaming(wav, chunk_s=15.0, overlap_s=3.0).hashes)
    no_ovl = set(fingerprint_from_file_streaming(wav, chunk_s=15.0, overlap_s=0.0).hashes)
    # overlap must recover at least as many true keys as no-overlap
    assert _jaccard(full_keys, with_ovl) >= _jaccard(full_keys, no_ovl)


def test_streaming_never_loads_whole_signal(monkeypatch, tmp_path):
    y = _synth(40.0)
    wav = tmp_path / "m.wav"
    sf.write(wav, y, SR)
    import librosa
    from workspaces.alignment_prototype import landmark_fp

    seen = []
    real_load = librosa.load

    def spy_load(*args, **kwargs):
        seen.append(kwargs.get("duration"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(landmark_fp.librosa if hasattr(landmark_fp, "librosa") else librosa, "load", spy_load, raising=False)
    monkeypatch.setattr(librosa, "load", spy_load)
    fingerprint_from_file_streaming(wav, chunk_s=10.0, overlap_s=3.0)
    # every load bounded a single chunk window; NONE loaded the whole 40s
    assert seen, "expected chunked loads"
    assert all(d is not None and d <= 10.0 + 3.0 + 1e-6 for d in seen)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venvs/audio/bin/python -m pytest workspaces/alignment_prototype/tests/test_fingerprint_streaming.py -v`
Expected: FAIL — `ImportError: cannot import name 'fingerprint_from_file_streaming'`.

- [ ] **Step 3: Implement the streaming builder**

In `workspaces/alignment_prototype/landmark_fp.py`, after `fingerprint_from_audio`:

```python
def fingerprint_from_file_streaming(
    path: str | Path, *, chunk_s: float = 120.0, overlap_s: float = 3.0
) -> LandmarkFingerprint:
    """Memory-bounded ``fingerprint_from_audio``: fingerprint the file in
    ``chunk_s``-second windows (each loaded with ``overlap_s`` extra so landmark
    pairs straddling a boundary — up to ``dt_max``≈1.86 s — form inside a chunk),
    offset each chunk's anchor times by the chunk start, and merge the hash dicts.

    Because a fingerprint is ``{hash: (anchor_time_frames, ...)}``, this reproduces
    ``fingerprint_from_audio`` except for a few STFT-edge frames per boundary. Peak
    RAM is ONE chunk's audio + STFT, not the whole signal.
    """
    import librosa

    step = float(chunk_s)
    window = step + float(overlap_s)
    merged: dict[tuple[int, int, int], set[int]] = {}
    total_dur = 0.0
    t0 = 0.0
    while True:
        y, _ = librosa.load(
            str(path), sr=SR, mono=True, offset=t0, duration=window
        )
        if y.size == 0:
            break
        chunk_dur = y.size / SR
        total_dur = max(total_dur, t0 + chunk_dur)
        tf, fb = constellation(y)
        if tf.size:
            frame_off = int(round(t0 * FPS))
            for key, times in hashes(tf, fb).items():
                bucket = merged.setdefault(key, set())
                for tt in times:
                    bucket.add(int(tt) + frame_off)
        if chunk_dur < window:  # file ended within this window
            break
        t0 += step
    hashes_out = {k: tuple(sorted(v)) for k, v in merged.items()}
    return LandmarkFingerprint(fps=FPS, duration_s=total_dur, hashes=hashes_out)
```

Add `from pathlib import Path` to the imports if not present (the module uses `str | Path` — check the top; `landmark_fp.py` imports numpy/json; add `from pathlib import Path` under `from typing import Any`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `venvs/audio/bin/python -m pytest workspaces/alignment_prototype/tests/test_fingerprint_streaming.py -v`
Expected: PASS (3 tests). If `test_streaming_matches_full_signal_fingerprint` Jaccard is marginally below 0.9 on this synthetic, raise `overlap_s` in the test to 5.0 and re-run — do NOT lower the assertion below 0.85 without noting why in the report.

- [ ] **Step 5: Commit**

```bash
git add workspaces/alignment_prototype/landmark_fp.py workspaces/alignment_prototype/tests/test_fingerprint_streaming.py
git commit --no-verify -m "feat(fp): fingerprint_from_file_streaming — memory-bounded chunked fingerprint

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `mix_fp_store.load_or_build` — persistent file cache

**Files:**
- Create: `workspaces/pws_aligner/mix_fp_store.py`
- Test: `workspaces/pws_aligner/tests/test_mix_fp_store.py`

**Interfaces:**
- Consumes: `LandmarkFingerprint` (to_blob/from_blob), `fingerprint_from_file_streaming` (Task 1).
- Produces:
  ```python
  def load_or_build(cache_root: str | Path, key: str, mix_path: str | Path,
                    *, chunk_s: float = 120.0, overlap_s: float = 3.0,
                    build: Callable[..., LandmarkFingerprint] | None = None,
                    ) -> LandmarkFingerprint
  ```
  `build` is injectable for tests (defaults to `fingerprint_from_file_streaming`).

- [ ] **Step 1: Write the failing tests**

Create `workspaces/pws_aligner/tests/test_mix_fp_store.py`:

```python
from __future__ import annotations

from workspaces.alignment_prototype.landmark_fp import LandmarkFingerprint
from workspaces.pws_aligner.mix_fp_store import load_or_build

_FP = LandmarkFingerprint(fps=43.0, duration_s=12.0, hashes={(1, 2, 3): (10, 20)})


def test_builds_then_persists_and_reads_without_rebuilding(tmp_path):
    calls = {"n": 0}

    def fake_build(mix_path, **kw):
        calls["n"] += 1
        return _FP

    fp1 = load_or_build(tmp_path, "77", "/mix.m4a", build=fake_build)
    assert fp1.hashes == _FP.hashes
    assert (tmp_path / "77.fp").is_file()
    # second call reads the cache — build NOT invoked again
    fp2 = load_or_build(tmp_path, "77", "/mix.m4a", build=fake_build)
    assert fp2.hashes == _FP.hashes
    assert calls["n"] == 1


def test_corrupt_blob_triggers_rebuild(tmp_path):
    (tmp_path / "9.fp").write_bytes(b"not a valid blob")
    calls = {"n": 0}

    def fake_build(mix_path, **kw):
        calls["n"] += 1
        return _FP

    fp = load_or_build(tmp_path, "9", "/mix.m4a", build=fake_build)
    assert fp.hashes == _FP.hashes
    assert calls["n"] == 1  # rebuilt over the corrupt file


def test_atomic_no_tmp_left_behind(tmp_path):
    load_or_build(tmp_path, "5", "/mix.m4a", build=lambda p, **kw: _FP)
    assert not list(tmp_path.glob("*.tmp"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venvs/audio/bin/python -m pytest workspaces/pws_aligner/tests/test_mix_fp_store.py -v`
Expected: FAIL — `ModuleNotFoundError: workspaces.pws_aligner.mix_fp_store`.

- [ ] **Step 3: Implement the store**

Create `workspaces/pws_aligner/mix_fp_store.py`:

```python
"""Persistent per-mix landmark-fingerprint file cache.

One compact ``{cache_root}/{key}.fp`` blob per mix (``key`` = set_audio_id).
Removes the ~10 min/set live recompute: the corpus harvest reads the cached
fingerprint instead of running the full-mix STFT. Builds are memory-bounded
(streaming) so warming the cache is safe even on the pis.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from workspaces.alignment_prototype.landmark_fp import (
    LandmarkFingerprint,
    fingerprint_from_file_streaming,
)


def load_or_build(
    cache_root: str | Path,
    key: str,
    mix_path: str | Path,
    *,
    chunk_s: float = 120.0,
    overlap_s: float = 3.0,
    build: Callable[..., LandmarkFingerprint] | None = None,
) -> LandmarkFingerprint:
    """Return the fingerprint for ``mix_path``, reading ``{cache_root}/{key}.fp``
    if present (rebuilding on a corrupt blob), else building (streaming) and
    persisting atomically.
    """
    root = Path(cache_root)
    cache_file = root / f"{key}.fp"
    if cache_file.is_file() and cache_file.stat().st_size > 0:
        try:
            return LandmarkFingerprint.from_blob(cache_file.read_bytes())
        except Exception:
            pass  # corrupt/incompatible → rebuild below

    builder = build or fingerprint_from_file_streaming
    fp = builder(mix_path, chunk_s=chunk_s, overlap_s=overlap_s)

    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f"{key}.fp.tmp"
    tmp.write_bytes(fp.to_blob())
    os.replace(tmp, cache_file)
    return fp
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venvs/audio/bin/python -m pytest workspaces/pws_aligner/tests/test_mix_fp_store.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add workspaces/pws_aligner/mix_fp_store.py workspaces/pws_aligner/tests/test_mix_fp_store.py
git commit --no-verify -m "feat(harvest): mix_fp_store.load_or_build — persistent fingerprint file cache

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Wire the cache into the harvest scorer

**Files:**
- Modify: `workspaces/pws_aligner/cotrain_seam.py` (`real_probe_scorer` ~line 588–641)
- Modify: `workspaces/pws_aligner/corpus_harvest.py` (`ScorerFactory`, `_default_scorer_factory`, `run_corpus_harvest`, `main`)
- Test: extend `workspaces/pws_aligner/tests/test_corpus_harvest.py`

**Interfaces:**
- Consumes: `mix_fp_store.load_or_build` (Task 2); `MixFeatureCache` (existing).
- Produces:
  - `real_probe_scorer(..., compute_mix_fp: Callable[[object], object] | None = None)` → passes it to `MixFeatureCache(compute_mix_fp=compute_mix_fp)`.
  - `ScorerFactory = Callable[[Path, Path, "object | None"], RefMixScorer]` (3rd arg = `compute_mix_fp`).
  - `run_corpus_harvest(..., mix_fp_cache_root: Path | None = None)`.
  - `corpus_harvest --mix-fp-cache <root>`.

- [ ] **Step 1: Write the failing test**

Add to `workspaces/pws_aligner/tests/test_corpus_harvest.py`:

```python
def test_harvest_uses_cached_fp_when_cache_root_set(tmp_path):
    # Pre-warm a cache file for set_audio_id 77; harvest must read it, not build live.
    from workspaces.alignment_prototype.landmark_fp import LandmarkFingerprint
    from workspaces.pws_aligner import mix_fp_store

    cache_root = tmp_path / "fpcache"
    cache_root.mkdir()
    (cache_root / "77.fp").write_bytes(
        LandmarkFingerprint(fps=43.0, duration_s=10.0, hashes={(1, 1, 1): (5,)}).to_blob()
    )
    builds = {"n": 0}

    def spy_build(mix_path, **kw):
        builds["n"] += 1
        return LandmarkFingerprint(fps=43.0, duration_s=1.0, hashes={})

    # capture the compute_mix_fp the factory receives and exercise it
    seen = {}

    def factory(mix_full_path, mix_stem_dir, compute_mix_fp=None):
        seen["fn"] = compute_mix_fp
        return lambda cand, span: _agree(cand.recording_id, ("fp", "chroma"))

    slots = [_slot(set_id="S", set_audio_id=77, recording_id="R1", cue_time_s=100.0)]
    out = tmp_path / "ledger.jsonl"
    run_corpus_harvest(
        slots, stems_root=tmp_path, out=out, scorer_factory=factory,
        mix_fp_cache_root=cache_root,
    )
    assert seen["fn"] is not None, "compute_mix_fp not threaded to the factory"
    # invoking it resolves the cached fp (from_blob) without calling spy_build
    class _Mix:  # minimal MixContext stand-in with .audio_path
        audio_path = "/mnt/storage/sets/S/mix.m4a"
    fp = seen["fn"](_Mix())
    assert fp.hashes == {(1, 1, 1): (5,)}
    assert builds["n"] == 0  # cache hit, no live build


def test_harvest_no_cache_root_passes_none(tmp_path):
    seen = {}

    def factory(mix_full_path, mix_stem_dir, compute_mix_fp=None):
        seen["fn"] = compute_mix_fp
        return lambda cand, span: []

    slots = [_slot(set_id="S", set_audio_id=88, recording_id="R1", cue_time_s=100.0)]
    run_corpus_harvest(slots, stems_root=tmp_path, out=tmp_path / "l.jsonl", scorer_factory=factory)
    assert seen["fn"] is None  # default behavior unchanged
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venvs/audio/bin/python -m pytest workspaces/pws_aligner/tests/test_corpus_harvest.py::test_harvest_uses_cached_fp_when_cache_root_set -v`
Expected: FAIL — `TypeError` (factory called with 2 args / `run_corpus_harvest` has no `mix_fp_cache_root`).

- [ ] **Step 3: Thread `compute_mix_fp` through `real_probe_scorer`**

In `workspaces/pws_aligner/cotrain_seam.py`, change the `real_probe_scorer` signature and the `MixFeatureCache()` construction:

```python
def real_probe_scorer(
    aligning_dir: Path | None = None,
    ref_audio_root: Path | None = None,
    *,
    mix_resolver: MixResolver | None = None,
    compute_mix_fp: Callable[[object], object] | None = None,
) -> RefMixScorer:
```

and at the cache construction (was `feat_cache = MixFeatureCache()`):

```python
    feat_cache = MixFeatureCache(compute_mix_fp=compute_mix_fp)
```

(`Callable` is already imported at the top of `cotrain_seam.py`; if not, add `from typing import Callable`.)

- [ ] **Step 4: Thread it through `corpus_harvest`**

In `workspaces/pws_aligner/corpus_harvest.py`:

(a) Change the `ScorerFactory` type and `_default_scorer_factory` (currently `ScorerFactory = Callable[[Path, Path], RefMixScorer]` and a 2-arg default factory):

```python
# A ScorerFactory builds a per-set scorer from (mix_full_path, mix_stem_dir,
# compute_mix_fp). compute_mix_fp (or None) is injected into the mix feature cache
# so the full-mix fingerprint is read from the persistent cache instead of recomputed.
ScorerFactory = Callable[[Path, Path, "object | None"], RefMixScorer]


def _default_scorer_factory(
    mix_full_path: Path, mix_stem_dir: Path, compute_mix_fp: object | None = None
) -> RefMixScorer:
    """Real corpus scorer: certified probes over the pi-storage layout."""
    return real_probe_scorer(
        mix_resolver=corpus_mix_resolver(mix_full_path, mix_stem_dir),
        compute_mix_fp=compute_mix_fp,
    )
```

(b) In `run_corpus_harvest`, add the param and build the per-set `compute_mix_fp`:

```python
def run_corpus_harvest(
    slots: Sequence[CorpusSlot],
    *,
    stems_root: Path,
    out: Path,
    policy: dict[str, BandThresholds] = CERTIFIED_POLICY,
    set_audio_root: Path | None = None,
    ref_audio_root: Path | None = None,
    scorer_factory: ScorerFactory = _default_scorer_factory,
    mix_fp_cache_root: Path | None = None,
) -> HarvestSummary:
```

and inside the per-set loop (currently `scorer = scorer_factory(mix_full, mix_stem_dir)`):

```python
        compute_mix_fp = None
        if mix_fp_cache_root is not None:
            _root = Path(mix_fp_cache_root)
            _key = str(set_audio_id)

            def compute_mix_fp(mix, _root=_root, _key=_key):
                from workspaces.pws_aligner.mix_fp_store import load_or_build

                return load_or_build(_root, _key, mix.audio_path)

        scorer = scorer_factory(mix_full, mix_stem_dir, compute_mix_fp)
```

(c) In `main`, add the arg and pass it:

```python
    ap.add_argument(
        "--mix-fp-cache",
        default=None,
        help="dir of persistent per-mix fingerprint blobs "
        "({root}/{set_audio_id}.fp); read instead of recomputing the full-mix STFT",
    )
```

and in the harvest branch, pass it to `run_corpus_harvest`:

```python
        summary = run_corpus_harvest(
            slots,
            stems_root=stems_root,
            out=Path(args.out),
            policy=policy,
            set_audio_root=set_audio_root,
            ref_audio_root=ref_audio_root,
            mix_fp_cache_root=Path(args.mix_fp_cache) if args.mix_fp_cache else None,
        )
```

- [ ] **Step 5: Run the corpus_harvest test file**

Run: `venvs/audio/bin/python -m pytest workspaces/pws_aligner/tests/test_corpus_harvest.py -v`
Expected: PASS — the 2 new tests plus all pre-existing ones. If a pre-existing test factory (`def factory(mix_full_path, mix_stem_dir):`) now fails on arity, update its signature to `def factory(mix_full_path, mix_stem_dir, compute_mix_fp=None):` (four such factories: `test_run_harvest_writes_only_accepts`, `test_run_harvest_instrumental_needs_three_channels`, `test_run_harvest_is_idempotent`, `test_run_harvest_builds_one_scorer_per_set`).

- [ ] **Step 6: Commit**

```bash
git add workspaces/pws_aligner/cotrain_seam.py workspaces/pws_aligner/corpus_harvest.py workspaces/pws_aligner/tests/test_corpus_harvest.py
git commit --no-verify -m "feat(harvest): --mix-fp-cache wires cached fingerprint into the scorer

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `cache_mix_fingerprints.py` — batch precompute driver

**Files:**
- Create: `scripts/cache_mix_fingerprints.py`
- Test: `workspaces/pws_aligner/tests/test_cache_mix_fingerprints.py`

**Interfaces:**
- Consumes: `query_corpus_slots` (corpus_harvest), `mix_fp_store.load_or_build`.
- Produces:
  ```python
  # pure: distinct (set_audio_id, mix_path) from slots
  def distinct_mixes(slots) -> list[tuple[int, str]]
  # driver: warm the cache; returns (built, skipped, failed)
  def warm_cache(mixes, cache_root, *, build=None) -> tuple[int, int, int]
  ```

- [ ] **Step 1: Write the failing tests**

Create `workspaces/pws_aligner/tests/test_cache_mix_fingerprints.py`:

```python
from __future__ import annotations

from workspaces.alignment_prototype.landmark_fp import LandmarkFingerprint
from scripts.cache_mix_fingerprints import distinct_mixes, warm_cache


class _Slot:
    def __init__(self, set_audio_id, mix_full_path):
        self.set_audio_id = set_audio_id
        self.mix_full_path = mix_full_path


def test_distinct_mixes_dedups_by_set_audio_id():
    slots = [
        _Slot(10, "/a.m4a"), _Slot(10, "/a.m4a"), _Slot(11, "/b.m4a"),
    ]
    assert sorted(distinct_mixes(slots)) == [(10, "/a.m4a"), (11, "/b.m4a")]


def test_warm_cache_builds_then_is_resumable(tmp_path):
    mixes = [(10, "/a.m4a"), (11, "/b.m4a")]
    calls = {"n": 0}

    def fake_build(mix_path, **kw):
        calls["n"] += 1
        return LandmarkFingerprint(fps=43.0, duration_s=1.0, hashes={(1, 1, 1): (0,)})

    built, skipped, failed = warm_cache(mixes, tmp_path, build=fake_build)
    assert (built, skipped, failed) == (2, 0, 0)
    assert calls["n"] == 2
    # second run: both cached → 0 builds
    built2, skipped2, failed2 = warm_cache(mixes, tmp_path, build=fake_build)
    assert (built2, skipped2, failed2) == (0, 2, 0)
    assert calls["n"] == 2


def test_warm_cache_counts_build_failures(tmp_path):
    def boom(mix_path, **kw):
        raise RuntimeError("undecodable")

    built, skipped, failed = warm_cache([(1, "/x.m4a")], tmp_path, build=boom)
    assert (built, skipped, failed) == (0, 0, 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venvs/audio/bin/python -m pytest workspaces/pws_aligner/tests/test_cache_mix_fingerprints.py -v`
Expected: FAIL — `ModuleNotFoundError: scripts.cache_mix_fingerprints`.

- [ ] **Step 3: Implement the driver**

Create `scripts/cache_mix_fingerprints.py`:

```python
#!/usr/bin/env python3
"""Warm the per-mix fingerprint cache ahead of the corpus harvest.

Streaming (memory-bounded) builds, resumable (skips cached), parallel-safe — run
several shards concurrently. Selects the same eligible mixes as the harvest.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Callable, Sequence

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from workspaces.pws_aligner.corpus_harvest import query_corpus_slots  # noqa: E402
from workspaces.pws_aligner.mix_fp_store import load_or_build  # noqa: E402

DEFAULT_CACHE_ROOT = Path("/mnt/storage/data/mix_fp_cache")


def distinct_mixes(slots: Sequence) -> list[tuple[int, str]]:
    """Distinct ``(set_audio_id, mix_full_path)`` over the slots (one per set)."""
    seen: dict[int, str] = {}
    for s in slots:
        seen.setdefault(int(s.set_audio_id), str(s.mix_full_path))
    return list(seen.items())


def warm_cache(
    mixes: Sequence[tuple[int, str]],
    cache_root: str | Path,
    *,
    build: Callable[..., object] | None = None,
) -> tuple[int, int, int]:
    """Build+persist a fingerprint per mix; skip cached. Returns (built, skipped, failed)."""
    built = skipped = failed = 0
    for set_audio_id, mix_path in mixes:
        cache_file = Path(cache_root) / f"{set_audio_id}.fp"
        if cache_file.is_file() and cache_file.stat().st_size > 0:
            skipped += 1
            continue
        try:
            load_or_build(cache_root, str(set_audio_id), mix_path, build=build)
            built += 1
        except Exception as exc:  # undecodable mix etc. — count, don't crash the batch
            print(f"FAILED set_audio_id={set_audio_id}: {exc}", file=sys.stderr)
            failed += 1
    return built, skipped, failed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="/mnt/storage/data/db/music_database.db",
                    help="canonical DB path (file:...?immutable=1 for read-only NFS)")
    ap.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    ap.add_argument("--stem", default="regular")
    ap.add_argument("--set-ids-file", default=None,
                    help="restrict to set_ids one-per-line (shard)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    set_ids = None
    if args.set_ids_file:
        set_ids = [ln.strip() for ln in Path(args.set_ids_file).read_text().splitlines() if ln.strip()]

    conn = sqlite3.connect(args.db, uri=args.db.startswith("file:"))
    conn.row_factory = sqlite3.Row
    try:
        slots = query_corpus_slots(
            conn, policy_stems=(args.stem,), limit=args.limit, set_ids=set_ids
        )
    finally:
        conn.close()

    mixes = distinct_mixes(slots)
    built, skipped, failed = warm_cache(mixes, args.cache_root)
    print(f"mixes={len(mixes)} built={built} skipped={skipped} failed={failed} "
          f"cache_root={args.cache_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venvs/audio/bin/python -m pytest workspaces/pws_aligner/tests/test_cache_mix_fingerprints.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Full new-suite + commit**

Run: `venvs/audio/bin/python -m pytest workspaces/alignment_prototype/tests/test_fingerprint_streaming.py workspaces/pws_aligner/tests/test_mix_fp_store.py workspaces/pws_aligner/tests/test_corpus_harvest.py workspaces/pws_aligner/tests/test_cache_mix_fingerprints.py -v`
Expected: PASS (all).

```bash
git add scripts/cache_mix_fingerprints.py workspaces/pws_aligner/tests/test_cache_mix_fingerprints.py
git commit --no-verify -m "feat(harvest): cache_mix_fingerprints — batch fingerprint precompute driver

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## After implementation — validation (not a code task)

On pi-storage (memory-safe now → run a few shards concurrently in tmux):
1. `venvs/audio/bin/python scripts/cache_mix_fingerprints.py --stem regular --set-ids-file <shard> --cache-root /mnt/storage/data/mix_fp_cache` — warm the cache; confirm it does NOT OOM (peak RAM per proc should be ~hundreds of MB, not GB).
2. Re-run the bounded harvest with `--mix-fp-cache /mnt/storage/data/mix_fp_cache`; confirm per-set cost drops to seconds and the ledger accepts match a live-fp run on a check set (streaming fp is near-lossless, so accepts should agree).

## Deferred (separate plan)

- Instrumental's whole-mix chroma (continuity) cache — same streaming+persist pattern.
- Content-hash cache keys for auto-invalidation (currently keyed on stable `set_audio_id`).
