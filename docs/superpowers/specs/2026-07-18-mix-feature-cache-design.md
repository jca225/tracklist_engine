# Mix-Feature Cache (streaming) — Design

**Date:** 2026-07-18
**Branch:** cotrain-corpus-harvest (PR #14 lineage)
**Status:** Design (pre-implementation)

## Problem

The corpus-harvest scorer computes the **full-mix landmark fingerprint live, per set**
(`mix_feature_cache._default_mix_fp` → `librosa.load(entire mix)` +
`fingerprint_from_audio` → `constellation` STFT over a 1–2 hr signal). This is:

- **RAM-heavy** — the whole-signal STFT peaks at multiple GB. On the 7.6 GB / 4-core
  pi cluster, 3 concurrent harvest procs OOM (pi-storage OOM-killed a shard; pi-worker
  swap-thrashed into an unrecoverable, unreachable state). Safe concurrency ≈ 1–2/box.
- **Recomputed every run** — ~10+ min/set, paid again on every harvest pass.

A sequential 829-set run is ~100–140 hrs; parallelism is capped by RAM, not cores.
(Full evidence: memory `project_corpus_harvest_census`, points 4–5.)

**Key narrowing:** the certified **regular** policy uses only **fp + chroma**, and
`chroma` is *windowed* (per-span, already memory-bounded). So for the regular
flywheel there is exactly **one** full-mix, RAM-heavy feature to fix: the landmark
fingerprint.

This is a **caching/performance layer over an existing feature**, not a new probe —
it does not touch the alignment_prototype "sensor phase is closed" freeze.

## Goal

Precompute + persist the per-mix landmark fingerprint **once**, computed in a
**streaming (memory-bounded)** way, so the harvest reads a compact cached fingerprint
instead of recomputing. Result: per-set harvest cost collapses from ~10 min + multi-GB
to seconds + bounded RAM, making high concurrency safe on the pis and the pass
repeatable.

## Non-goals

- Instrumental's whole-mix **chroma** (continuity, the 3rd channel) — same pattern,
  deferred until regular is running (YAGNI; regular is the certified, biggest axis).
- Any change to the fingerprint *algorithm* or the probes' matching logic.
- A DB-backed store (avoids canonical-DB schema change + write-contention + the
  read-only-NFS-write hazard). File cache instead.

## Why streaming is near-lossless here

`LandmarkFingerprint` is `{hash: (anchor_time_frames, ...)}` with `fps` +
`duration_s`, and already serializes via `to_blob()`/`from_blob()`. Because a
fingerprint is fundamentally *a dict of hash → times*, a chunked computation that
**offsets each chunk's anchor times by the chunk's start-frame and merges the hash
dicts** reproduces the full-signal fingerprint exactly, except for landmark *pairs*
straddling a chunk boundary.

`hashes(tf, fb, fan=8, dt_max=80)` pairs a peak only with peaks within **`dt_max=80`
frames** (`FPS = 22050/512 ≈ 43.07` → **~1.86 s**). Therefore a chunk **overlap ≥
`dt_max` frames** guarantees every pair is fully formed inside some chunk; overlapping
chunks then emit duplicate `(hash, anchor_time)` entries, removed by dedup on merge.
Overlap default: **3.0 s** (comfortably > 1.86 s).

## Architecture — four small units

```
cache_mix_fingerprints.py  (batch precompute driver: warm cache over a set list)
        │  per set, once, resumable, parallel-safe (memory-bounded)
        ▼
mix_fp_store.load_or_build(cache_root, key, mix_path) ──► {cache_root}/{key}.fp  (to_blob)
        │  read compact blob (from_blob) or build+persist (atomic temp→rename)
        ▼
landmark_fp.fingerprint_from_file_streaming(path, chunk_s=120, overlap_s=3.0)
        │  peak RAM = ONE chunk (~100-200 MB), not the whole signal
        ▼
corpus_harvest  (--mix-fp-cache <root>)  ──►  MixFeatureCache(compute_mix_fp=load_or_build(...))
        harvest reads compact fp + windowed chroma → per-set cost = seconds, bounded RAM
```

### 1. `fingerprint_from_file_streaming` (in `alignment/landmark_fp.py`)

```python
def fingerprint_from_file_streaming(
    path: str | Path, *, chunk_s: float = 120.0, overlap_s: float = 3.0
) -> LandmarkFingerprint
```

- Iterate the mix in windows of `chunk_s` (stepping by `chunk_s`, loading
  `chunk_s + overlap_s` via `librosa.load(path, sr=SR, offset=t0, duration=...)`).
- Per chunk: `tf, fb = constellation(chunk_y)` → local anchor frames; compute the
  chunk's `hashes(tf, fb)`; **offset each anchor time by `round(t0 * FPS)`**; merge
  into the global `{hash: set-of-times}` accumulator (a set dedups the overlap
  duplicates), then freeze to sorted tuples.
- `duration_s` = total decoded duration (track across chunks); `fps = FPS`.
- Returns a `LandmarkFingerprint` structurally identical to `fingerprint_from_audio`.
- Reuses `constellation` + `hashes` unchanged — no DSP reimplementation.

### 2. `mix_fp_store.py` (new, in `pws_aligner/`)

```python
def load_or_build(cache_root: Path, key: str, mix_path: str | Path,
                  *, chunk_s: float = 120.0, overlap_s: float = 3.0) -> LandmarkFingerprint
```

- Cache file: `Path(cache_root) / f"{key}.fp"` holding `fp.to_blob()`.
- If present and non-empty → `LandmarkFingerprint.from_blob(file.read_bytes())`.
- Else → `fingerprint_from_file_streaming(mix_path, ...)`; write atomically
  (`{key}.fp.tmp` → `os.replace`); return it.
- Corrupt/short blob (`from_blob` raises) → rebuild (treat as cache miss).
- `key` = `str(set_audio_id)` (stable, unique per set mix). `cache_root` default
  `/mnt/storage/data/mix_fp_cache/`.

### 3. `cache_mix_fingerprints.py` (new, in `scripts/`)

Batch driver to warm the cache ahead of harvest.

- Args: `--db` (URI-aware, matching corpus_harvest), `--stem regular`, `--cache-root`,
  `--set-ids-file` (shard, reusing the harvest's sharding), `--limit`, `--chunk-s`,
  `--overlap-s`.
- Selects the eligible mixes (`(set_audio_id, mix_path)` distinct over the same
  eligibility as `query_corpus_slots`), and for each calls `load_or_build` (skips
  when cached → **resumable**). Prints progress + a summary (built / skipped / failed).
- Parallel-safe: each build is memory-bounded, so run N shards concurrently (now the
  pis can do more than 1–2 without OOM).

### 4. Wiring `corpus_harvest`

- New arg `--mix-fp-cache <root>` (default: unset → current live behavior, unchanged).
- When set, the per-set scorer's `MixFeatureCache` is built with
  `compute_mix_fp = lambda mix: mix_fp_store.load_or_build(root, set_audio_id, mix.audio_path)`.
  `run_corpus_harvest` already groups by `set_audio_id`, so the key is in scope at
  scorer-build time; thread `set_audio_id` into the `scorer_factory`.
- **Lazy fill:** if a set's `.fp` is absent, the harvest builds+persists it (streaming)
  on first touch — so the precompute driver is an optimization (warm ahead), not a
  hard prerequisite.

## Data flow

precompute (streaming, once/mix, parallel) → `{set_audio_id}.fp` on `/mnt/storage`
(pi-storage local; pi-worker reads over the read-only NFS mount) → harvest reads the
compact fp + computes windowed chroma per span → certified banding → ledger.

## Error handling

- Undecodable / missing mix → `librosa` raises → the builder propagates; the harvest
  path catches (as today) and the probe abstains (no crash, no false harvest).
- Partial cache write → atomic temp→rename; a crash leaves no half-file.
- Corrupt cache blob → rebuild on read.
- Cache-root not writable (e.g. someone points harvest at the RO NFS mount) → build
  still works in-memory but persist fails; log a warning and continue (degrades to
  live-per-run, not a crash). The precompute driver must run where the root is writable.

## Testing

1. **Streaming ≈ full-signal equivalence** (the load-bearing test): on a short
   synthetic signal (a few concatenated tones, ~30–60 s so it spans ≥2 chunks with a
   small `chunk_s`), assert `fingerprint_from_file_streaming` produces a hash dict
   equal to `fingerprint_from_audio` on the same signal (same keys; per-key times equal
   as sets). Prove the overlap+dedup recovers boundary pairs (a variant with `overlap_s=0`
   should DIFFER at the boundary, confirming the overlap matters).
2. **Store round-trip + skip:** `load_or_build` writes a `.fp`, a second call reads it
   without rebuilding (inject a counting builder); corrupt-blob → rebuild.
3. **Memory guard:** assert `fingerprint_from_file_streaming` never holds the whole
   signal — e.g. a fake loader that records the largest `duration` requested and assert
   it ≤ `chunk_s + overlap_s`.
4. **Driver resumability:** run over a 2-set fixture list twice; second run builds 0.
5. **Harvest wiring:** with `--mix-fp-cache` pointed at a pre-warmed fixture, the
   per-set scorer reads the cached fp (counting builder shows 0 live fingerprint builds).

## Rollout

1. Land units 1–4 (TDD) on `cotrain-corpus-harvest`.
2. Warm the cache with `cache_mix_fingerprints.py` on pi-storage over the bounded set
   list (now memory-safe → run several shards concurrently). This is the one-time cost.
3. Re-run the bounded harvest with `--mix-fp-cache` — per-set cost now seconds; validate
   the ledger + that streaming-fp harvest matches live-fp harvest on a check set (same
   accepts).
4. Scale out (more sets / the full 829) once per-set cost is cheap.

## Open questions (resolve in planning)

- `chunk_s` default (120 s balances chunk count vs per-chunk RAM; tune if needed).
- Whether to also key the cache on a mix content hash for auto-invalidation, or rely on
  `set_audio_id` stability (canonical mixes don't change) — default to `set_audio_id`.
