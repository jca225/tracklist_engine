# Vocal HuBERT Segment Lane — Design

**Date:** 2026-07-19  
**Status:** implemented (shadow); real-set outcome ledgered NO-GO for peak HuBERT  
**Home:** `workspaces/alignment_prototype/fp_segments/` (shadow-only; not canonical)  
**Prerequisite:** `e704e53` like-for-like stem lanes; vocal landmark FP ledgered
NO-GO in `attic/EXPERIMENTS.md`

## Problem

The vocal segment lane's routing is correct (`mix_vocals.flac` ↔ reference
`vocals.flac`) and the NULL-aware segment decoder is reusable, but Shazam-style
landmark hashes are the wrong vocal observation. Real-set shadow runs were too
sparse and collision-prone after separation. The architecture must stay
vocal-to-vocal; only the correspondence producer changes.

## Goal

Replace landmark hashes on the vocal lane with sparse HuBERT-L9 correspondences
derived from a whole-mix cosine similarity matrix, then feed the existing
`decode_constituent` path unchanged. Prove the observation swap on BB11/BB12
shadow runs (diagnostic only). Keep the instrumental landmark lane untouched
until this vocal observation path is landed.

## Non-goals

- Renaming `LandmarkMatch` / generalizing the correspondence schema
- Cross-stem or full-audio vocal matching
- Threshold mining on BB11/BB12 (freeze sparsification from a synthetic unit
  test; real sets are diagnostic)
- Timeline materialization, tracklist attribution, or default-driver integration
- Phonetic/lyric anchors (deferred if HuBERT peaks also fail as a representation)
- Changing canonical FP index / DB / shared fingerprint caches
- Hand-typing alignment headline numbers (cite `docs/alignment_status.md` only)

## Design

### Invariant (unchanged)

```text
mix_vocals.flac  ↔  reference vocals.flac   only
```

Enforced by `fp_segments.routes.lane("vocal")`. Missing vocal audio abstains;
never fall back to regular or instrumental.

### Observation adapter (new)

Add `fp_segments/hubert_retrieve.py`:

1. Resolve `mix_vocals.flac` via `find_aligning_dir` + route.
2. Resolve each acappella span's reference vocal via existing
   `_vocal_ref_path` / manifest helpers (same as `prepare.py`).
3. Load HuBERT-L9 features through `path_decode._ensure_feat` so `.feat_cache`
   is shared with the rest of the aligner.
4. Pool both sides with `trajectory.features.pool_bins` at frozen `bin_s=0.5`
   (same raster as ridge diagnostic / trajectory).
5. Build whole-mix cosine similarity \(M = \mathrm{mix\_bins} @ \mathrm{ref\_bins}^T\).
6. Sparsify \(M\) to a point cloud and emit `LandmarkMatch` rows for
   `decode_constituent`.

No dependency on `eda/alignment/ridge_diagnostic` (wrong direction). Reuse its
cosine + pooling ideas only.

### Sparsification rule (frozen before real shadow)

Turn dense \(M\) into sparse `(mix_time_s, ref_time_s, weight)` points:

1. Suppress cells below `peak_frac * max(M)` (relative to that matrix's peak).
2. Keep a cell only if it is a local maximum in a small 2-D neighborhood
   (e.g. 3×3).
3. Cap retained peaks per constituent (`max_peaks`) so the decoder never sees a
   near-dense cloud.
4. Weight each peak by its cosine value (positive, finite).
5. Map bin indices to seconds via `bin_s` (bin center).

Constants are chosen only against a synthetic planted-ridge unit test (recover
known diagonal peaks; reject a flat/noisy matrix). Once frozen, do not retune
on BB11/BB12.

`hash_frequency` on `LandmarkMatch` is set to `1` for HuBERT peaks (schema
reuse; not a hash count).

### Runner integration

Extend `fp_segments/run.py` (and a thin prepare helper if needed):

| Flag | Behavior |
|---|---|
| `--lane vocal --observation hubert` | HuBERT matrix → peaks → decode (primary vocal path) |
| `--lane vocal --observation landmark` | Existing landmark path (kept for ledger/repro; not the recommended vocal path) |
| `--lane instrumental` | Unchanged landmark instrumental path |

Default for `--lane vocal` becomes `hubert`. Instrumental ignores
`--observation` (or rejects non-landmark).

Output remains a shadow segment bank under
`workspaces/alignment_prototype/out/fp_segments/`, never a PredictedTimeline.

### Evaluation protocol

1. **Unit safety net:** synthetic \(M\) with a planted ridge recovers expected
   peak correspondences and decodes a sensible segment; empty/noisy \(M\)
   abstains.
2. **Real shadow (primary):** BB11 (`2nvzlh2k`) and BB12 (`1fsnxchk`) with
   existing GT-stem timelines + stem overrides, same recall / false-ratio
   diagnostic used for the landmark vocal lane.
3. **Ledger:** if HuBERT peaks remain weak or false-path dominated, record a
   representation NO-GO (or partial) in `attic/EXPERIMENTS.md` and keep the
   lane/routing; do not re-mine thresholds on these two sets.
4. **No promotion:** canonical SOTA / default driver unchanged.

### Files

| Action | Path |
|---|---|
| Create | `fp_segments/hubert_retrieve.py` |
| Modify | `fp_segments/run.py` (`--observation`, vocal branch) |
| Create | `tests/alignment_prototype/test_fp_segment_hubert_retrieve.py` |
| Modify | `docs/superpowers/plans/2026-07-19-sparse-fingerprint-segment-aligner.md` (checkpoint) |
| Modify | `workspaces/alignment_prototype/attic/EXPERIMENTS.md` (after real shadow) |

Optionally a tiny `fp_segments/prepare_hubert.py` only if whole-mix HuBERT
precompute needs an explicit CLI; otherwise `_ensure_feat` on first run is
enough.

### Sequencing after this work

Instrumental landmark next (user choice): collision weighting / BB11–BB12
asymmetry analysis, still shadow-only, still no threshold mining on the two
real sets as the sole tuning loop.

## Success criteria

- Strict vocal-to-vocal routing preserved mechanically.
- Synthetic peak sparsification test green.
- Both real sets produce a vocal HuBERT segment bank without touching
  canonical state.
- Outcome recorded (go / partial / NO-GO) without claiming a SOTA change.
- `make check` green.

## Open questions (resolved in conversation)

| Question | Decision |
|---|---|
| Similarity matrix vs matched-filter votes vs lyrics-first | Matrix → local peaks (**A**) |
| Synthetic-first vs real-first | Real shadow primary; synthetic unit safety net (**B**) |
| Schema rename now? | No — reuse `LandmarkMatch` |
| Approach | Observation adapter only |
