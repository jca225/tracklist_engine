# Task 3 Report: Wire the cache into the harvest scorer

**Date:** 2026-07-18
**Branch:** `cotrain-corpus-harvest` (worktree: `cotrain-accept-precision`)
**Base SHA:** `3e5d7ddd8d64f2893dac7d3bd39a8a4d197a2e01` (verified)
**New commit SHA:** `84ee46a1f82bdddb8991bacd34de2a8ffcff189c`

## What Was Implemented

Task 3 threads an optional `compute_mix_fp` callable through the harvest scorer
so that when `--mix-fp-cache <root>` is set, the per-set full-mix fingerprint is
read from `{root}/{set_audio_id}.fp` instead of recomputed live. When unset,
behavior is byte-for-byte unchanged (`compute_mix_fp=None` → `MixFeatureCache`
falls back to its existing `_default_mix_fp`).

### Files changed

1. **`workspaces/pws_aligner/cotrain_seam.py`** — added `compute_mix_fp:
   Callable[[object], object] | None = None` kwarg to `real_probe_scorer`; changed
   `feat_cache = MixFeatureCache()` → `feat_cache = MixFeatureCache(compute_mix_fp=compute_mix_fp)`.
   (`Callable` was already imported at line 50 — no new import needed.)

2. **`workspaces/pws_aligner/corpus_harvest.py`** — three edits:
   - (a) `ScorerFactory` type alias changed to 3-arg `Callable[[Path, Path, "object | None"], RefMixScorer]`;
     `_default_scorer_factory` updated to accept and forward `compute_mix_fp`.
   - (b) `run_corpus_harvest` received new `mix_fp_cache_root: Path | None = None` param;
     inner loop builds per-set `compute_mix_fp` closure (calling `mix_fp_store.load_or_build`)
     when cache root is set, else leaves it `None`; `scorer_factory` call updated to 3-arg.
   - (c) `main` received `--mix-fp-cache` CLI arg; passed to `run_corpus_harvest` in harvest branch.

3. **`workspaces/pws_aligner/tests/test_corpus_harvest.py`** — added 2 new tests
   (`test_harvest_uses_cached_fp_when_cache_root_set`, `test_harvest_no_cache_root_passes_none`);
   updated 4 pre-existing test factory signatures from 2-arg to
   `def factory(mix_full_path, mix_stem_dir, compute_mix_fp=None)`.

## TDD Evidence

**RED** (first new test before implementation):
```
FAILED workspaces/pws_aligner/tests/test_corpus_harvest.py::test_harvest_uses_cached_fp_when_cache_root_set
TypeError: run_corpus_harvest() got an unexpected keyword argument 'mix_fp_cache_root'
1 failed in 0.15s
```

**GREEN** (whole file after implementation):
```
29 passed in 0.14s
```

**Package tests** (no collateral breakage):
```
94 passed in 2.10s
```

## Self-Review

- **Live code exactly matched brief's assumptions.** `real_probe_scorer` was at
  lines 588–641 as stated; `MixFeatureCache()` construction was at line 641;
  `ScorerFactory = Callable[[Path, Path], RefMixScorer]` was at line 215;
  `_default_scorer_factory` was 2-arg at line 218; `scorer_factory(mix_full,
  mix_stem_dir)` call was at line 248. No deviations from brief's assumed state.
- **Exactly 4 pre-existing factories needed arity fixes** as predicted
  (`test_run_harvest_writes_only_accepts`, `test_run_harvest_instrumental_needs_three_channels`
  — both `two_channel` and `three_channel` factories — `test_run_harvest_is_idempotent`,
  `test_run_harvest_builds_one_scorer_per_set`). Note: `test_run_harvest_instrumental_needs_three_channels`
  contained 2 factory definitions; both were updated.
- **No unused imports added; no extra behavior added** beyond the brief's spec (YAGNI).
- **`MixFeatureCache.__init__`** was already 3-arg with `compute_mix_fp` kwarg as stated —
  `mix_feature_cache.py` was not modified.

## Concerns

None. Implementation is verbatim from brief; all assumptions about live code signatures
were confirmed correct.
