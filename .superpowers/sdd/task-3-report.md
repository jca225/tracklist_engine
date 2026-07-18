# Task 3 Report: `run_corpus_harvest` (batch loop)

**Date:** 2026-07-18
**Branch:** `align-f0-scorer-clean` (worktree: `cotrain-accept-precision`)

## What Was Implemented

Four additions to `workspaces/pws_aligner/corpus_harvest.py` (after `build_corpus_cases`):

1. `HarvestSummary` — frozen dataclass `(n_sets, n_cases, n_harvested, n_written)` with `to_json() -> dict`.
2. `ScorerFactory = Callable[[Path, Path], RefMixScorer]` — type alias for the injected factory seam.
3. `_default_scorer_factory(mix_full_path, mix_stem_dir)` — real corpus scorer via `real_probe_scorer(mix_resolver=corpus_mix_resolver(...))`.
4. `run_corpus_harvest(slots, *, stems_root, out, policy, set_audio_root, ref_audio_root, scorer_factory)` — groups by `set_audio_id`, builds ONE scorer per set, calls `build_corpus_cases`, delegates to `harvest` + `write_ledger`.

Also added to `corpus_harvest.py` imports (were absent / unused, re-added per brief):
- `Callable` from `typing`
- `BandThresholds`, `RefMixScorer`, `corpus_mix_resolver`, `real_probe_scorer` from `cotrain_seam`
- `CERTIFIED_POLICY`, `harvest`, `write_ledger` from `harvest`

Four tests appended to `workspaces/pws_aligner/tests/test_corpus_harvest.py`:
- `test_run_harvest_writes_only_accepts` — 2 regular slots, 2-channel agreement → both ACCEPT; ledger contains both recording_ids with stem=regular.
- `test_run_harvest_instrumental_needs_three_channels` — 2-channel → 0 harvested; 3-channel → 1 harvested.
- `test_run_harvest_is_idempotent` — second run on same ledger → n_written=0, exactly 1 line on disk.
- `test_run_harvest_builds_one_scorer_per_set` — 2 slots same set_audio_id → factory called once; stem dir = `stems_root/77`.

Also added `import json` to test file (needed by `test_run_harvest_writes_only_accepts`).

## TDD Evidence

**RED** (`-k run_harvest` before implementation):
```
ImportError: cannot import name 'run_corpus_harvest' from 'workspaces.pws_aligner.corpus_harvest'
1 error in 0.13s
```

**GREEN** (`-k run_harvest` after implementation):
```
4 passed, 6 deselected in 0.08s
```

**Whole file** (Tasks 1–3 = 10 tests):
```
10 passed in 0.07s
```

## Files Changed

- `workspaces/pws_aligner/corpus_harvest.py` — imports expanded; `HarvestSummary`, `ScorerFactory`, `_default_scorer_factory`, `run_corpus_harvest` appended.
- `workspaces/pws_aligner/tests/test_corpus_harvest.py` — `import json` added; `AlignmentResult` + `run_corpus_harvest` imports added; `_agree` helper + 4 tests appended.

## Self-Review

- **Completeness vs brief:** all 4 named additions present; signatures match spec exactly.
- **YAGNI:** nothing extra added; `_default_scorer_factory` is the only real-scorer wiring, unchanged from brief.
- **Idempotency test:** re-runs `run_corpus_harvest` twice on same `out`, asserts `n_written=0` and 1 line on disk — tests the actual `write_ledger` dedup path, not a mock.
- **Instrumental 3-channel gate:** test explicitly tries 2-channel (0 harvested) then 3-channel (1 harvested) on the same `instrumental` slot — exercises `CERTIFIED_POLICY["instrumental"]` banding, not just the regular path.
- **Per-set factory:** asserts `len(calls)==1` (not just that factory is callable) and checks `calls[0][1] == stems_root / "77"` — confirms stem dir routing.
- **No unused imports:** `Callable` is used by `ScorerFactory`; all cotrain_seam/harvest names are used in implementation or `_default_scorer_factory`.

## Concerns

None. Implementation is verbatim from brief; all 10 tests pass clean.
