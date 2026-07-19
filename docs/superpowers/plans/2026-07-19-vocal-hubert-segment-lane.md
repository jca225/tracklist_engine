# Vocal HuBERT Segment Lane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive the existing vocal segment decoder with sparse HuBERT-L9 peaks from a whole-mix cosine similarity matrix, shadow-only on BB11/BB12.

**Architecture:** Observation adapter only (`hubert_retrieve.py`) → `LandmarkMatch` → unchanged `decode_constituent`. Strict `mix_vocals ↔ vocals` routing preserved. Landmark vocal path remains available for repro.

**Tech Stack:** Python, numpy, existing `_ensure_feat` / `pool_bins`, pytest, shadow CLI under `fp_segments`.

**Spec:** `docs/superpowers/specs/2026-07-19-vocal-hubert-segment-lane-design.md`

## Global Constraints

- Shadow-only; no canonical DB / shared FP index mutation
- No BB11/BB12 threshold mining; freeze sparsification from synthetic unit test
- No schema rename; reuse `LandmarkMatch` with `hash_frequency=1`
- `bin_s=0.5`, HuBERT layer 9
- Cite `docs/alignment_status.md` only for headline numbers

## File map

| File | Role |
|---|---|
| `fp_segments/hubert_retrieve.py` | cosine \(M\) + peak sparsification → matches |
| `fp_segments/run.py` | `--observation hubert\|landmark`; vocal hubert branch |
| `tests/alignment_prototype/test_fp_segment_hubert_retrieve.py` | synthetic ridge + noise gates |
| plan + EXPERIMENTS + design status | checkpoint after real shadow |

---

### Task 1: Peak sparsification (TDD)

**Files:** create `hubert_retrieve.py`, create test file

- [x] Write failing tests: planted diagonal ridge → peaks near diagonal; flat/noisy matrix → empty
- [x] Run tests, confirm fail
- [x] Implement `sparsify_similarity` + `matches_from_similarity`
- [x] Run tests, confirm pass

### Task 2: Feature → matches helper

- [x] Add `retrieve_hubert_matches(mix_feat, ref_feat, ...)` using `pool_bins` + cosine
- [x] Unit test with synthetic feature matrices (no audio model load)
- [x] Confirm pass

### Task 3: Runner wiring

- [x] Add `--observation` to `run.py`; default vocal→hubert, instrumental→landmark
- [x] Vocal hubert path: resolve mix/ref vocals, `_ensure_feat`, retrieve, decode
- [x] Landmark vocal path unchanged when requested
- [x] Mix-hash-cache not required for hubert observation

### Task 4: Real shadow + ledger

- [x] Run BB12 then BB11 vocal hubert shadow
- [x] Record diagnostic recall/false_ratio
- [x] Update EXPERIMENTS + plan checkpoint + design status
- [x] `make check`

### Task 5: Commit (only if user asks)

- [ ] Stage relevant files; conventional commit message

### Shadow diagnostic (not headline metrics)

Same crude slot diagnostic used for the landmark vocal lane:

- BB12 HuBERT: slots 99, decoded 66, recall 0.000, false_ratio 27.89
- BB11 HuBERT: slots 94, decoded 37, recall 0.085, false_ratio 4.03

Verdict: representation NO-GO for peak-sparsified HuBERT into this decoder.
