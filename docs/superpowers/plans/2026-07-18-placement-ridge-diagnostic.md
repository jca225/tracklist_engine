# Placement Ridge Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small EDA study that, for ~8–12 identity-correct / placement-wrong spans on BB11+BB12, renders mix↔ref similarity matrices under four existing representations and decides per case whether the GT diagonal ridge is present (decoder wall) or absent (representation wall / encoder earned).

**Architecture:** A new `eda/alignment/ridge_diagnostic/` package. Select hard cases from agentic (or fallback) timelines via `score_spans`; for each case crop mix/ref audio, compute \(M(t,s)\) under HuBERT-L9 / chroma / fp-hit / instrumental-stem channels using existing feature caches; measure ridge contrast against the GT diagonal; write heatmaps + a contrast table + FINDINGS. No training, no new aligner probes.

**Tech Stack:** Python 3 (`from __future__ import annotations`, frozen dataclasses, full type hints), numpy, librosa, matplotlib, pytest. Interpreter: `venvs/audio/bin/python`.

**Spec:** [docs/superpowers/specs/2026-07-18-placement-ridge-diagnostic-design.md](../specs/2026-07-18-placement-ridge-diagnostic-design.md)

## Global Constraints

- Interpreter: `venvs/audio/bin/python`; run/import from repo root.
- **Set ids (verified — do NOT swap): BB11 = `2nvzlh2k`, BB12 = `1fsnxchk`.**
- **EDA only.** Live under `eda/alignment/ridge_diagnostic/`. Do NOT add probes/channels to `alignment/{infer,harness,drivers}` — sensor phase is closed.
- **No training.** No contrastive encoder, no Milvus, no new retrieval stack.
- **Looking exercise, not statistics.** n≈8–12 by design. Do not claim feature-engineered findings from n&lt;50. Verdict language: "ridge present / absent," not p-values.
- **Numbers SSOT:** do not hand-type alignment headline metrics into FINDINGS — cite `docs/alignment_status.md`.
- **Audio local:** `~/aligning/<set_id>__*/`. If missing: `venvs/audio/bin/python labeling/pull_set_for_alignment.py <set_id>`. Do not mutate pi-storage.
- Style: frozen dataclasses for records; I/O at edges; small focused files.
- Commit after each task only if the user/operator asked for commits; otherwise leave a clean working tree of logical units.

---

## File Structure

```
eda/alignment/ridge_diagnostic/
  __init__.py                 # CREATE (empty or docstring)
  README.md                   # CREATE: how to run + decision rule
  cases.py                    # CREATE: hard-case selection → CaseRecord list
  features.py                 # CREATE: load/crop audio + 4-channel M(t,s)
  ridge.py                    # CREATE: GT diagonal mask + ridge contrast
  plot.py                     # CREATE: heatmap PNGs with GT overlay
  run.py                      # CREATE: CLI orchestrator
  FINDINGS.md                 # CREATE at end (after running the study)
  out/                        # gitignored outputs
    cases.json
    contrast_table.tsv
    heatmaps/*.png
tests/eda/alignment/
  test_ridge_diagnostic.py    # CREATE: unit tests on synthetic matrices
eda/alignment/ridge_diagnostic/.gitignore  # CREATE: ignore out/
```

---

### Task 1: Package scaffold + CaseRecord + hard-case selector

**Files:**
- Create: `eda/alignment/ridge_diagnostic/__init__.py`
- Create: `eda/alignment/ridge_diagnostic/.gitignore`
- Create: `eda/alignment/ridge_diagnostic/cases.py`
- Create: `tests/eda/alignment/test_ridge_diagnostic.py`
- Create: `eda/alignment/ridge_diagnostic/README.md` (stub — fill run instructions in Task 5)

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class CaseRecord:
      case_id: str              # e.g. "bb12_1w1"
      set_id: str               # 1fsnxchk | 2nvzlh2k
      set_label: str            # BB12 | BB11
      slot: str
      recording_id: str
      claimed_stem: str         # from matched GT row
      span_class: str           # linear|multiseg|loop|oddratio
      place_err_s: float
      pred_set_start_s: float
      gt_set_start_s: float
      gt_audible_onset_s: float
      gt_ref_start_s: float
      gt_set_end_s: float
      gt_ref_end_s: float | None
      tempo_ratio: float
      pitch_shift_semi: int
      ref_segments: tuple[tuple[float, float, float], ...]  # (mix_start, ref_start, ref_end)
      source_timeline: str      # path used for selection
      taxonomy_tags: tuple[str, ...]  # optional labels for reading
  ```
- Produces: `select_hard_cases(*, n: int = 12, min_place_err_s: float = 15.0, timeline_by_set: dict[str, Path] | None = None) -> list[CaseRecord]`

**Selection rules (exact):**
1. Prefer agentic timelines at `alignment/out/<set_id>_agentic_timeline.json`.
2. If missing, fall back to `alignment/out/<set_id>_predicted_timeline_lt.json` (same source `failure_analysis/build_span_table` uses), then any `*_predicted_timeline*.json`.
3. Score with `score_spans(set_id, timeline_path)` from `alignment.score_timeline_vs_gt`.
4. Keep spans where `id_correct is True` and `place_err_s is not None` and `place_err_s >= min_place_err_s`.
5. Sort by `place_err_s` descending; take top `n` across both sets (mix BB11+BB12).
6. Enrich each survivor from GT YAML (`labeling/fixtures/bb1{1,2}_ground_truth.yaml`) for tempo_ratio, pitch_shift_semi, ref_segments, audible onset via `gt_placement_onset`.

- [ ] **Step 1: Write the failing test for CaseRecord selection helpers**

```python
# tests/eda/alignment/test_ridge_diagnostic.py
from __future__ import annotations

from eda.alignment.ridge_diagnostic.cases import CaseRecord, _rank_candidates


def test_rank_candidates_prefers_large_place_err_with_id_correct() -> None:
    rows = [
        {"id_correct": True, "place_err_s": 40.0, "slot": "a"},
        {"id_correct": True, "place_err_s": 5.0, "slot": "b"},
        {"id_correct": False, "place_err_s": 99.0, "slot": "c"},
        {"id_correct": True, "place_err_s": None, "slot": "d"},
    ]
    ranked = _rank_candidates(rows, min_place_err_s=15.0)
    assert [r["slot"] for r in ranked] == ["a"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/johnnycabrahams/Desktop/tracklist_engine
venvs/audio/bin/python -m pytest tests/eda/alignment/test_ridge_diagnostic.py::test_rank_candidates_prefers_large_place_err_with_id_correct -v
```

Expected: FAIL (module/import missing).

- [ ] **Step 3: Implement scaffold + selector**

`.gitignore`:
```
out/
```

`cases.py` key pieces:

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from alignment.path_decode import gt_placement_onset
from alignment.score_timeline_vs_gt import score_spans

_REPO = Path(__file__).resolve().parents[3]
_OUT = Path(__file__).resolve().parent / "out"
_ALN_OUT = _REPO / "alignment" / "out"

SET_META = {
    "1fsnxchk": ("BB12", _REPO / "labeling/fixtures/bb12_ground_truth.yaml"),
    "2nvzlh2k": ("BB11", _REPO / "labeling/fixtures/bb11_ground_truth.yaml"),
}


@dataclass(frozen=True)
class CaseRecord:
    ...  # fields as in Interfaces


def _rank_candidates(rows: list[dict], *, min_place_err_s: float) -> list[dict]:
    kept = [
        r
        for r in rows
        if r.get("id_correct") is True
        and r.get("place_err_s") is not None
        and float(r["place_err_s"]) >= min_place_err_s
    ]
    return sorted(kept, key=lambda r: float(r["place_err_s"]), reverse=True)


def resolve_timeline(set_id: str, override: Path | None = None) -> Path:
    if override is not None:
        return override
    preferred = [
        _ALN_OUT / f"{set_id}_agentic_timeline.json",
        _ALN_OUT / f"{set_id}_predicted_timeline_lt.json",
    ]
    for p in preferred:
        if p.is_file():
            return p
    hits = sorted(_ALN_OUT.glob(f"{set_id}_predicted_timeline*.json"))
    if not hits:
        raise FileNotFoundError(
            f"no timeline for {set_id} under {_ALN_OUT} — run "
            f"`make race SETS={set_id} DRIVERS=agentic` or `make align SET={set_id}`"
        )
    return hits[0]


def select_hard_cases(
    *,
    n: int = 12,
    min_place_err_s: float = 15.0,
    timeline_by_set: dict[str, Path] | None = None,
) -> list[CaseRecord]:
    # score each set, rank, take top-n, enrich from GT yaml → CaseRecord
    ...
```

Also write `dump_cases(cases: list[CaseRecord], path: Path) -> None` that writes `cases.json`.

If no agentic/predicted timelines exist locally, the selector must fail with an actionable message (command above) — do not silently invent cases from taxonomy alone in v1. Taxonomy tags may be filled later by hand in FINDINGS.

- [ ] **Step 4: Run unit test + a dry import of select**

```bash
venvs/audio/bin/python -m pytest tests/eda/alignment/test_ridge_diagnostic.py::test_rank_candidates_prefers_large_place_err_with_id_correct -v
venvs/audio/bin/python -c "
from eda.alignment.ridge_diagnostic.cases import resolve_timeline, select_hard_cases
for s in ('1fsnxchk','2nvzlh2k'):
    try:
        print(s, resolve_timeline(s))
    except FileNotFoundError as e:
        print('MISSING', s, e)
"
```

Expected: unit test PASS. Print either timeline paths or the exact pull/race command needed. If timelines are missing, run (expensive — only if needed):

```bash
make race SETS=1fsnxchk,2nvzlh2k DRIVERS=agentic
```

Or, if a classical/predicted timeline already exists and agentic is too expensive for this session, proceed with the fallback timeline and record `source_timeline` honestly in `cases.json`.

- [ ] **Step 5: Commit (if operator requested commits)**

```bash
git add eda/alignment/ridge_diagnostic tests/eda/alignment/test_ridge_diagnostic.py
git commit -m "$(cat <<'EOF'
feat(eda): scaffold placement ridge-diagnostic case selector

EOF
)"
```

---

### Task 2: Feature panel — four-channel \(M(t,s)\)

**Files:**
- Create: `eda/alignment/ridge_diagnostic/features.py`
- Modify: `tests/eda/alignment/test_ridge_diagnostic.py`

**Interfaces:**
- Consumes: `CaseRecord`, aligning dir via `alignment.refine_ref_offsets.find_aligning_dir` (or `recon_probe.find_aligning_dir` — same pattern).
- Produces:
  ```python
  Channel = Literal["hubert", "chroma", "fp_hit", "instr_stem"]

  @dataclass(frozen=True)
  class SimMatrix:
      channel: Channel
      M: object            # np.ndarray shape (Tm, Tr), float32 in [-1, 1] or [0, 1]
      mix_bin_s: float     # seconds per mix row
      ref_bin_s: float     # seconds per ref col
      mix_audio: str       # path used
      ref_audio: str       # path used
      notes: str           # e.g. "tempo-stretched ref by 1.026"
  ```
- Produces: `compute_panel(case: CaseRecord, *, bin_s: float = 0.5, pad_s: float = 15.0) -> dict[Channel, SimMatrix]`

**Audio resolution (exact):**
1. `set_dir = find_aligning_dir(case.set_id)` under `~/aligning/<set_id>__*`.
2. Mix full: first `set_dir.glob("mix.*")` that is a file.
3. Mix vocals / instrumental: `set_dir / "mix_vocals.flac"`, `set_dir / "mix_instrumental.flac"` (warn + skip channel if absent).
4. Ref track: resolve from `set_dir / "manifest.json"` by `recording_id` / `track_id` (same as scorer's `_resolve_ref_audio` pattern). Stem audio under `set_dir / "stems" / <slot>__* / {vocals,instrumental}.flac`.
5. Crop mix to `[gt_audible_onset_s - pad_s, gt_set_end_s + pad_s]` (clamp to file). Crop ref to a window covering GT ref span ± pad (or full ref if short).
6. Tempo-lock: resample ref by `tempo_ratio` (librosa) before feature extract so GT diagonal is ~45°.

**Channel math:**

| channel | matrix |
|---|---|
| `hubert` | L2-normalized pooled HuBERT-L9 bins via `path_decode._ensure_feat(path, path, "hubert", 9)` then `trajectory.features.pool_bins`; \(M = mm @ rm.T\) |
| `chroma` | same with `"chroma"` |
| `fp_hit` | landmark hashes from `landmark_fp.fingerprint_from_audio` on cropped mix/ref; bin hash co-occurrences into a `(Tm, Tr)` density map (row = mix time bin, col = ref time bin). Normalize to [0,1]. This is *not* cosine — it is the sparse-peak channel. |
| `instr_stem` | same as hubert/chroma cosine but on `mix_instrumental` × ref `instrumental.flac`. If either missing, omit channel with a note (do not fabricate). For acappella-only cases where instrumental stems are irrelevant, still compute if files exist — absence of ridge is informative. |

Reuse caches: HuBERT/chroma go through `_ensure_feat` so `.feat_cache/` is shared with the aligner. Do not reimplement HuBERT loading.

- [ ] **Step 1: Write failing tests for cosine sim + diagonal contrast helpers**

```python
import numpy as np
from eda.alignment.ridge_diagnostic.features import cosine_sim_matrix
from eda.alignment.ridge_diagnostic.ridge import ridge_contrast, gt_diagonal_mask


def test_cosine_sim_matrix_identity_has_strong_diagonal() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 8)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    M = cosine_sim_matrix(x, x)
    assert M.shape == (20, 20)
    assert float(np.mean(np.diag(M))) > 0.9


def test_ridge_contrast_high_on_planted_diagonal() -> None:
    M = np.zeros((30, 30), dtype=np.float32)
    for i in range(30):
        M[i, i] = 1.0
    mask = gt_diagonal_mask(M.shape, offsets_s=(0.0,), bin_s=1.0, band_bins=1)
    c = ridge_contrast(M, mask)
    assert c > 5.0  # true diag >> background
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
venvs/audio/bin/python -m pytest tests/eda/alignment/test_ridge_diagnostic.py -v
```

- [ ] **Step 3: Implement `features.py`**

Minimal surface:

```python
def cosine_sim_matrix(mix_bins: np.ndarray, ref_bins: np.ndarray) -> np.ndarray:
    """(Tm,D)+(Tr,D) L2-normalized → (Tm,Tr) cosine."""
    return (mix_bins @ ref_bins.T).astype(np.float32)


def compute_panel(case: CaseRecord, *, bin_s: float = 0.5, pad_s: float = 15.0) -> dict[str, SimMatrix]:
    ...
```

`fp_hit` implementation sketch (keep in this file; do not add a harness probe):

```python
from alignment.landmark_fp import fingerprint_from_audio

def fp_hit_matrix(mix_y, ref_y, sr, *, bin_s: float) -> np.ndarray:
    # hash landmarks; for each matching hash pair accumulate into
    # (mix_t // bin_s, ref_t // bin_s); return density / max
    ...
```

- [ ] **Step 4: Run unit tests**

```bash
venvs/audio/bin/python -m pytest tests/eda/alignment/test_ridge_diagnostic.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit (if requested)**

```bash
git commit -m "$(cat <<'EOF'
feat(eda): four-channel similarity matrices for ridge diagnostic

EOF
)"
```

---

### Task 3: Ridge contrast + GT diagonal overlay

**Files:**
- Create: `eda/alignment/ridge_diagnostic/ridge.py`
- Modify: `tests/eda/alignment/test_ridge_diagnostic.py`

**Interfaces:**
- Produces:
  ```python
  def gt_diagonal_mask(
      shape: tuple[int, int],
      *,
      offsets_s: tuple[float, ...],
      bin_s: float,
      band_bins: int = 1,
      mix_origin_s: float = 0.0,
      ref_origin_s: float = 0.0,
  ) -> np.ndarray:  # bool (Tm, Tr)
      """True on cells within band_bins of any GT diagonal.

      offset_s semantics: mix_time ≈ ref_time + offset
      where offset = gt_audible_onset_s - gt_ref_start_s for a straight span.
      For each ref_segment (mix_start, ref_start, ref_end), add
      offset = mix_start - ref_start.
      """

  def ridge_contrast(M: np.ndarray, mask: np.ndarray) -> float:
      """mean(M[mask]) / (mean(M[~mask]) + eps). Higher = clearer true ridge."""
  ```

- For multiseg/loop cases, pass one offset per `CaseRecord.ref_segments` entry; for linear, one offset from audible onset − ref_start.

- [ ] **Step 1: Extend tests for multiseg offsets**

```python
def test_gt_diagonal_mask_multiseg_two_offsets() -> None:
    mask = gt_diagonal_mask((40, 40), offsets_s=(0.0, 10.0), bin_s=1.0, band_bins=0)
    assert mask[5, 5]
    assert mask[5, 15]
    assert not mask[5, 25]
```

- [ ] **Step 2: Implement `ridge.py` and pass tests**

```bash
venvs/audio/bin/python -m pytest tests/eda/alignment/test_ridge_diagnostic.py -v
```

- [ ] **Step 3: Commit (if requested)**

---

### Task 4: Plotting + contrast table

**Files:**
- Create: `eda/alignment/ridge_diagnostic/plot.py`

**Interfaces:**
- `save_heatmap(M, mask, path: Path, *, title: str, contrast: float) -> None` — `imshow` of M, GT mask contour/overlay in a distinct color, title includes channel + contrast.
- `write_contrast_table(rows: list[dict], path: Path) -> None` — TSV columns:
  `case_id, set_id, slot, claimed_stem, span_class, place_err_s, channel, ridge_contrast, verdict_channel` where `verdict_channel` is `ridge_present` if contrast ≥ `threshold` else `ridge_absent`.

**Default threshold:** `2.0` (true-diag mean at least 2× background). Expose as CLI flag `--contrast-threshold`. Document that the threshold is a *looking aid*, not a statistical claim — human still reads the heatmaps.

- [ ] **Step 1: Implement plot.py** (matplotlib; Agg backend)

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

- [ ] **Step 2: Smoke-test on a synthetic matrix written to `/tmp`**

```bash
venvs/audio/bin/python - <<'PY'
from pathlib import Path
import numpy as np
from eda.alignment.ridge_diagnostic.ridge import gt_diagonal_mask, ridge_contrast
from eda.alignment.ridge_diagnostic.plot import save_heatmap
M = np.eye(40, dtype=np.float32)
mask = gt_diagonal_mask(M.shape, offsets_s=(0.0,), bin_s=1.0)
c = ridge_contrast(M, mask)
p = Path("/tmp/ridge_smoke.png")
save_heatmap(M, mask, p, title=f"smoke c={c:.2f}", contrast=c)
print(p, p.stat().st_size, c)
PY
```

Expected: PNG exists, contrast ≫ 1.

- [ ] **Step 3: Commit (if requested)**

---

### Task 5: CLI orchestrator + README

**Files:**
- Create: `eda/alignment/ridge_diagnostic/run.py`
- Modify: `eda/alignment/ridge_diagnostic/README.md`

**CLI:**

```bash
venvs/audio/bin/python -m eda.alignment.ridge_diagnostic.run \
  --n 12 \
  --min-place-err-s 15 \
  --bin-s 0.5 \
  --contrast-threshold 2.0
```

Flags:
- `--cases-json PATH` — skip selection; reload a prior `cases.json` (for replot)
- `--sets 1fsnxchk,2nvzlh2k`
- `--timeline SET=PATH` repeatable override

**`run.py` flow:**
1. `cases = select_hard_cases(...)` (or load JSON)
2. Write `out/cases.json`
3. For each case × available channel: `compute_panel` → mask → contrast → heatmap PNG
4. Write `out/contrast_table.tsv`
5. Print a one-screen summary: per case, which channels have ridge_present; overall counts of decoder-wall vs representation-wall cases
6. Do **not** auto-write FINDINGS.md — that is Task 6 (human+agent reading)

Per-case aggregate verdict for the summary line:
- if any channel `ridge_present` → `decoder_wall`
- else → `representation_wall`

- [ ] **Step 1: Implement `run.py` as `__main__`**

Ensure `python -m eda.alignment.ridge_diagnostic.run --help` works (add `if __name__ == "__main__"` or package `__main__.py` — prefer `run.py` with module `-m` path matching other eda scripts).

- [ ] **Step 2: Write README.md**

Include: purpose, decision rule (copy from spec), exact run command, inputs (timelines, `~/aligning/`), outputs, and the explicit non-goal (no encoder build from this package).

- [ ] **Step 3: End-to-end dry run on n=2**

```bash
# Preconditions
ls ~/aligning/1fsnxchk__* ~/aligning/2nvzlh2k__* | head
ls alignment/out/*agentic_timeline.json \
   alignment/out/*predicted_timeline*.json 2>/dev/null | head

venvs/audio/bin/python -m eda.alignment.ridge_diagnostic.run --n 2 --min-place-err-s 15
ls eda/alignment/ridge_diagnostic/out/cases.json \
   eda/alignment/ridge_diagnostic/out/contrast_table.tsv \
   eda/alignment/ridge_diagnostic/out/heatmaps/ | head
```

Expected: 2 cases, ≥1 heatmap PNG per available channel, contrast TSV rows.

- [ ] **Step 4: Full run n=12**

```bash
venvs/audio/bin/python -m eda.alignment.ridge_diagnostic.run --n 12 --min-place-err-s 15
```

If HuBERT first-time extract is slow, that is expected (cache fills `.feat_cache/`). Do not parallelize across GPUs unless already trivial — CPU/MPS via existing helpers is fine.

- [ ] **Step 5: Commit (if requested)**

```bash
git commit -m "$(cat <<'EOF'
feat(eda): ridge-diagnostic CLI to test placement diagonal presence

EOF
)"
```

---

### Task 6: Read the pictures and write FINDINGS.md

**Files:**
- Create: `eda/alignment/ridge_diagnostic/FINDINGS.md`

This task is the study, not more code. An agent (or John) opens the heatmaps + contrast table and writes the verdict.

**FINDINGS.md structure (exact sections):**

```markdown
# Ridge diagnostic findings

Date: YYYY-MM-DD
SHA: <git rev-parse --short HEAD>
Timelines used: <from cases.json source_timeline fields>
n cases: N
Contrast threshold (aid only): 2.0

## Decision rule
- ridge present in ≥1 channel → decoder/voting wall
- ridge absent in all channels → representation wall (encoder earned)

## Per-case table
| case_id | stem | span_class | place_err_s | hubert | chroma | fp_hit | instr_stem | verdict |
|---|---|---|---|---|---|---|---|---|

## Aggregate
- decoder_wall: k/N
- representation_wall: m/N

## What we saw (qualitative, ≤10 bullets)
...

## Recommendation
One paragraph: either (a) next lever is decoder/voting on channel X for class Y,
or (b) encoder is earned — freeze this cases.json as the eval set; do not start
encoder work in this package.

## Non-claims
No statistical generalization from n<50. No new aligner channel shipped.
Headline alignment numbers: see docs/alignment_status.md only.
```

- [ ] **Step 1: Open every heatmap; fill the per-case table from `contrast_table.tsv`, overriding a cell only when the eye disagrees with the threshold (note the override).**

- [ ] **Step 2: Write the Recommendation paragraph. Stop. Do not start an encoder.**

- [ ] **Step 3: Commit FINDINGS (if requested)**

```bash
git add eda/alignment/ridge_diagnostic/FINDINGS.md
git commit -m "$(cat <<'EOF'
docs(eda): ridge-diagnostic findings — decoder vs representation wall

EOF
)"
```

---

## Verification checklist (before declaring done)

```bash
venvs/audio/bin/python -m pytest tests/eda/alignment/test_ridge_diagnostic.py -v
venvs/audio/bin/python -m eda.alignment.ridge_diagnostic.run --n 12
test -f eda/alignment/ridge_diagnostic/out/cases.json
test -f eda/alignment/ridge_diagnostic/out/contrast_table.tsv
test -f eda/alignment/ridge_diagnostic/FINDINGS.md
# make check should still pass (no aligner surface changed)
make check
```

## Fallback if timelines are missing

1. Try existing predicted timelines under `alignment/out/`.
2. If none: `make align SET=1fsnxchk` then `make align SET=2nvzlh2k` (classical base). Agentic preferred but not mandatory for v1 — record source in `cases.json`.
3. Taxonomy-only hand-pick is **out of scope for v1** (loses "model's actual error"). If forced, document the deviation in FINDINGS and tag cases `taxonomy_only`.

## Explicitly out of scope (do not do in this plan)

- Training InfoNCE / mashup-invariant encoder
- Wiring a new probe into `harness/` or `infer.py`
- Changing `path_decode`, voting gates, or stretch grids
- Updating `docs/alignment_status.md` (no new headline metric from this looking study)
- Adding results to the EXPERIMENTS ledger until FINDINGS has a closed verdict worth recording
