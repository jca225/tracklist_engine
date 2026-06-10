# Phase 3 — RoFormer stem-separation migration plan

> **Status:** draft (2026-06-09)  
> **Program slot:** P3 in [alignment_program_plan.md](alignment_program_plan.md)  
> **Companions:** [analysis/CLAUDE.md](../analysis/CLAUDE.md) (current backends),
> [stem_discovery_playbook.md](stem_discovery_playbook.md) (human re-source),
> [alignment_review_20260530.md](alignment_review_20260530.md) (quality motivation)

Replace the corpus **separation floor** (`htdemucs_ft` via Demucs) with a
**Mel-Band RoFormer + BS-RoFormer + SCNet-XL** ensemble. The goal is higher SDR
and less vocal bleed on instrumentals — the main pain point from BB12 labeling on
non-EDM and dense vocal mixes ([alignment_review §4](alignment_review_20260530.md)).

**Not in scope:** finding official acappellas online (P2 stem cascade), or
replacing the UVR vocal-cleanup chain. RoFormer is the new **default floor** when
no better external stem exists; UVR remains an opt-in dry-vocal backend.

---

## Current state

| Piece | Today |
|-------|-------|
| Default backend | `demucs` — `htdemucs_ft`, 1 pass ([demucs_adapter.py](../analysis/adapters/demucs_adapter.py)) |
| Quality backend | `uvr` — MDX/VR cleanup chain ([uvr_chain_adapter.py](../analysis/adapters/uvr_chain_adapter.py)) |
| Contract | `StemSet(vocals, instrumental)` → `track_stems` rows `vocals` + `instrumental` only |
| On-disk layout | `/mnt/storage/stems/{track_audio_id}/{vocals,instrumental}.flac` (16-bit FLAC) |
| CLI selection | `--separator {demucs,uvr}` on `mac_analyze_loop.py`, `mac_analyze_sets.py`, `vast_loop.py` |
| QA tool | `scripts/separate.py --separator {demucs,uvr,both}` |
| Corpus coverage | ~820 distinct `track_audio_id`s with Demucs stems (pi-storage, 2026-06) |

Both backends plug into `pipeline.run_separation()`; downstream schema, labeling
pull (`pull_set_for_alignment.py`), and GT `ref_source=demucs` paths are
unchanged when switching backends.

**Terminology:** Demucs `track_stems.stem_name` (`vocals`, `instrumental`) is
unrelated to identity `track_audio.stem` (`regular` / `acappella` /
`instrumental`). This plan only changes **how** we derive Demucs-axis stems from
a `regular` reference file.

---

## Target state

| Piece | After migration |
|-------|-----------------|
| New backend id | `roformer` |
| Models | Config-driven ensemble — initial trio per [alignment_program_plan §P3](alignment_program_plan.md): Mel-Band RoFormer, BS-RoFormer, SCNet-XL |
| Ensemble | Magnitude-spectrogram average (`avg_fft`), same pattern as UVR karaoke stage |
| Default | `roformer` on new analysis runs; `demucs` kept as fallback + A/B baseline |
| GT / ingest | `ref_source=roformer` when labeling uses separated stems (P1 export table already reserves this) |
| P2 cascade | Tier 4 floor becomes RoFormer-first; Demucs only when RoFormer unavailable or validation fails |

The ensemble list is **data, not code** — swap models when the MVSEP leaderboard
moves without adapter rewrites (mirror `uvr_chain.yaml` / `separation_config.py`).

---

## Design constraints (keep the contract)

1. **Two persisted stems only** — `vocals.flac` + `instrumental.flac`. No drums/bass/other rows.
2. **Same `StemSet` / `StemAsset` types** — no schema migration for `track_stems`.
3. **Same FLAC policy** — 16-bit lossless; rsync-friendly.
4. **Version string in `track_analysis.versions_json`** — e.g. `roformer: melband+bs+scnetxl@avg_fft` so re-separate jobs are auditable.
5. **Detect-then-correct** — never blanket re-separate the corpus; re-run only rows that fail quality gates or are on a whitelist (BB12 refs, acap/inst slots).

---

## Architecture

```
roformer_chain.yaml          # model list, ensemble algo, output format, model_dir
        ↓
separation_config.py         # extend or sibling RoFormerChainConfig (reuse StageSpec pattern)
        ↓
roformer_chain_adapter.py    # load() → RoFormerChainHandle; separate() → StemSet
        ↓
audio_separator_adapter.py   # reuse build() / run_stage() — RoFormer models are audio-separator arch
        ↓
pipeline.run_separation()    # dispatch on Analyzers.separator == "roformer"
```

### Config sketch (`analysis/roformer_chain.yaml`) — **pinned Phase B**

```yaml
model_dir: ~/roformer-models
output_format: FLAC
ensemble_algorithm: avg_fft

# Vocals: 3-model ensemble (SCNet helps vocals; MVSEP ensemble ≈11.5 vocal SDR)
vocal_models:
  - melband_roformer_mvsep.ckpt      # MVSEP-finetuned, not Kimberley-Jensen originals
  - bs_roformer_viperx.ckpt
  - scnet_xl.ckpt

# Instrumental: 2-model ensemble — Mel-Band + BS only (SCNet inst SDR ~17.0 vs ~17.6)
instrumental_models:
  - melband_roformer_mvsep.ckpt
  - bs_roformer_viperx.ckpt

# If bleed-judge fails on inst_ensemble, retry once with phase-aligned subtraction
# (mix - vocal_ensemble) and keep whichever scores lower bleed. Not the default path.
instrumental_fallback: vocal_subtract
bleed_retry_threshold: 0.02        # tune on BB12 EDM/pop clips in Phase B
```

Pin exact checkpoint filenames after MSST/audio-separator smoke. **Avoid** MVSEP
"max instrumental fullness" weights — they trade bleed for body and hurt the DJ
alignment use case.

Final YAML shape follows whatever MSST / `audio-separator` expects; the adapter
hides vendor details.

### `Analyzers` changes

- Add `roformer: RoFormerChainHandle | None`
- Extend `separator` to `Literal["demucs", "uvr", "roformer"]`
- `stems_version` returns the roformer handle when selected
- `load_analyzers(..., separator="roformer")` loads only the roformer handle (same mutual-exclusion pattern as demucs/uvr)

### CLI / workers

| Entry point | Change |
|-------------|--------|
| `scripts/mac_analyze_loop.py` | `--separator` adds `roformer` |
| `scripts/mac_analyze_sets.py` | same |
| `scripts/vast_loop.py` / `analysis/vast_worker.py` | same |
| `scripts/render_set_stems.py` | same (chunked full-mix separation) |
| `scripts/separate.py` | `--separator roformer` and `both` → `demucs+roformer` compare |

---

## Host deployment

RoFormer inference is **heavier** than Demucs (~3–5× wall time per track is a
reasonable planning assumption until benchmarked). Run on GPU hosts only for
corpus batch; Pi CPU is out of scope.

| Host | Stack | Notes |
|------|-------|-------|
| **Vast.ai (CUDA)** | **MSST-WebUI clone** via `scripts/setup_roformer_separation.sh` (run after `vast_bootstrap.sh`; reuses `/venv/main` CUDA torch, symlinks `venvs/msst -> /venv/main`) | Primary batch path |
| **Mac (Apple Silicon)** | Same MSST clone, dedicated `venvs/msst` (MPS torch) | **Validated 2026-06-10:** MSST runs natively on MPS (no CPU fallback) — the earlier mlx-audio-separator recommendation is obsolete. But slow: ~16 min/model-pass on a ~4 min track → ~45 min per track for the full ensemble. QA only. |
| **pi-storage CPU** | Not supported for RoFormer batch | Keep Demucs-off or queue jobs to Mac/Vast |

> **Interpreter gotcha (2026-06-10):** `roformer_chain_adapter` imports MSST
> **in-process** (`sys.path` insert), so the *calling* interpreter needs the MSST
> deps — drive roformer runs with `venvs/msst/bin/python`, **not** `venvs/audio`.
> Also `modules/bs_roformer` imports `neuralop.models.FNO` — the `neuraloperator`
> package is **missing from upstream MSST requirements**; it's pinned in
> [requirements-msst.txt](../requirements-msst.txt).

### New setup script

Add `scripts/setup_roformer_separation.sh` (sibling of `setup_separation.sh`):

1. Detect host (CUDA vs MLX-Mac vs unsupported)
2. Install the correct package variant + ffmpeg
3. Pre-download ensemble checkpoints into `~/roformer-models` (or `MODEL_DIR`)
4. Smoke-separate a 30 s clip; verify provider + output labels

Document Mac MLX install in [analysis/CLAUDE.md](../analysis/CLAUDE.md) once validated.

---

## Validation gate (must pass before default flip)

Do not promote RoFormer to default until these pass on a **held-out clip set**
(~20 tracks spanning EDM, pop, rock, dense vocal, instrumental-heavy):

| Check | Method | Pass criterion |
|-------|--------|----------------|
| **SDR** | Standard source-separation metrics on clips with known stems (MUSDB-style or official store pairs) | RoFormer ensemble **beats** `htdemucs_ft` on median vocal SDR and instrumental SDR |
| **Bleed-as-judge** | Run RoFormer vocal model on **known official instrumentals** (tier-1 P2 samples) | Residual vocal energy ≈ 0 (same gate as P2 §2d.3) |
| **Listening** | A/B on 5 BB12 slots where Demucs failed labeling | Annotator prefers RoFormer ≥ 4/5 |
| **Downstream** | MERT re-embed 10 swapped tracks | No regression in chroma self-similarity vs reference |
| **Speed / VRAM** | Benchmark 1 track + 60-min chunked set on Mac MLX + Vast 4090 | Document p50 track time; set chunk budget for `render_set_stems.py` |

Artifacts: `eda/alignment/` or `workspaces/separation_qa/` notebook + saved WAV pairs;
check into repo as small fixtures only (not full stems).

---

## Rollout phases

### Phase A — Adapter + config (no production flip)

- [x] Add `roformer_chain.yaml` + `roformer_chain_adapter.py`
- [x] Wire `pipeline.load_analyzers` / `run_separation`
- [x] Extend `scripts/separate.py` for local A/B
- [x] `scripts/setup_roformer_separation.sh` — host-detecting (Mac venv / Vast `/venv/main` reuse), pinned MSST commit + ckpt download (2026-06-10)
- [ ] Unit test: mock `audio_separator` return list → correct `StemSet` paths (mirror UVR tests)

**Exit:** `separate.py --separator roformer` works on one file on Mac *(2026-06-10:
running on MPS via `venvs/msst` interpreter — see gotcha above)* and Vast *(pending
first Vast box)*.

### Phase B — Validation + benchmark

- [ ] Curate held-out clip list + official instrumental bleed set
- [ ] Run metrics table (Demucs vs UVR vs RoFormer) — publish summary in this doc § Results
- [ ] Fix ensemble YAML (model choice, inst branch strategy) from failures
- [ ] Size re-separate cost: `(tracks_to_redo) × (p50_seconds) × (GPU $/hr)`

**Exit:** validation gate table all green.

### Phase C — Selective re-separation

Re-separate **only** rows that need it:

| Queue | Rationale |
|-------|-----------|
| BB12 reference `track_audio_id`s used in GT | Aligner training inputs |
| Slots where `ref_source=demucs` in exported GT | Human already accepted separated stems |
| `track_audio.stem IN (acappella, instrumental)` derived from separation | Variant MERT (P4 §6c) depends on clean stems |
| Rows flagged in labeling review as "Demucs inadequate" | detect-then-correct |

Driver options:

- New `scripts/roformer_restem_loop.py` (clone `mac_analyze_loop.py` separation-only path), or
- `mac_analyze_loop.py --separator roformer --only-stems --track-audio-ids …`

On re-separate:

1. Write new FLACs to pi-storage `stems/{track_audio_id}/`
2. `DELETE` old `track_stems` rows for that `track_audio_id`; insert new paths
3. Invalidate downstream analysis for that row (`track_analysis`, `track_mert_measures`) — same pattern as `replace_track_audio.py`
4. Queue MERT re-embed (do not auto-run full pipeline unless requested)

**Do not** bulk re-separate ~820 rows without the cost estimate from Phase B.

### Phase D — Default flip

- [ ] Change `load_analyzers` default `separator="roformer"`
- [ ] Update `requirements-audio.txt` / Vast bootstrap to pull RoFormer deps
- [ ] Update [analysis/CLAUDE.md](../analysis/CLAUDE.md) backend table
- [ ] Update [stem_discovery_playbook.md](stem_discovery_playbook.md) baby-rule line (Demucs → "separated stems")
- [ ] Keep `demucs` backend indefinitely for regression and hosts without RoFormer weights

**Exit:** new `mac_analyze_loop` runs write `versions_json.roformer=…`; Demucs stem count stops growing.

### Phase E — P2 integration (can overlap C)

Once RoFormer is validated:

- [ ] `ingest/stem_quality.py` bleed judge calls RoFormer vocal pass on candidate instrumentals
- [ ] `ingest/stem_resolver.py` tier-4 uses `ref_source=roformer` when separation wins
- [ ] GT export maps `stems/…/vocals.flac` from roformer runs to `ref_source=roformer` (path alone is ambiguous — tag via sidecar or `track_analysis.versions_json` lookup)

---

## Storage & cost sizing

| Resource | Estimate (order of magnitude) |
|----------|-------------------------------|
| Model cache | ~2–6 GB per checkpoint × 3 models → plan **15–25 GB** on Vast + Mac |
| Per-track stem disk | Unchanged (~same as Demucs 2-stem FLAC) |
| Re-separate 500 tracks @ 3 min/track on 4090 | ~25 GPU-hours |
| Re-separate 500 tracks @ 10 min/track on Mac MLX | ~80 wall-hours |

Query before Phase C:

```sql
SELECT COUNT(DISTINCT track_audio_id) FROM track_stems;
SELECT COUNT(*) FROM track_stems WHERE stem_name = 'vocals';
```

---

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| RoFormer slower → Vast/Mac backlog | Separation-only loop; don't block beat/MERT; prioritize BB12 whitelist |
| Mac MPS silent CPU fallback | MLX path only; loud-fail in adapter if expected device ≠ actual |
| Ensemble YAML drift / bad checkpoint | Pin versions in yaml; `stems_version` string in DB; MVSEP re-eval quarterly |
| Worse on some genres vs Demucs | Keep demucs fallback; per-row `versions_json` tells which backend produced stems |
| Blanket re-separate invalidates MERT | detect-then-correct + incremental MERT re-embed ([embedding_backfill_plan.md](embedding_backfill_plan.md)) |
| `ref_source` ambiguity (demucs vs roformer paths identical) | Always join `track_analysis.versions_json` or write `separator` column in future migration |

---

## Verification checklist

- [ ] **A:** `separate.py --separator roformer` succeeds on Mac MLX + Vast CUDA
- [ ] **B:** Median SDR beats Demucs on held-out set
- [ ] **B:** Bleed-judge ≈ 0 on ≥ 3 official instrumentals
- [ ] **C:** BB12 whitelist re-separated; `track_stems.path` updated on pi-storage
- [ ] **C:** MERT re-embed completed for changed rows
- [ ] **D:** Default separator is `roformer`; new analyses show roformer in `versions_json`
- [ ] **E:** P2 bleed gate uses same RoFormer handle (no duplicate model load logic)

---

## Phase B decisions (pinned 2026-06-09)

Goal: **best possible quality** for a corpus that is mostly **EDM + pop** full mixes
where the instrumental stem is used for acappella warping — **vocal bleed on the
instrumental** is the primary failure mode, not raw vocal SDR.

### 1. Instrumental source → **dual-model inst ensemble + bleed fallback**

| Strategy | Verdict | Why |
|----------|---------|-----|
| Single dedicated inst model | ❌ default | One checkpoint leaves harmony bleed on pop; EDM synth+vocal overlap hurts single-pass inst |
| Vocal-subtract (`mix − vocals`) | ⚠️ fallback only | Often lowest bleed when the vocal model is aggressive, but EDM sidechain/pumping and pop reverb tails cause phasey/underwater artifacts as a blanket default |
| **Mel-Band + BS-RoFormer inst `avg_fft`** | ✅ **primary** | MVSEP ensemble series peaks here (~17.8 inst SDR); two strong inst heads averaged beats either alone on dense vocal mixes |
| SCNet in inst ensemble | ❌ | Weaker instrumental head (~17.0); keep SCNet in **vocal** ensemble only |

**Pipeline:**

1. Run vocal 3-model ensemble → `vocals`.
2. Run instrumental 2-model ensemble (same Mel-Band + BS checkpoints, separate forward passes) → `instrumental_candidate`.
3. **Bleed-judge** the candidate (lightweight RoFormer vocal pass on the instrumental, same gate as P2 §2d.3).
4. If bleed > threshold → compute phase-aligned, gain-matched `mix − vocal_ensemble` → pick whichever instrumental scores lower bleed.

Phase B benchmark matrix (held-out **EDM + pop** clips, ~20 tracks): log SDR *and*
bleed-judge *and* one listening pass. Expect inst ensemble to win median bleed;
subtraction should win a minority of pop harmony cases — that's what the fallback is for.

### 2. Python package → **MSST on Vast; MLX on Mac for QA only**

| Host | Choice | Why |
|------|--------|-----|
| **Vast / corpus batch** | **ZFTurbo MSST** | Broadest access to MVSEP-finetuned RoFormer checkpoints and native multi-model ensemble; canonical path for Phase B metrics and Phase C re-stem |
| Mac dev / A-B | `ssmall256/mlx-audio-separator` | Stock `audio-separator` + MPS silently drops ops to CPU; MLX is the only viable local GPU path |
| `audio-separator[gpu]` | Fallback | Acceptable if Phase A smoke proves **identical checkpoints + identical bleed scores** vs MSST on 3 clips; otherwise don't split stacks |

**Do not** pick the package before the checkpoint. Pin MVSEP-finetuned Mel-Band + BS
weights first, then install whichever host tool loads them reliably.

Phase B runs on **Vast 4090 only** for model/strategy selection. Mac MLX parity is
a secondary gate (same 3 clips, ears + bleed-judge), not the source of truth.

### 3. Mac default → **keep `demucs` until MLX parity proven**

RoFormer is the corpus default only after Vast validation. Mac stays on Demucs for
`mac_analyze_loop` until MLX matches Vast bleed scores on the same 3 clips. Labeling
QA uses `scripts/separate.py --separator roformer` on Mac; bulk re-stem queues to Vast.

### 4. Schema → **infer from `versions_json` for now**

No `track_stems.separator` column in Phase B/C. `track_analysis.versions_json` already
keys by backend; add `instrumental_strategy: inst_ensemble|vocal_subtract` inside the
roformer version string when fallback fires. Revisit a column only if join pain shows up in P1 export.

---

## Open decisions (remaining)

| Decision | Status |
|----------|--------|
| Exact MVSEP checkpoint filenames | Pin in Phase A MSST smoke |
| `bleed_retry_threshold` numeric | Tune in Phase B on EDM/pop held-out |
| MSST vs `audio-separator[gpu]` on Vast | Phase A parity smoke on 3 clips |

---

## Results

*Fill in after Phase B validation.*

| Backend | Median vocal SDR | Median inst SDR | p50 track time (4090) | p50 track time (Mac MPS) | Notes |
|---------|------------------|-----------------|------------------------|---------------------------|-------|
| demucs (`htdemucs_ft`) | — | — | — | — | baseline |
| uvr chain | — | — | — | — | vocal-cleanup reference |
| roformer ensemble | — | — | **89.5 s** *(1 track, ~4 min source — Phase A smoke 2026-06-10, instance 4090/$0.43hr)* | ~45 min *(same track — QA only)* | candidate default |

---

## Critical files

| Action | Path |
|--------|------|
| **New** | `analysis/roformer_chain.yaml`, `analysis/adapters/roformer_chain_adapter.py`, `scripts/setup_roformer_separation.sh`, optional `scripts/roformer_restem_loop.py` |
| **Edit** | `analysis/pipeline.py`, `analysis/CLAUDE.md`, `scripts/separate.py`, `scripts/mac_analyze_loop.py`, `scripts/vast_loop.py`, `scripts/render_set_stems.py`, `requirements-audio.txt`, `scripts/vast_bootstrap.sh` |
| **Reuse** | `analysis/adapters/audio_separator_adapter.py`, `analysis/separation_config.py`, `analysis/models.py`, `analysis/persistence.py` |
| **Tests** | `tests/test_roformer_adapter.py` (mocked separator outputs) |
