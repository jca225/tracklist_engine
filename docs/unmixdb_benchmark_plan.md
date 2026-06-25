# UnmixDB benchmark + NMF baseline — build plan

**Goal:** stop guessing whether we're SOTA. Run our pipeline through **UnmixDB
v1.1** on the field-standard MAE metrics, reproduce **André/Schwarz/Fourer 2024
multi-pass NMF** as the target line, and produce one table: *our methods vs the
SOTA, same data, same units*. Executes item #1 of
[alignment_research_plan.md](alignment_research_plan.md) §9.

**Success criterion:** a committed `eval_bench` table with rows {Kim-DTW,
grid+matched-filter (ours), quad-fingerprint, NMF-baseline} × columns {placement
MAE, tempo/warp MAE, identity acc, coverage@abstain}, on the same UnmixDB split.

## 0. What already exists (reuse, don't rebuild)

- `external/unmixdb.py` — **GT loader is done**: `discover_root`, `good_mix_ids`
  (the v1.1 "good mixes" filter), `parse_labels` → `UnmixTrackSpan(track_idx,
  set_start_s, set_end_s, ref_start_s, ref_end_s, tempo_ratio, bpm)`,
  `UnmixMix(mix_id, mix_audio, track_audio, spans)`. The `.labels.txt` carries
  start/stop, **fade-in/out (cue regions)**, **speed (warp)**, bpm.
- `external/feature_series.py` — chroma/MERT feature series (cached).
- Our pipeline pieces: `refine_ref_offsets.detect_offset`, `continuity_refine`
  (grid stretch, stacking), `path_decode.decode_path` (warp), `landmark_fp.py`
  (fingerprint, to upgrade to quads).

## 1. Data acquisition (gated — big external fetch)

- **UnmixDB v1.1**, Zenodo record **1422385**. Tens of GB (1,931 synthetic
  mixes + source excerpts). Download to `~/data/unmixdb-v1.1/`.
- Start with the **good-mix subset** (`good_mix_ids`, silence < 1 s) to keep the
  first pass small; scale to full later.
- Decision needed: download full now, download a subset, or defer (see §7).

## 2. Components to build

### 2a. `eval_bench.py` — the harness (cornerstone)
- Input: a dataset adapter (UnmixDB now; BB12/BB11 later) yielding
  `(mix_audio, per-track ref audio, GT spans)`.
- Runs any **method** (a function `mix, refs → predicted spans`) and scores:
  - **Placement MAE/median** — |pred ref_start − GT ref_start| (and set_start).
  - **Tempo/warp MAE** — |pred tempo_ratio − GT speed|, % error (André's unit).
  - **Identity accuracy** — predicted track_idx vs GT (closed pool = the mix's
    own tracks; later, distractor pool for open-set).
  - **Cue/boundary** — error vs fade-in/out regions (within ±Δ tolerance).
  - **Abstain curve** — coverage vs accuracy when methods emit a confidence.
- Output: a tidy DataFrame + printed table; persists to `aux.db`-style or CSV.
- Match André's **MAE definition exactly** (read their eval section) so numbers
  are comparable — flag any divergence in the table.

### 2b. `nmf_baseline.py` — reference-conditioned multi-pass NMF (André v0)
- Model: mix magnitude spectrogram `V ≈ Σ_k (D_k · A_k)` where `D_k` = dictionary
  of source-track `k`'s spectrogram slices, `A_k` = activations over mix time.
- v0 (affine warp): build `D_k` at a small set of grid-derived stretches
  (`_grid_stretches`), multiplicative-update NMF for `A_k`, read off per-track:
  **placement** (argmax activation onset), **tempo** (winning stretch),
  **presence/gain** (activation envelope = our `gain_curve`).
- v1 (their "multi-pass"): iterate — refine warp from the activation diagonal,
  rebuild `D_k`, re-solve; handles loops/jumps via non-monotone activation.
- This doubles as both the **baseline-to-beat** and a **superposition-aware
  backbone** (it models the mix as a sum — our documented root cause).

### 2c. Method adapters (wrap what we have)
- `method_grid_mf` — our `detect_offset` + grid-lock + `path_decode` warp.
- `method_kim_dtw` — key-invariant beat-sync subsequence DTW (Kim baseline).
- `method_qfp` — `landmark_fp` upgraded to quad hashes (identity/instrumental).
- `method_nmf` — 2b.

## 3. Metrics ↔ which borrow they test

| Metric | Tests | Borrow that should move it |
|---|---|---|
| placement MAE | beat-sync features, GCC-PHAT refine | Kim, Schwarz–Fourer |
| tempo/warp MAE | grid-lock, NMF warp | ours, André |
| identity acc | quad fingerprint | Qfp |
| gain/presence | NMF activations | André |

## 4. Order of work

1. **Data**: fetch UnmixDB good-mix subset → `~/data/unmixdb-v1.1/`.
2. **`eval_bench.py`** + UnmixDB adapter (reuse `unmixdb.py`); wire
   `method_grid_mf` first → get OUR number on UnmixDB.
3. **`method_kim_dtw`** baseline → the published yardstick.
4. **`nmf_baseline.py`** v0 → the SOTA target line.
5. **Quad-fp** upgrade → identity/instrumental.
6. One table; compare to André's reported MAE; write verdict into the research
   plan §1.

## 5. Verification

- **No-data smoke**: a synthetic 2-track mini-mix (resample + overlap-add two
  source clips at known offset/stretch) → every metric and method runs end-to-end
  and recovers the known GT within tolerance, before any download.
- **Tiny-slice run**: 5–10 good mixes once data lands; eyeball MAE sanity.
- Headless-exec each module via `venvs/audio` like the notebook smoke pattern;
  `make check` before commit.

## 6. Risks

- **Download size/time** (tens of GB) → start with good-mix subset; cache.
- **NMF convergence / warp non-convexity** → v0 affine + grid-seeded; multi-pass
  is v1, don't block the harness on it.
- **MAE-definition mismatch with André** → read their eval; replicate exactly;
  if ambiguous, report both our def and theirs.
- **UnmixDB ≠ real mixes** (synthetic, affine warp, no real superposition mess) →
  it's the *comparable* benchmark; keep BB12/BB11 as the real-world cross-check.

## 7. Decision for the user

UnmixDB is a tens-of-GB Zenodo fetch. Options: (a) download the **good-mix
subset** now and start the harness against real data; (b) download the **full**
v1.1; (c) **build the harness + NMF against the synthetic smoke first**, defer the
download. (c) lets all the code land and smoke-pass today with zero bandwidth,
then the download just flips it onto real data.

## Critical files

- New: `workspaces/alignment_prototype/eval_bench.py`,
  `workspaces/alignment_prototype/nmf_baseline.py`.
- Reuse: `external/unmixdb.py`, `external/feature_series.py`,
  `refine_ref_offsets.py`, `continuity_refine.py`, `path_decode.py`,
  `landmark_fp.py`.
