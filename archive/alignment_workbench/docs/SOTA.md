# SOTA audio-alignment pipeline

**Current validation**: mean mix-IoU **0.891** on `tests/fixtures/bigbootie11_ground_truth.yaml`
(5 hand-annotated GT refs; per-row IoUs 0.625–1.000; argmax baseline 0.751; raw no-snap 0.872).

> Do **not** replace components here without re-running the eval and beating
> the baseline. Dropped experiments live in [`alignment_archive.md`](alignment_archive.md).

---

## 1. The problem this pipeline solves

Given:
- A DJ mix audio file (stems already separated via demucs)
- A list of scraped reference tracks that (allegedly) play somewhere in the mix
- For each reference track: its own audio file, demucs stems, and a canonical cue-detr cue list

Produce, for each reference:
- `set_start_s`, `set_end_s` — mix-side time window where the ref plays
- Confidence score
- Ref-side bracket (which musical sections of the ref are played)

Three structural priors the pipeline exploits:
1. **Mutual exclusion within universe** — acapellas don't layer on other acapellas; same for instrumentals. Fulls don't layer with anything. So for each "universe" (acapella / instrumental / full), at any mix-measure at most ONE ref is active.
2. **Tracks play once, contiguously, near their scraped cue** — DJs don't re-enter the same track; the played span sits within ~±80 s of the scraped cue.
3. **Ref position advances roughly monotonically during a play** — no random jumps between mix_t and ref_t.

---

## 2. The pipeline — end to end

```
┌──────────────────────────────────────────────────────────────────────────┐
│  0. PREREQUISITE DATA                                                    │
│    - Mix audio + demucs stems  (set_audio, set_stems)                    │
│    - Ref audio + demucs stems  (track_audio, track_stems)                │
│    - Canonical cue-detr cues   (canonical_track_cue_points keyed by      │
│                                 track_id, computed on the FULL-song      │
│                                 variant at sensitivity=0.5)              │
│    - Chromaprint fingerprints  (track_fingerprints, set_fingerprint_hits)│
│    - Beat/measure grids        (set_measures, track_measures)            │
└──────────────────────────────────────────────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  1. STEM-ROUTED MERT SIMILARITY                                          │
│    Per ref, compute (N_ref, N_mix) MERT-layer-6 cosine-similarity matrix.│
│    Stems are routed by `version_tag`:                                    │
│      - 'instrumental' → compare vs mix's instrumental stem               │
│      - 'acappella'    → compare vs mix's vocals stem                     │
│      - 'full'         → compare vs full mix                              │
│    Embedding cached per (audio_path, layer, measure_grid, offset, dur)   │
│    in `data/cache/mert/*.npz` via `_cache_measure_embeddings`.           │
└──────────────────────────────────────────────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  2. PER-UNIVERSE VITERBI — WHICH REF IS PLAYING                          │
│    Three independent Viterbi decodes, one per universe.                  │
│      states:     {ref_1 … ref_K, SILENCE}                                │
│      emission:   w1·persistence(sim − pre_cue_baseline)                  │
│                + w2·ATR-normalised MACD histogram                        │
│                + w3·cross-sectional z within universe                    │
│      transitions: stay/exit/enter costs; forbidden cross-ref (must go    │
│                   via SILENCE); cue-gated (a ref state is only reachable │
│                   after its scraped cue_s).                              │
│    Output: per mix measure, which ref (or SILENCE) is active.            │
└──────────────────────────────────────────────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  3. WITHIN-UNIVERSE FINGERPRINT ANCHORS                                  │
│    For each ref, chromaprint hits from `set_fingerprint_hits` are        │
│    density-clustered (≥_FP_MIN_DENSITY hits within ±_FP_DENSITY_WINDOW_S)│
│    → confirmed measures → subtract _FP_ANCHOR_BONUS from emission cost.  │
│    Reinforces Phase-1's decision where chromaprint is confident.         │
└──────────────────────────────────────────────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  4. CROSS-UNIVERSE FULL-TRACK EXCLUSION                                  │
│    When a 'full'-variant ref is fingerprint-confirmed at measure t, the  │
│    instrumental and acapella universes are forced to SILENCE at t        │
│    (full tracks typically replace any instrumental/vocal underlay).      │
│    Main driver of Bastille's 0.837→0.966 IoU jump on BB11: Antoine's    │
│    full-track cluster forces Bastille to exit at ~185s.                  │
└──────────────────────────────────────────────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  5. EARLIEST-NEAR-CUE POST-PROCESS                                       │
│    Per ref: merge small SILENCE gaps inside the decoded span             │
│    (_MERGE_GAP_M measures), then keep only the EARLIEST merged run       │
│    that starts within ±_CUE_TOLERANCE_S of the scraped cue. Drops late   │
│    "re-entry" false positives (tracks play once, not twice).             │
└──────────────────────────────────────────────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  6. PER-REF MONOTONIC REF-POSITION VITERBI                               │
│    Second Viterbi decode, one per ref, over the (N_ref, N_mix) similarity│
│    matrix:                                                               │
│      states:     ref-measure index 0..N_ref-1                            │
│      emission:   1 − sim[ref, mix]                                       │
│      transitions:                                                        │
│        stay (Δ=0)         → cost 0   (DJ holds ref)                      │
│        advance (Δ=+1)     → cost 0   (normal forward play)               │
│        skip (Δ≥2)         → cost 0.3·Δ   (speed-up)                      │
│        backward (Δ<0)     → cost 2.0·|Δ| (loop-back, rare)               │
│    Output: `ref_measure_idx[mix_measure_idx]` — MONOTONIC by construction│
│    (subsequence-DTW family). Replaces naive argmax which gave descending │
│    cue brackets like `[117-60s]`.                                        │
└──────────────────────────────────────────────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  7. CANONICAL-CUE SNAP — FINAL MIX-SIDE BOUNDARIES                       │
│    For each (acapella/instrumental) predicted span:                      │
│      ref_t_start = per_ref_meas_times[median_argmax_over_first_3_measures]│
│      ref_t_end   = per_ref_meas_times[median_argmax_over_last_3_measures] │
│      ref_cue_start = nearest_cue(ref_t_start, canonical_cues[track_id])   │
│      ref_cue_end   = nearest_cue(ref_t_end,   canonical_cues[track_id])   │
│      snap_mix_start = pred_start − (ref_t_start − ref_cue_start)          │
│      snap_mix_end   = pred_end   − (ref_t_end   − ref_cue_end)            │
│    'full' refs are NOT snapped (empirically regressed on Antoine at BB11).│
│    Final prediction = Viterbi-snap boundaries.                            │
└──────────────────────────────────────────────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  8. PERSIST TO set_section_alignment                                      │
│    Write one row per aligned tracklist ref with                          │
│    `confidence_source='sota_v2'` and `section_idx = tracklist row_index`.│
│    Prior rows for the set (any source) are deleted first — the UI reads  │
│    only this source. Label column is populated so the timeline shows     │
│    human-readable text.                                                  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Code layout

| File | Role |
|---|---|
| **[`sota.py`](../audio_pipeline/alignment/sota.py)** | **The canonical SOTA orchestrator.** Loads every tracklist ref with audio + measures, runs the full stack, persists `confidence_source='sota_v2'` rows. This is the single entry point. |
| [`indicators_debug.py`](../audio_pipeline/alignment/indicators_debug.py) | Holds the Viterbi primitives (`viterbi_universe`, `ref_position_viterbi`, `_clean_path`, `_bracket_cue_points`, `_snap_via_position`) that `sota.py` imports. Also runs the IoU validation against the GT fixture. No longer a persistence writer. |
| [`ref_position_viterbi`](../audio_pipeline/alignment/indicators_debug.py) (function) | Step 6 — monotonic ref-position Viterbi |
| [`viterbi_universe`](../audio_pipeline/alignment/indicators_debug.py) (function) | Step 2 — per-universe ref-selection Viterbi |
| [`canonical_cues.py`](../audio_pipeline/analysis/canonical_cues.py) | Populates `canonical_track_cue_points` by downloading full-song audio + running cue-detr at sensitivity=0.5. |
| [`eval.py`](../audio_pipeline/alignment/eval.py) | Evaluation harness — scores `set_section_alignment` against `tests/fixtures/*_ground_truth.yaml`. |
| [`alignment_archive.md`](alignment_archive.md) | Dropped-experiment log with measured deltas + root-cause analysis. |

---

## 4. Schema touch-points

| Table | Column | Added / existing | Purpose |
|---|---|---|---|
| `track_audio` | `variant_tag` | Added this session | `'original'` / `'acappella'` / `'instrumental'` / `'remix'` — which version of the song this file IS |
| `canonical_track_cue_points` | (whole table) | Added this session | One cue-list per `track_id`, shared across all audio variants |
| `set_section_alignment` | `confidence_source` | Added this session | `'sota_v2'` (the only writer) — UI reads this source exclusively |
| `set_section_alignment` | `label` | Added this session | Human label for UI display (avoids empty `text_excerpt` issue) |

---

## 5. How to run it

```bash
# One-time corpus prep — download original full-song audio for tracks
# whose scraped variant is an acappella/instrumental, run cue-detr on
# the originals, store in canonical_track_cue_points. Idempotent.
venvs/audio/bin/python -m audio_pipeline.analysis.canonical_cues \
    --set-id 2nvzlh2k

# SOTA alignment on a set — writes confidence_source='sota_v2' rows
# keyed on tracklist row_index. This is the ONLY alignment writer.
venvs/audio/bin/python -m audio_pipeline.alignment.sota \
    --set-id 2nvzlh2k

# Validate against the ground-truth fixtures.
venvs/audio/bin/python -m audio_pipeline.alignment.eval \
    --db data/db/music_database.db

# Internal consistency check — reproduces the sota_v2 result on the 5
# GT refs and prints per-row IoU vs ground truth. Not a persistence
# writer anymore, just a debug/validation harness.
venvs/audio/bin/python -m audio_pipeline.alignment.indicators_debug

# View aligned sets in the browser DAW. See browser_daw/README.md for setup.
# (The legacy Streamlit "Alignment review" app is archived under archive/ui/.)
./browser_daw/run_browser_daw.sh
```

---

## 6. Key parameters (in `indicators_debug.py`)

| Constant | Value | Role |
|---|---|---|
| `_EMIT_PERSIST_WEIGHT` | `1.5` | Weight of persistence signal in per-universe emission |
| `_EMIT_MACD_WEIGHT` | `1.0` | Weight of MACD-histogram momentum in emission |
| `_EMIT_CS_Z_WEIGHT` | `0.5` | Weight of cross-sectional z (winner-picker within universe) |
| `_SILENCE_EMIT` | `1.2` | Bar that refs must clear to beat SILENCE |
| `_MERGE_GAP_M` | `10` | Measures of SILENCE gap absorbed inside a ref's run |
| `_MIN_DURATION_M` | `5` | Minimum surviving run length |
| `_CUE_TOLERANCE_S` | `80.0` | Max allowed offset between scraped cue and chosen run start |
| `_FP_MIN_DENSITY` | `2` | Chromaprint hits required in density window to anchor |
| `_FP_DENSITY_WINDOW_S` | `10.0` | ± seconds around each measure for density count |
| `_FP_ANCHOR_BONUS` | `1.5` | Emission cost reduction at fingerprint-confirmed measures |
| `_CUE_SNAP_MAX_SHIFT_S` | `0.0` | (Kept but disabled — SOTA snap via Viterbi, not argmax) |
| cue-detr sensitivity | `0.5` | Used by `canonical_cues.py` instead of Wilder default 0.9 |

---

## 7. How we validate

```bash
venvs/audio/bin/python -m audio_pipeline.alignment.eval \
    --db data/db/music_database.db
```

Reads every `tests/fixtures/*_ground_truth.yaml`, scores the current
`set_section_alignment` rows against the hand-annotated spans.
Metrics:
- **mean mix-IoU** (the primary baseline to beat): 1.0 = pred span matches GT exactly, 0.0 = no overlap.
- **span inflation**: `pred_duration / gt_duration`. >>1 means over-reported (old CCC ran 17.5×); <<1 means under-reported.
- **row recall**: fraction of GT tracks with any predicted row.

Current SOTA on BB11: mean IoU **0.891**, span inflation near 1.0, recall 5/5 on GT refs.

---

## 8. Dropped experiments (for future sessions — DO NOT RE-TRY WITHOUT EVAL)

See [`alignment_archive.md`](alignment_archive.md). Summary:

| Experiment | Why dropped |
|---|---|
| MACD crossover transition bonuses | Neutral (+0.013) — bonus drowned out by emission gradient |
| Wilder ADXR/DMI trust gate + entry/exit locks | Degraded — ADXR never clears Wilder's 20/25 on MERT-sim data; timescale mismatch |
| Per-ref BPM matching penalty | Broke Gnash IoU 0.857→0.381 — DJs tempo-shift refs routinely |
| Argmax-based ref-position inference | Non-monotonic → descending ref-cue brackets like `[117-60s]`; replaced by Step 6 Viterbi |
| Cue-detr at default sensitivity=0.9 | Acapella/instrumental-only audio has few surviving cues — replaced by sens=0.5 on full-song originals |

---

## 9. Known limits + next steps

- **MERT argmax remains unreliable for position inference even in short spans** — CRJ's 15-s play has all 3 position samples collapse to ref_t=29s, so the snap becomes degenerate (same as no-snap). Partial symptoms still visible at IoU 0.625.
- **Full variants don't benefit from cue snap** — tested Antoine, regressed 0.950→0.826. Skipped in the current pipeline.
- **Non-GT refs now get real alignment** — as of sota.py (2026-04-22), every
  tracklist row with audio + measures is aligned by the full stack. The prior
  cue-based 60 s placeholder path (`populate_cue_fallbacks.py`) is deleted.
  First-run MERT compute for ~120 refs is ~30 min cold; `.npz` cache amortises
  re-runs to minutes.
- **Variant detection during scraping is not automated** — `track_audio.variant_tag` is hand-populated. Parsing scraped row text (`(Acappella)` / `(Instrumental)` markers) should fill this automatically for future sets.
- **Demucs has not been re-run on `variant_tag='original'` downloads** — Fray's new original (track_audio_id 122) has no stems. MERT alignment still uses the scraped-acappella variant (ta=3). Running demucs on originals would let the pipeline use dense cue-based stems everywhere.

---

## 10. File-level pointer for a future LLM / contributor

- Start at [`sota.py`'s module docstring](../audio_pipeline/alignment/sota.py) — it lists the pipeline
  stages and matches the diagram above.
- `sota.py` is the only persistence writer. `indicators_debug.py` holds the
  Viterbi primitives it imports + an offline IoU harness against the GT
  fixture (no DB writes).
- Cross-reference with [`ROADMAP.md`](ROADMAP.md) "CURRENT SOTA" header.
- Before proposing a change, read [`alignment_archive.md`](alignment_archive.md)
  to confirm you're not re-trying something already ruled out.
- Any proposed change is gated on `audio_pipeline/alignment/eval.py` IoU not regressing.
