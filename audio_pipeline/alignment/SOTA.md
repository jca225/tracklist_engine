# SOTA audio-alignment pipeline

**Current validation**: mean mix-IoU **0.891** on `tests/fixtures/bigbootie11_ground_truth.yaml`
(5 hand-annotated GT refs; per-row IoUs 0.625–1.000; argmax baseline 0.751; raw no-snap 0.872).

> Do **not** replace components here without re-running the eval and beating
> the baseline. Dropped experiments live in `_archive/README.md`.

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
│    (same family as production `measure_dtw.py`; equivalent to            │
│    subsequence DTW). Replaces naive argmax which gave descending cue     │
│    brackets like `[117-60s]`.                                            │
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
│    Write rows tagged `confidence_source='indicators_sota_v1'` at         │
│    section_idx 100000+ (offset from legacy 0-range) so the UI's          │
│    Ableton timeline can pick them up via the existing data path.         │
│    Label column populated from REFS.label so humans see meaningful text. │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Code layout

| File | Role |
|---|---|
| [`indicators_debug.py`](indicators_debug.py) | **The SOTA implementation.** Docstring lists the signal stack. |
| [`ref_position_viterbi`](indicators_debug.py) (function) | Step 6 — monotonic ref-position Viterbi |
| [`viterbi_universe`](indicators_debug.py) (function) | Step 2 — per-universe ref-selection Viterbi |
| [`_persist_sota`](indicators_debug.py) (function) | Step 8 — DB write |
| [`measure_dtw.py`](measure_dtw.py) | Production Viterbi with `ref_measure × pitch_shift` states + stem-mask coherence + structural priors. Same family as Step 6, more features. |
| [`viterbi_pipeline.py`](viterbi_pipeline.py) | Production pipeline wrapper — routes through `TRACKLIST_ALIGN_ALGO=viterbi`. |
| [`../analysis/canonical_cues.py`](../analysis/canonical_cues.py) | Populates `canonical_track_cue_points` by downloading full-song audio + running cue-detr at sensitivity=0.5. |
| [`populate_cue_fallbacks.py`](populate_cue_fallbacks.py) | Writes rough cue-based fallback rows for scraped tracks without a real SOTA prediction, so the UI shows the whole tracklist without paying full MERT cost. |
| [`eval.py`](eval.py) | Evaluation harness — scores `set_section_alignment` against `tests/fixtures/*_ground_truth.yaml`. |
| [`_archive/README.md`](_archive/README.md) | Dropped-experiment log with measured deltas + root-cause analysis. |

---

## 4. Schema touch-points

| Table | Column | Added / existing | Purpose |
|---|---|---|---|
| `track_audio` | `variant_tag` | Added this session | `'original'` / `'acappella'` / `'instrumental'` / `'remix'` — which version of the song this file IS |
| `canonical_track_cue_points` | (whole table) | Added this session | One cue-list per `track_id`, shared across all audio variants |
| `set_section_alignment` | `confidence_source` | Added this session | `'indicators_sota_v1'` vs `'legacy'` — so UI can filter |
| `set_section_alignment` | `label` | Added this session | Human label for UI display (avoids empty `text_excerpt` issue) |

---

## 5. How to run it

```bash
# One-time corpus prep — download original full-song audio for tracks
# whose scraped variant is an acappella/instrumental, run cue-detr on
# the originals, store in canonical_track_cue_points. Idempotent.
venvs/audio/bin/python -m audio_pipeline.analysis.canonical_cues \
    --set-id 2nvzlh2k

# Run the SOTA alignment on a set and persist to set_section_alignment.
venvs/audio/bin/python -m audio_pipeline.alignment.indicators_debug

# Optional: fill in cue-based fallback rows for scraped tracks that
# don't have real SOTA predictions, so the UI shows every track in
# the tracklist. Cheap (no MERT).
venvs/audio/bin/python -m audio_pipeline.alignment.populate_cue_fallbacks \
    --set-id 2nvzlh2k

# Evaluate against hand-annotated fixtures.
venvs/audio/bin/python -m audio_pipeline.alignment.eval \
    --db data/db/music_database.db

# View in the Streamlit UI — "Alignment review" page shows SOTA rows.
venvs/audio/bin/streamlit run ui/app.py
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

See [`_archive/README.md`](_archive/README.md). Summary:

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
- **Non-GT refs get cue-based fallback spans only** — real SOTA on 119 BB11 refs would need ~30 min MERT compute (cache-miss path). The cache is now wired in, so subsequent runs for the same set are fast.
- **Variant detection during scraping is not automated** — `track_audio.variant_tag` is hand-populated. Parsing scraped row text (`(Acappella)` / `(Instrumental)` markers) should fill this automatically for future sets.
- **Demucs has not been re-run on `variant_tag='original'` downloads** — Fray's new original (track_audio_id 122) has no stems. MERT alignment still uses the scraped-acappella variant (ta=3). Running demucs on originals would let the pipeline use dense cue-based stems everywhere.

---

## 10. File-level pointer for a future LLM / contributor

- Start at [`indicators_debug.py`'s module docstring](indicators_debug.py) — it lists the pipeline inline.
- Cross-reference with [`ROADMAP.md`](../ROADMAP.md) "CURRENT SOTA" header.
- Before proposing a change, read [`_archive/README.md`](_archive/README.md) to confirm you're not re-trying something already ruled out.
- Any proposed change is gated on `alignment/eval.py` IoU not regressing.
