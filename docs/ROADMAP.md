# Audio pipeline — queued work

Design sketches for features that are agreed in principle but not yet
implemented. Each entry is a pointer at the integration surface, not a
full spec — flesh out before building.

---

## CURRENT SOTA

The single canonical alignment entry point is
**[alignment/sota.py](alignment/sota.py)**.

Run:
```bash
venvs/audio/bin/python -m audio_pipeline.alignment.sota --set-id <set_id>
```

Writes rows to `set_section_alignment` with `confidence_source='sota_v2'`,
`section_idx = tracklist row_index`. The Streamlit "Alignment review" page
reads ONLY this source.

Mean mix IoU **0.891** on `tests/fixtures/bigbootie11_ground_truth.yaml`
(outer span only; the eval does not score `ref_segments` / loops yet).

Pipeline stages (see [SOTA.md](alignment/SOTA.md) for the full diagram):

1. Per-ref MERT cosine similarity, stem-routed by `version_tag`
2. Monotonic ref-position Viterbi — replaces argmax; `ref_t[mix_t]` is non-descending
3. Per-universe K+1-state Viterbi (states = universe refs + SILENCE) with mutual exclusion inside universe, cue-gated emission
4. Chromaprint-hit-density anchors reinforce emission within-universe
5. **Two-pass full-track exclusion** — pass 1 decodes the `full` universe with no cross-universe forcing; pass 2 uses the DECODED full path to force SILENCE in acappella/instrumental universes (principled at scale; raw-fp union over-suppresses when there are many `full` refs)
6. Per-ref earliest-run-near-cue cleanup (DJ plays each track once)
7. Canonical cue-detr bracket on the ref-position Viterbi endpoints → implied mix-side snap (skipped for `full` refs; regresses there)

Viterbi primitives (`viterbi_universe`, `ref_position_viterbi`, `_clean_path`,
cue-snap helpers) live in [alignment/indicators_debug.py](alignment/indicators_debug.py);
`sota.py` imports and composes them. `indicators_debug.py` also runs an
IoU-validation harness against the GT fixture and is no longer a persistence writer.

**Canonical cue points** come from [analysis/canonical_cues.py](analysis/canonical_cues.py):
cue-detr at `sensitivity=0.5` on the full-song `variant_tag='original'` audio,
stored in `canonical_track_cue_points` keyed by `track_id` — shared across all
variants (acapella / instrumental / full / remix).

Dropped experiments (do NOT re-try without re-eval): see [alignment/_archive/README.md](alignment/_archive/README.md).
  - MACD crossover transition bonuses — neutral
  - Wilder ADXR/DMI trust gate + entry/exit locks — degraded
  - Per-ref BPM matching penalty — broke on DJ tempo-shift
  - Argmax-based ref-position inference — non-monotonic; superseded by `ref_position_viterbi()`
  - `indicators_sota_v1` writer + `populate_cue_fallbacks.py` — superseded by `sota_v2` from `sota.py`
  - Non-SOTA pipelines (DTW / CCC / production viterbi / fragment / MERT-orchestrator) —
    pruned 2026-04-22; none beat the SOTA and all diverged on schema + UI integration.

---

## Deferred work (designed, not implemented)

Items below are queued ideas tied to specific known SOTA gaps. They do not
describe current code. Prior drafts that referenced now-deleted modules
(`correlate_pipeline`, `measure_dtw`, `viterbi_pipeline`, `fragment_pipeline`,
`features*`, `orchestrator`, `render/playback`, `identify/*`) were excised in
the 2026-04-22 prune — see git history for the old prose if you need it.

### 1. Loop detection in `set_section_alignment`

The current per-ref Viterbi decodes a single monotonic ref-position run, so
ref-restart loops (Good Grief: ref 32–113 s played, then ref 32–87 s
replayed at mix 1:46) collapse to one outer span. Candidate approach:
finite `backward_cost` on `delta < 0` transitions in `ref_position_viterbi`
and carry the resulting non-monotone path through into multiple
`measure_alignment` rows per tracklist row. The eval harness
[alignment/eval.py](alignment/eval.py) would need to score `ref_segments`
list-to-list rather than outer-span IoU to measure success.

### 2. Global ILP across rows

Cross-universe exclusion in `sota.py` is a greedy two-pass decode. A
mixed-integer program over `(mix_measure × universe → ref_or_silence)`
cells could enforce the per-bar "≤1 vocal + ≤1 instr-or-full" invariant
globally and resolve conflicts by total-confidence rather than pass order.

### 3. Render-and-compare eval

Per-set MFCC distance between a Rubber-Band-rendered reconstruction and
the actual mix audio is a no-yaml correctness signal. Needs a new
renderer — the prior `render/playback.py` was pruned when it stopped
consuming any writer's `measure_alignment` rows.

### 4. Variant detection from tracklist text

`track_audio.variant_tag` is currently hand-populated. Parsing scraped
row text for `(Acappella)` / `(Instrumental)` / `(Dub)` etc. at ingestion
time would fix sota.py's stem routing for rows where the DJ didn't mark
the variant in 1001tracklists but did actually play the instrumental.

### 5. Multi-hypothesis ref tournament

When a tracklist row has no explicit variant tag, sota.py defaults to
`full`. On BB11, Good Grief is labelled "Remix" but the DJ played the
instrumental stem — neither the tokenizer nor sota.py recover this.
Fix: 3-way tourney over (full, vocals, instrumental) picking the
hypothesis with the highest mean similarity. Triples MERT compute per
row; `.npz` cache makes re-runs tolerable.

### 6. Unified content-ID for unlabeled rows

Rows scraped as `ID - ID` produce no alignment. A sliding chromaprint
scan across the full mix, querying every corpus fingerprint, would
surface matches that bypass the tracklist entirely and write a separate
UI layer (`set_unlabeled_identifications`). Detailed design was deleted
with the old `identify/` module.

### 7. Cold-run MERT speed

First-time alignment on ~120 refs costs ~30 min of MERT compute. Two
avenues: GPU-batched inference (`mert_adapter` is MPS-aware but chunks
serially — batching across refs would parallelise), or a content-hashed
embedding cache keyed on `(audio_path, mtime)` that survives across
runs. `mert_align._cache_measure_embeddings` already writes `.npz` files
per-track — extend the key so the cache actually hits on ref audio files
that haven't changed.
