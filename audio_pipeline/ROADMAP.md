# Audio pipeline — queued work

Design sketches for features that are agreed in principle but not yet
implemented. Each entry is a pointer at the integration surface, not a
full spec — flesh out before building.

---

## CURRENT SOTA (as of 2026-04-21)

The single alignment pipeline is [alignment/indicators_debug.py](alignment/indicators_debug.py) —
MERT-embedding per-universe Viterbi + ref-position Viterbi + canonical-cue-detr
snap. Writes `set_section_alignment` rows tagged `confidence_source='indicators_sota_v1'`;
the UI reads exactly that source.

Run:
```bash
venvs/audio/bin/python -m audio_pipeline.alignment.indicators_debug
```

Mean mix IoU **0.891** on `tests/fixtures/bigbootie11_ground_truth.yaml`
(outer span only; the eval does not yet score `ref_segments` / loops).

**Canonical cue points** come from [analysis/canonical_cues.py](analysis/canonical_cues.py):
cue-detr is run once on the **full-song** (`variant_tag='original'`) version of
each canonical track at `sensitivity=0.5`, and the results are stored in
`canonical_track_cue_points` keyed by `track_id` — shared across all variants.

Dropped experiments (do NOT re-try without re-eval): see [alignment/_archive/README.md](alignment/_archive/README.md).
  - MACD crossover transition bonuses — neutral
  - Wilder ADXR/DMI trust gate + entry/exit locks — degraded
  - Per-ref BPM matching penalty — broke on DJ tempo-shift
  - Argmax-based ref-position inference — non-monotonic; superseded by `ref_position_viterbi()`
  - Non-SOTA pipelines (DTW / CCC / production viterbi / fragment / MERT-orchestrator) —
    pruned 2026-04-22; none beat the SOTA and all diverged on schema + UI integration.

---

## Alignment revamp — landed 2026-04-21

Goal: capture "what a listener hears" — which measures of which ref
tracks play at which mix times, at what pitch shift and tempo ratio,
on which stems. Baseline before revamp was mean mix-IoU=0.090 with
17.5× span inflation on Big Bootie 11 (CCC stitcher runaway).

**Delivered:**

- `_stitch_play` cap tightened from 2.5× → 1.2× ref duration; stitched
  successors gated to a ±180s cue-anchored window ([alignment/correlate_pipeline.py](alignment/correlate_pipeline.py)).
- Schema migrations: `track_measures`, `set_measures`,
  `measure_alignment`, `set_playback_score`, `track_fingerprints`,
  `set_fingerprint_hits`, `track_sections` ([web_crawler/database/schema.sql](../web_crawler/database/schema.sql)).
- Evaluation harness: yaml-IoU + span-inflation metric per row ([alignment/eval.py](alignment/eval.py)).
- Playback renderer: Rubber Band-based `measure_alignment` → WAV + MFCC
  reconstruction distance ([render/playback.py](render/playback.py)).
- Pre-summed `instrumental = drums+bass+other` stem at analysis time
  ([analysis/adapters/demucs_adapter.py](analysis/adapters/demucs_adapter.py)); one-shot backfill for existing analysed
  tracks ([adapters/instrumental_backfill.py](adapters/instrumental_backfill.py)).
- 3-hypothesis tournament (`full` / `instrumental` / `acappella`)
  replacing the 4-stem one; tag-gated hypothesis when tracklist text
  carries `(Acappella)` / `(Instrumental)` ([alignment/pipeline.py](alignment/pipeline.py), [alignment/orchestrator.py](alignment/orchestrator.py)).
- Chromaprint fingerprinting: ref ingestion + sliding mix-scan with
  `maxlength=7200s` (default 120s too short for DJ mixes) ([identify/acoustid_adapter.py](identify/acoustid_adapter.py),
  [identify/scan.py](identify/scan.py)).
- Context-aware measure-DTW (Viterbi) with structural transition
  priors, pitch-shift state, stem-mask coherence ([alignment/measure_dtw.py](alignment/measure_dtw.py),
  [alignment/measure_features.py](alignment/measure_features.py), [alignment/viterbi_pipeline.py](alignment/viterbi_pipeline.py)).
- Orchestrator routes through Viterbi via `TRACKLIST_ALIGN_ALGO=viterbi`;
  writes `measure_alignment` rows + derived `set_section_alignment`
  summary ([alignment/orchestrator.py:align_set_viterbi](alignment/orchestrator.py)).

**Measured deltas:**

| metric (Big Bootie 11) | baseline | CCC + tightened stitcher | Δ |
|---|---|---|---|
| mean mix IoU          | 0.090 | 0.503 | **+5.6×** |
| mean span inflation   | 17.5× | 0.66× | near-perfect |
| matched GT rows / 6   | 6     | 5     | -1 (One Week not yet in partial re-run) |

Viterbi-on-full-set run pending completion at time of writing;
re-eval expected to reduce the remaining IoU loss on loop-back cases
(e.g. Good Grief cuts short at 17s in CCC, should be 162s via
section-start bonus).

**Queued (not yet implemented):**

- Global ILP across rows (per-bar per-stem energy budgets; mashup
  co-placement).
- Fingerprint hit density → Viterbi candidate windowing (narrow the
  search to ref sections the fingerprint scan already localised).
- Render-and-compare eval: per-set MFCC distance between
  `reconstructed.wav` and actual mix audio, as a no-yaml metric.

---

## GPU feature-extraction backend (prototype, A/B behind a flag)

**Status**: landed as an opt-in prototype. Default remains librosa.

**How to toggle**

```bash
# Default (librosa, CPU — matches paper, reference implementation)
./venvs/audio/bin/python -m audio_pipeline.align_main --set-id 2nvzlh2k

# GPU path (torchaudio + MPS on Apple Silicon)
TRACKLIST_ALIGN_BACKEND=torch ./venvs/audio/bin/python -m audio_pipeline.align_main --set-id 2nvzlh2k

# Pin torch to CPU (useful to isolate CPU-vs-CPU parity from MPS differences)
TRACKLIST_ALIGN_BACKEND=torch TRACKLIST_ALIGN_DEVICE=cpu \
  ./venvs/audio/bin/python -m audio_pipeline.align_main --set-id 2nvzlh2k
```

**What's GPU-accelerated**

- STFT (torchaudio `Spectrogram`)
- Chroma via librosa's pitch-class filterbank × STFT on device
- MFCC (torchaudio `MFCC`)
- Beat-sync mean-pool (custom, cumsum-based)

**What stays on CPU (for now)**

- Audio load — librosa's resampler, preserves file semantics.
- HPSS — median filtering, no clean MPS path. Runs once per stream;
  results are moved to device for downstream work.
- Beat tracking — `librosa.beat.beat_track`, tiny.

**Known parity gap**

- Librosa `chroma_cens` is CQT-based with windowed CENS normalisation.
  The torch path uses STFT × librosa's chroma filterbank + log
  compression + per-frame L2. Column-wise cosine similarity on a
  chord fixture is ~0.5–0.8 — same energy, different contour. The
  A/B on Big Bootie measures whether this shifts match rates.
- MFCC agrees to ~1e-3 (same DCT-II basis, same mel filterbank count).

**Next steps if the A/B looks promising**

- Install `nnAudio` to get CQT on GPU and close the chroma_cens parity
  gap. Adds one dep, no code changes beyond `features_torch.py`.
- Port HPSS to torch (custom median filter) for a full on-device path.
- Batch the 12-shift DTW on GPU — see `dtw.py`, each shift is
  independent and runs a separate `librosa.sequence.dtw` call today.

**Evaluation protocol**

Pick one or two Big Bootie volumes with full prerequisites (audio +
stems + measures present). Run both backends, compare:
- total wall time (alignment stage only)
- mean / median `match_rate` across aligned rows
- number of rows with `match_rate >= 0.4`
- transposition histogram

Same inputs, same `set_section_alignment` schema — the only variable
is the env var.

**Critical files**

- [audio_pipeline/alignment/features.py](alignment/features.py) — backend dispatch
- [audio_pipeline/alignment/features_torch.py](alignment/features_torch.py) — GPU path
- [tests/test_audio_pipeline_features_torch.py](../tests/test_audio_pipeline_features_torch.py) — parity tests (CPU-pinned for portability)

---

## Batched PyTorch DTW (deferred — correctness risk outweighs tonight's gain)

**Status**: designed, not implemented. Skipped during the overnight
optimization sprint because a custom DTW kernel is the single change
most likely to silently corrupt alignment results. The
feature-extraction port already hits the diminishing-returns part of
the curve once the feature cache is populated (cache-hit rows skip
feature extraction entirely, so their remaining cost is DTW + I/O).

**Plan when it's time**

- Implement as `audio_pipeline/alignment/dtw_torch.py` behind a
  `TRACKLIST_ALIGN_DTW=torch` env var — keep the librosa Cython path
  as reference.
- Anti-diagonal iteration: for k in `range(1, N+M-1)`, compute all
  cells on diagonal k in parallel. Only 2 prior diagonals needed in
  memory. ~600 diagonal iterations for an N=400 × M=200 matrix.
- Batch all 12 chroma shifts as a single (12, N, M) cost tensor — one
  `einsum` builds the full cost matrix, the DTW recurrence operates on
  the batch dimension in parallel.
- Traceback the winning shift only (not all 12) — path-length per
  shift needed for normalized-cost comparison, so run trace on each
  but path reconstruction only on the winner.
- Validation: parity against `librosa.sequence.dtw` on a fixed set of
  random feature matrices. Paths must match to the step.

**Expected speedup**: ~10–15× on the DTW stage, which is roughly 25% of
per-row cost today. Net per-row: ~20–25% faster on cold rows, ~negligible
on cache-hit rows (DTW is already the bottleneck there).

---

## Feature cache (biggest remaining speedup per row)

**Status**: designed, not implemented. Measured numbers justify it as
the single biggest available win — moves re-runs of alignment from
~42 s/row CPU or ~28 s/row GPU down to **near-zero** on the ref side.

**Why it's worth it**

Per-row breakdown on the CPU baseline: feature extraction (HPSS + CENS
chroma + MFCC + beat-sync) is ~60% of wall time; DTW is ~25%; audio
load/decode is ~7% (measured 1.39 s for a 244 s m4a via audioread,
0.11 s for wav via soundfile — m4a transcode win is real but small);
overhead is the rest. Features are deterministic given audio + code
version + backend; caching them eliminates the biggest per-row cost on
every re-run where we're iterating on DTW, cue extraction, the
tournament policy, or Stage-5.

**Key design**

```
cache_key = blake2b(
    audio_path + mtime(audio_path)
    + offset_s + duration_s
    + backend                # 'librosa' | 'torch'
    + FEATURES_VERSION       # bump when feature code changes
    + beat_frames_hash       # for compute_on_beats only
).hexdigest()[:24]

cache_path = data/cache/features/{cache_key}.npz
```

The `FEATURES_VERSION` constant is the escape hatch: any breaking change
to `_features_from_signal` (e.g. swapping HPSS margin, changing chroma
normalisation) increments the version, invalidating stale entries
without manual cleanup. mtime catches ref audio replacement.

**Storage**

`.npz` with `chroma` (float32, ~5 KB/beat × 12 pitch classes per track),
`mfcc` (float32, same), `beat_times_s`, `beat_frames`. Per-track
footprint ≈ 200 KB. At ~3,000 refs × 5 streams (full + 4 stems) × 2
backends × ~200 KB ≈ 6 GB — fits comfortably on the audio drive.

**Invalidation hooks**

- `FEATURES_VERSION` bump — already the natural workflow when changing
  features.py.
- mtime on the source audio — catches file replacement.
- Env var `TRACKLIST_ALIGN_FEATURE_CACHE=0` — disable entirely for
  debugging numerical differences.

**Concurrency**

Multiprocessing (next item on the list) will race on cache writes.
Mitigate with atomic write: render to `{cache_path}.tmp.{pid}` then
`os.replace` — rename is atomic on POSIX and macOS.

**Where it plugs in**

One decorator in `features.py` wraps `compute` and `compute_on_beats`;
`features_torch.compute` / `compute_on_beats` share the same wrapper so
the GPU path benefits equally. No changes needed in `pipeline.py`,
`orchestrator.py`, or downstream — they're unaware of the cache.

---

## Multiprocessing across rows

**Status**: designed, not implemented. Complements the cache — one is
cold-run throughput, the other is re-run latency.

`align_set` iterates cue-anchored rows sequentially. Each row is fully
independent (different ref, same set audio — cheap to share). A
`multiprocessing.Pool(n_workers)` around the per-row body should give
~3–5× throughput on an 8–10 core machine. Gotchas:

- SQLite writes must be serialised. Either collect results and write
  from the main process, or use `PRAGMA journal_mode=WAL` and a single
  connection per worker.
- MPS contexts can't be shared across processes — workers would each
  create their own CUDA/MPS context (small memory overhead, fine).
- librosa's internal thread pools + OpenBLAS can oversubscribe if you
  set `n_workers = cpu_count`. Set `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`
  in workers.

Less compelling once the feature cache lands, because the feature cache
turns re-runs into I/O-bound work where multiprocessing's wins are
smaller. Worth it for the *first* pass over a large corpus.

---

## m4a decode fix (low priority)

**Status**: investigated, deferred as not worth the effort *relative to
the cache*. Measured savings: ~5–7% of per-row time.

libsndfile 1.2.2 (bundled with soundfile 0.13.1) doesn't decode
MP4/AAC, so `librosa.load` on `.m4a` falls through to `audioread` →
ffmpeg subprocess. Stems are `.wav` and hit libsndfile directly.

Clean fix is to transcode refs to FLAC on ingestion in
`audio_pipeline/adapters/` — one-time cost, eliminates the fallback
forever. Storage inflates ~2.5× (AAC → FLAC) per ref, still trivial
on the current drive budget.

Not worth doing until the feature cache lands, because the cache
absorbs the load cost anyway once a ref is seen once.

---

## Unified content-ID + long-held-track detector

**Problem this solves (two symptoms of the same gap)**

1. *Long holds fall off the cue-window cliff.* Stage-1 DTW only searches a
   ~90 s window around each scraped cue (`WINDOW_PAD_S` in
   `audio_pipeline/alignment/orchestrator.py`). If a track is held longer
   than the window — held-vocal loops, long breakdowns — the reported
   set-side span is truncated at the window ceiling. Good Grief in Big
   Bootie hits this.
2. *Unidentified sections stay unidentified.* Tracklist rows with
   `row_kind='id'` or no scraped `track_id` produce no alignment at all.
   If the DJ actually played a track that's labeled in some *other* set
   we've scraped, we currently have no way to surface that.

Both need the same piece of infrastructure: a **sliding fingerprint scan
over the full DJ mix**, querying a corpus-wide index of every reference
track we've downloaded. Build it once; it serves both use cases.

**Approach**

- **Fingerprint library**: [chromaprint](https://acoustid.org/chromaprint)
  via the `pyacoustid` / `fpcalc` binding. Mature, fast, tolerant to EQ
  and moderate time-stretch. Shazam is closed; Acoustid's MusicBrainz
  catalog is spotty on EDM/remixes. Our own corpus of scraped refs is a
  better index for the job.
- **Compute fingerprints at ref ingestion time**, not at query time.
  Extend `audio_pipeline/analysis/` with an `acoustid_adapter.py` that
  runs `fpcalc` on each downloaded ref and persists the fingerprint.
- **Scan the full mix** in fixed-size hops (start with 10 s window, 2 s
  hop) against the in-memory fingerprint index. Rank hits per window by
  chromaprint's standard similarity score.
- **Confidence gating**: require a minimum hit-density (e.g. ≥N
  consecutive hops with the same matched track) before promoting a
  detection. This is what filters crowd noise, MC shout-outs, transitions.

**Where it plugs into the existing pipeline**

New stage, run *after* Stage-1 but *before* the Streamlit UI build:

```
Stage-1 (windowed DTW)  →  Stage-1b (full-mix fingerprint scan)  →  Stage-5 refinement
```

Stage-1b reads from `set_section_alignment` and `set_timeline`, identifies
(a) gaps / unaligned stretches and (b) rows where `set_end_s - set_start_s`
is suspiciously close to the cue window ceiling, scans those regions, and
emits new rows:

- For (a): insert into a new `set_unlabeled_identifications(set_id,
  start_s, end_s, matched_track_id, confidence, score)` table. UI renders
  them as a separate "auto-identified" layer on the timeline.
- For (b): extend the existing `set_section_alignment` row's `set_end_s`
  once fingerprint evidence confirms the hold continues. Keep a
  `duration_source` column so we can tell scraped vs. auto-extended.

**Schema additions**

```sql
CREATE TABLE track_fingerprints (
    track_id TEXT NOT NULL,
    variant_tag TEXT NOT NULL DEFAULT 'original',  -- 'original', 'instrumental', ...
    fingerprint BLOB NOT NULL,                     -- chromaprint raw fp
    duration_s REAL NOT NULL,
    PRIMARY KEY (track_id, variant_tag)
);

CREATE TABLE set_unlabeled_identifications (
    set_id TEXT NOT NULL,
    start_s REAL NOT NULL,
    end_s REAL NOT NULL,
    matched_track_id TEXT NOT NULL,
    confidence REAL NOT NULL,      -- 0..1, post hit-density gate
    score REAL NOT NULL,           -- raw chromaprint score
    PRIMARY KEY (set_id, start_s, matched_track_id),
    FOREIGN KEY (set_id)           REFERENCES dj_sets(set_id)           ON DELETE CASCADE,
    FOREIGN KEY (matched_track_id) REFERENCES canonical_tracks(track_id) ON DELETE CASCADE
);
```

**Critical files to modify when building**

- `audio_pipeline/analysis/` — new `acoustid_adapter.py`, hook into
  `pipeline.analyze_track` to compute fingerprint alongside stems / MERT.
- `audio_pipeline/alignment/orchestrator.py` — add Stage-1b call after
  the existing per-row loop, or fork into a new `audio_pipeline/identify/`
  subpackage if the scope grows.
- `web_crawler/database/schema.sql` — add the two tables above.
- `ui/app.py` — render auto-identifications as a distinct timeline layer.

**Honest limits**

- Mashup sections defeat fingerprinting the same way they defeat DTW —
  chromaprint on 4-deck layered audio is fundamentally underdetermined.
  This is not a rescue for the hard rows.
- Only finds tracks *already in our corpus*. Unreleased ID tracks that
  appear in exactly one mix stay unidentified. That's a corpus-growth
  problem, not a detector problem.
- Computational cost: a 60-min mix at 10 s window / 2 s hop produces
  ~1800 fingerprint queries. Cheap per-query but adds up; budget for a
  GPU-batched variant if the index grows past a few thousand refs.

**Feedback loop (nice side effect)**

Every confident auto-identification becomes a new cue-anchored segment
Stage-1 can then align properly. Run content-ID → re-run Stage-1 →
timeline gets richer. Converges in one or two passes.

**Scope boundary**

This TODO is the unified detector. Version-disambiguation (deciding
*which* remix / instrumental of a known track was played) is a separate,
smaller problem — revisit after the content-ID pass is working, because
fingerprint queries against a per-`track_id` variants library reuse the
same adapter.
