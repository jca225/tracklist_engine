# looptrace — NOTES

Running log for the acappella loop-tracing effort (non-monotonic alignment of DJ-mix
vocals). Plan of record: the phased brief (Phase 0 recon → Phase 1 ill-posedness audit
→ Phase 2 loop-collapse → Phase 3 landmark Hough + segment-cover DP → Phases 4–5
conditional patches). Context doc: [docs/loop_tracing_research_brief.md](../../../docs/loop_tracing_research_brief.md).

Module home: `workspaces/alignment_prototype/looptrace/` — inside the existing
alignment workspace so it can import the legacy machinery (`ref_fibers`,
`path_decode.trajectory_acc`, `landmark_fp`) without path gymnastics, while staying a
separate module the legacy decoder never imports. Runnable side-by-side.

---

## Phase 0 — repo recon (2026-07-06)

### The existing decoder(s)

- **`workspaces/alignment_prototype/path_decode.py`** — the legacy segment decoder this
  effort must beat. Viterbi over ref-offset states (`_viterbi`, lines ~99–149; stay-on-
  diagonal free, jump costs `lam`, `lam_back` optional — the asymmetric sweep HURT,
  don't revisit). Emissions = windowed normalized matched filter
  (`_scores_at_stretch`, ~152–178; 12 s window, 2 s hop) on chroma or HuBERT
  (`--feature`, acappella routes to HuBERT layer 9). `decode_path` (~181–253) returns
  the segment list. **`--eval` slices the mix at GT `set_start_s` — the CLI eval IS the
  oracle-placement baseline** from the research brief.
- **`joint_ref_decode.py`** — applies `decode_path` post-infer, feature-routed
  (acappella→HuBERT, else chroma), writes `ref_segments` into the predicted timeline
  JSON (real-placement numbers).
- Matched-filter scalar ref-offset: `refine_ref_offsets.py` (single-window, coin-flips
  repeats — known-failed approach #1). Continuity stack: `continuity_refine.py`
  (no-op for acappella — 93% of acappella spans non-linear).

### Existing landmark fingerprinting (REUSE for Phase 3)

- **`landmark_fp.py`** — Shazam-style constellation already implemented:
  `constellation()` (STFT NFFT=2048, FHOP=512, peak_size=19, 60 dB floor),
  `hashes()` (peak-pair `(f1, f2, dt)` keys, fan-out 8, dt_max 80 frames),
  `fp_offset()` (offset vote histogram over a stretch band → offset, votes, stretch,
  sharpness), `LandmarkFingerprint.from_blob()` (cached blobs in DB
  `track_fingerprints`, kind=landmark).
- **`mix_fp_hits.py`** — mix-side scanning: `span_from_offset_votes` (vote-extent →
  set_start), `offset_candidates` (top-K diagonals). Thresholds: HIT_MIN_VOTES=25,
  HIT_MIN_SHARPNESS=1.2, z<1.0 band gate.
- So Phase 3's "point cloud" step is a **modification**, not a from-scratch build: the
  existing pipeline collapses matches into an offset *histogram* per stretch; we need
  to keep the raw `(t_mix, t_song)` match points, transform to intercepts at the known
  slope, and do 1D Hough + segment-cover DP over them. Known caveat from the corpus:
  the full-mix fp is **weak on vocals** — Phase 3 must hash `mix_vocals.flac` against
  the ref *vocal stem* (axis routing per `harness/axes.py`), and fixtures must test
  repitched (tempo+pitch) vs key-locked stretch, since ~31% of BB11 acappellas are
  re-pitched (breaks fixed-frequency hashes → need coarse freq quantization or
  Panako-style ratio-invariant hashes; decide on fixtures).

### Fiber machinery (REUSE for Phase 1)

- **`ref_fibers.py`** — self-repeat equivalence classes over the ref:
  `compute_fibers` / `compute_fibers_soft` (HuBERT + RMS silence-gate
  (ratio 0.35) + `_long_repeats` diagonal scan (min 6 s, thresh 0.5) +
  `_avg_linkage` grouping (thresh 0.5)). Output: per-frame labels @ ~8 Hz,
  membership μ, per-fiber confidence. **Fibers are HuBERT-based, never chroma**
  (chroma blobs the whole track into one fake fiber).
- Phase 1 extends this: fibers say *which regions repeat*; the audit adds
  **sample-accurate cross-correlation within each fiber** to split CLONE
  (production copy-paste, residual at noise floor) from DISTINCT TAKE.

### Eval harness, test mixes, answer keys

- **Frozen legacy metric:** `path_decode.trajectory_acc` (lines ~293–325) — sample
  mix-times at 1 s steps across the span, predicted ref(mix_t) via piecewise-linear
  interpolation of the segment list, correct if within **2.0 s** of GT ref(mix_t);
  span score = fraction correct. Fiber-aware variant additionally credits a prediction
  landing in the same fiber as GT. Span classes via `_span_class` (~328–336):
  `loop` (is_loop) / `multiseg` (has ref_segments) / `oddratio` (tempo_ratio outside
  0.9–1.15) / `linear`. **This metric is frozen**; all phases also report the brief's
  per-second accuracy at ±0.25 s and ±1.0 s (stricter tolerances, same sampling
  scheme) as the new metric.
- **Answer keys:** `labeling/fixtures/bb12_ground_truth.yaml` (set `1fsnxchk`,
  ~/aligning/1fsnxchk__Two Friends - Big Bootie Mix Volume 12/) and
  `labeling/fixtures/bb11_ground_truth.yaml` (set `2nvzlh2k`, Episode 11 folder).
  Schema: per span `set_start_s/set_end_s/ref_start_s/ref_end_s/tempo_ratio/
  pitch_shift_semi/is_loop/ref_segments[{mix_start_s, ref_start_s, ref_end_s}]/
  gain_curve/claimed_stem`. GT is FINAL for both sets (150 rows); never tune on it —
  thresholds tune on synthetic fixtures only (§9 of the brief).
- **End-to-end scorer (real placement):** `score_timeline_vs_gt.py --fibers`.
- **Reproduction command (oracle placement, the baseline this task quotes):**

  ```bash
  venvs/audio/bin/python -m workspaces.alignment_prototype.path_decode --eval \
      --feature hubert --stems acappella --fibers --workers 8 \
      [--gt labeling/fixtures/bb11_ground_truth.yaml]
  ```

### Audio I/O, sample rates, tempo ratio

- Feature SR=22050, HOP=512 → FPS≈43.066 (path_decode); landmark fp uses NFFT=2048,
  FHOP=512. Mix audio: `~/aligning/<set>/mix_vocals.flac` (Roformer vocal stem of the
  mix). Ref audio: `manifest.json` rows → `local_path` + `stems.vocals`.
  HuBERT/chroma features cached at `workspaces/alignment_prototype/.feat_cache/`
  (666 files present).
- **Tempo ratio:** GT carries `tempo_ratio` per span; at decode time the stretch band
  comes from the **beat grids** (`_stretch_band`: mix `set_measures` bar length around
  the span vs ref `track_measures` bar length via `mert_store.load_bb12_mert`),
  octave-folded to [0.7, 1.45] × fine grid (0.96–1.04). So slope for Phase 3 is
  available without touching GT: beat-grid ratio, refined per span. GT `tempo_ratio`
  is used only for scoring/fixtures.
- `pitch_shift_semi` in GT flags re-pitched spans (repitch vs key-locked matters for
  landmark hashing).

### Baseline reproduction (Phase 0 gate)

Command above, run 2026-07-06 on this machine (log:
`looptrace/baseline_repro_hubert.log`). Oracle placement (GT set_start), HuBERT L9,
lam=0.15, traj-acc <2 s — **matches the brief exactly; GATE MET**:

| class | BB11 (2nvzlh2k) | BB12 (1fsnxchk) | fiber-aware BB11/BB12 |
|---|---|---|---|
| ALL acappella | 35% (n=17) | 44% (n=21) | 35% / 47% |
| linear | 43% (n=5) | 62% (n=8) | 43% / 62% |
| **multiseg** | **12% (n=7)** | **27% (n=7)** | 12% / 31% |
| loop | 0% (n=1) | 81% (n=1) | 0% / 100% |
| oddratio | 75% (n=4) | 32% (n=5) | 75% / 34% |

These are the **frozen baseline** numbers all later phases compare against (same
command, same tolerances). Note one BB12 span is skipped for no-audio (n=21 of 22).
Loop class is n=1 per set — never read anything into it alone.

### Known-failed approaches (do NOT re-walk)

1. Single-window matched filter (`refine_ref_offsets`) — coin-flips repeats.
2. Viterbi over windowed matched-filter emissions (`path_decode`) — noise decides
   among repeats.
3. Fiber-aware *scoring* — measures ambiguity, doesn't resolve it (machinery reused).
4. Backward-jump penalty (`lam_back` sweep) — hurt.
5. HuBERT features for localization — too self-similar within a track.
6. Seeding decode with GT start (soft ref_start prior, 2026-07-06) — flat.

### Plan deltas vs the brief (repo-specific adjustments)

- `looptrace/` lives under `workspaces/alignment_prototype/`, not top-level (repo
  rule: experiments incubate in workspaces/; new top-level dirs need justification).
- Phase 3 landmarks: reuse/extend `landmark_fp.py` instead of a fresh ~200-line
  implementation — it's already the proven localizer (0.2 s median on regular spans);
  the changes are (a) keep raw match points instead of histogramming, (b) vocal-stem
  routing, (c) pitch-tolerant hashing decided on fixtures.
- Phase 1 equivalence classes: extend `ref_fibers` fibers with sample-accurate
  within-fiber cross-correlation rather than building self-alignment from scratch.
- Loop detection (Phase 2): no existing DJ-loop detector in the repo (the Viterbi
  *represents* backward jumps but never detects mix-side digital copies) — this is
  genuinely new code.
