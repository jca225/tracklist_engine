# source_detection — object-detection on a mix timeline

Finds **when and where original songs (acapellas / instrumentals) appear inside
DJ mashup mixes** (Two Friends "Big Bootie"), robust to the DJ's pitch-shift and
time-stretch. Output per detection:

    (mix, source_song, channel, mix_start_s, mix_end_s,
     ref_start_s, ref_end_s, pitch_shift_semi, time_stretch, confidence)

Incubates in `workspaces/` per the repo convention (promote to a top-level
`alignment/` sibling when stable). It is the *detection* framing of the same
problem the sibling `workspaces/alignment_prototype/` attacks as span placement —
and it deliberately reuses that prototype's hard-won lessons.

> ## ⚠️ CANONICAL labeling session
>
> The ground truth comes from **`/Users/johnnycabrahams/Desktop/big bootie 12
> labeling Project/big bootie 12 labeling_fast.als`** — the hand-edited spine.
> "**Track N**" always means the **Ableton lane number in this session**, never a
> tracklist position or a slot label. The sibling `_slow.als` is only a cross-check.
> Regenerate the verified GT with `als_reconcile … --canonical fast`.
>
> Two annotations baked into the verified GT:
> - `audit.ignore: true` — **mix self-references** (the mix's own `mix`/`mix_instrumental`
>   audio used as a clip). Exclude from detection/eval.
> - `audit.source_note` — **outsourced hosts**. Track 61 (`61-mix_instrumental`, group
>   "Lux x Spaceman") is **"Lux Omega"**: not available online, so the mix's own
>   instrumental stem was substituted.

## Why this design (read before "improving" it)

The sibling prototype established three facts on this exact corpus (BB12), all of
which shape this pipeline:

1. **Pooled MERT cannot localize** — it's a retrieval signal ("is this song
   present?"), ~900 s off when used to place. → MERT is used here only as a
   candidate **prefilter**, never for timing.
2. **Full-mix chroma/fingerprint can't localize self-similar EDM** (the 2026-06
   "P0 fail", ~26 s floor). The fix that worked: **stem separation** — a mix
   moment is a *sum* of layers (host instrumental + overlaid acapellas), so we
   match per stem channel (`mix_vocals ↔ source vocals`, `mix_instrumental ↔
   source instrumental`), which un-entangles them and raises SNR.
3. **Route by audio, not by label.** Most BB12 "acappella" rows are actually full
   tracks. We therefore try *both* stem channels per source and keep the better
   match — we never trust `claimed_stem`.

The localization primitive is the **normalized FFT chroma matched filter** from
`alignment_prototype/refine_ref_offsets.py` (proven: BB11 151/151 relocated, peak
median 0.83 s), here run *reversed* — sliding each source template over the mix —
across 12 pitch rotations and a BPM-derived stretch set. Tempo is handled by the
domain rule **the host instrumental's BPM is fixed within a span and acapellas are
beat-synced to it**, so `time_stretch ≈ mix_local_bpm / source_bpm` and we search
a tight band around that rather than a blind seconds-space grid (which saturated
at its edges in v1).

## Pipeline (resumable; every expensive artifact cached by file hash)

    stems → prefilter(MERT) → match(matched-filter) → postprocess(merge/NMS) → report

- **stems** — resolve `mix_vocals/mix_instrumental` + per-source `vocals/instrumental`.
  For the BB12 pilot they already exist in `~/aligning/<set>/`; otherwise Demucs.
- **prefilter** — MERT per-measure cosine shortlists candidate sources (optional;
  skipped automatically when source count is small, e.g. the 10-song smoke test).
- **match** — per source, per channel: 12 chroma rotations × BPM-stretch band,
  sliding template over the mix; peak-pick → raw detections with pitch/stretch/score.
- **postprocess** — merge adjacent/looped hits of the same song, NMS across
  conflicting overlaps, threshold by confidence.
- **report** — per-mix CSV + JSON, an HTML timeline (one row per source, colored
  spans, hover = pitch/stretch/confidence), and side-by-side A/B audio clips
  (mix excerpt vs aligned source excerpt) for spot-checking by ear.

## Usage

    venvs/audio/bin/python -m workspaces.source_detection.detect \
        --mix  "$HOME/aligning/1fsnxchk__Two Friends - Big Bootie Mix Volume 12" \
        --sources "$HOME/aligning/1fsnxchk__Two Friends - Big Bootie Mix Volume 12/stems" \
        --stage all                       # or: stems | prefilter | match | postprocess | report

    # the curated 10-source end-to-end smoke test on BB12 (recommended first run):
    venvs/audio/bin/python -m workspaces.source_detection.detect --bb12-smoke

    # evaluate detections against hand-labeled GT (±5 s tolerance):
    venvs/audio/bin/python -m workspaces.source_detection.eval --set-id 1fsnxchk

A `--mix` may be either a folder already containing `mix_vocals.flac` /
`mix_instrumental.flac`, or a single audio file (then stems are computed). A
`--sources` dir holds one subdir per source, each with `vocals.flac` /
`instrumental.flac` (and/or a full mixdown).

## Status

Scaffold. The plumbing (IO, caching, CLI, prefilter, postprocess, report, eval)
is complete; the matcher is a working v1 (matched-filter, rotation + BPM-stretch).
First milestone: run `--bb12-smoke` (1 mix × 10 sources), read the precision
diagnostics, then `eval` to tune thresholds before scaling to all 293 sources.
