# analysis/ — per-track + per-set MIR analysis

Consumes downloaded audio (from [ingest/](../ingest/CLAUDE.md)) and produces the
analysis outputs: beat grids, stems, cue points, key/BPM/mood features, and MERT
section embeddings. Writes the *audio-pipeline tables* (`track_analysis`,
`track_audio_features`, `track_mert_sections`, `set_analysis`, `set_stems`,
`set_measures`, `canonical_track_cue_points`, …).

`pipeline.py` / `set_analysis.py` orchestrate; `adapters/` wrap each model;
`vast_worker.py` is the GPU-side batch worker; `persistence.py` writes results.

## Stem-separation backends (roformer | demucs | uvr)

Stem separation is a **selectable backend** behind one contract —
`StemSet(vocals, instrumental)`. All backends produce exactly those two stems,
so nothing downstream (schema, alignment, library) changes when you switch.

> **`roformer` is the current backend of choice.** `demucs` is **stale/legacy**:
> still wired and runnable (and the code default for backward compatibility), but
> new separation runs should pass `--separator roformer`. The old `ref_source:
> demucs` stems remain valid historical records — don't rewrite them.

| Backend | What | Speed | When |
|---|---|---|---|
| `roformer` *(current)* | MSST RoFormer vocal+instrumental ensemble ([adapters/roformer_chain_adapter.py](adapters/roformer_chain_adapter.py); needs `workspaces/msst_webui` + `venvs/msst`, see `scripts/setup_roformer_separation.sh`) | GPU ensemble | **default choice for new runs** |
| `demucs` *(stale/legacy, code default)* | `htdemucs_ft`, 2-stem ([adapters/demucs_adapter.py](adapters/demucs_adapter.py)) | ~1 model pass | legacy / fallback only |
| `uvr` | audio-separator cleanup **chain** ([adapters/uvr_chain_adapter.py](adapters/uvr_chain_adapter.py)): Kim Vocals 2 → karaoke ensemble → dereverb → de-echo → denoise (note: its `instrumental_cascade` re-sums via demucs) | ~5 sequential passes (much slower) | when a *clean dry lead vocal* matters more than throughput |

- `pipeline.run_separation()` dispatches on `Analyzers.separator`; select via
  `load_analyzers(device, separator="roformer")` or the `--separator
  {roformer,demucs,uvr}` flag on every loop (`mac_analyze_loop.py`,
  `mac_analyze_sets.py`, `vast_loop.py`, `vast_worker.py`). The code default is
  still `demucs` (legacy) — pass `roformer` explicitly.
- The chain is **data-driven** by [uvr_chain.yaml](uvr_chain.yaml) (parsed in
  [separation_config.py](separation_config.py)) — reorder/disable/retune stages,
  or set `enable_ensemble: false`, with no code change. Stage stem-selection
  matches the parenthesised label audio-separator writes into output filenames,
  via the Python API's returned file list (never globbing).
- The ensemble is **native** to audio-separator (`load_model([m1, m2])` +
  `ensemble_algorithm: avg_fft` = magnitude-spectrogram average) — no hand-rolled
  spectral math.
- One-off / A-B a single file: `scripts/separate.py --input X --separator both`.
- Setup per host (ffmpeg + correct onnxruntime variant + model pre-download +
  provider check): `scripts/setup_separation.sh`. **CUDA hosts need
  `onnxruntime-gpu`** — the `uvr` backend loud-fails if `device=cuda` was asked
  for but audio-separator fell back to CPU; on Mac/CPU a CPU/CoreML provider is
  expected and only warns.

## Full-set separation: use `render_set_stems.py`, not `mac_analyze_sets.py`

To produce a **continuous acappella + instrumental of a whole mix**
(`set_stems` for one `set_audio_id`), use
[scripts/render_set_stems.py](../scripts/render_set_stems.py) — **not**
`mac_analyze_sets.py`. The latter runs `beat_this` on the full mix *first*, and
beat_this is a transformer with no internal chunking: a 60-min mix OOMs MPS
(~26 GiB) before separation even starts. `render_set_stems.py` skips beat_this
(stems are all the full-set acappella goal needs), slices the mix into
fixed-length chunks (default 360s), runs the selected backend per chunk, and
concatenates — so MPS memory stays bounded regardless of mix length. The chunk
loop is **resumable** (a chunk whose part files exist is skipped) and writes the
two `set_stems` rows + rsyncs `{vocals,instrumental}.flac` to
`/mnt/storage/stems/set/<id>/` on completion. Caveats: hard-cut chunk
boundaries (no crossfade → faint seams at joins); on a Mac, run on **AC power
with the lid open** — `caffeinate -i` does not stop battery maintenance sleep,
which suspends the process for hours. (Validated on BB12 `set_audio_id=5`,
3729s, uvr backend, ~1.5h.)

## Which dependency runs where

| Component | Runs on | Why |
|---|---|---|
| beat_this (beats/downbeats) | pi-storage CPU **or** Mac MPS | PyTorch has aarch64 + MPS wheels; small model |
| cue-detr (EDM cues) | pi-storage CPU **or** Mac MPS | DETR transformer; small model |
| librosa, pyloudnorm | pi-storage **or** Mac | pure Python |
| **Essentia** (key/BPM/valence/mood/etc.) | **Vast.ai** *or* **Mac** | no aarch64 wheels — ships only x86_64 manylinux + macOS arm64, so the Mac has a `venvs/essentia/` Py3.13 sandbox and runs Essentia as a subprocess |
| **Stem separation** (roformer *(current)* / demucs *(stale)*) | **Vast.ai** *or* **Mac MPS** | GPU-bound; demucs ~30s/track Pi CPU vs ~1s/track 4090 vs ~3–5s/track MPS |
| **MERT** embeddings | **Vast.ai** *or* **Mac MPS** | GPU-bound; [adapters/mert_adapter.py](adapters/mert_adapter.py) auto-selects `cuda → mps → cpu` |

The Mac mirrors the pi-storage CPU stack (`venvs/audio/`) plus the
`venvs/essentia/` sandbox, so the **entire production analysis pipeline** is
exercisable locally — not just dev, but an actual production worker for batches
that don't justify spinning up Vast. GPU batch entry points:
[vast_worker.py](vast_worker.py) (driven by [scripts/vast_loop.py](../scripts/vast_loop.py))
and [scripts/mac_analyze_loop.py](../scripts/mac_analyze_loop.py) (~200–250 s/track
on MPS vs ~85 s on a 4090).

## MERT embedding choice

We use `m-a-p/MERT-v1-95M` at **hidden layer 6** (not the final layer) for both
analysis and alignment paths. The MERT paper shows mid-layers transfer best to
music-ID / structural-matching tasks; the top of the stack is more
tagging-oriented and the bottom too acoustic. Constant lives in
[adapters/mert_adapter.py](adapters/mert_adapter.py) as `MERT_DEFAULT_LAYER`.
(The legacy `mert_align.py` carried a duplicate `DEFAULT_LAYER` that had to be
kept in sync; it was removed with the old aligner. The future aligner should
import the constant from the adapter rather than redefine it.) When a learnable
scoring head is added on top (post-ground-truth labeling), replace the
single-layer pick with a learnable weighted sum over all hidden states (SUPERB
pattern) co-trained with the head.

**Backlog: upgrade to `m-a-p/MERT-v1-330M`.** The 330M variant has 24
transformer layers (vs 12 in 95M), and the deeper stack carries
task-specialized representations at well-defined depths:

| layer band | what it encodes | best for |
|---|---|---|
| 4–7   | low-level acoustic features | beat / tempo, onset detection |
| 8–13  | pitch + harmonic content    | key detection, chord recognition |
| 14–19 | timbre + instrumentation    | acapella-vs-instrumental discrimination, source-separation cues |
| 20–24 | high-level semantic         | genre, mood, structural segmentation |

For this pipeline, **don't pick a single layer** — use a learned weighted sum
across all 25 hidden states (the standard SSL probing approach, SUPERB / s3prl
pattern), co-trained with the scoring head. That lets each downstream task pull
from whichever band is most informative, instead of forcing one mid-layer
compromise across beat/key/timbre/structure at once.

Tradeoffs to plan for before flipping the constant:
- ~3.5× parameter count → ~3× inference time on MPS/CUDA. Vast cost is still
  bounded; Pi CPU becomes impractical (re-route 330M jobs to Mac MPS or Vast only).
- Cache key changes (layer-pick → weights identifier). The alignment cache must
  be flushed or namespaced when migrating.
- Frame rate (~75 Hz at 24 kHz) is unchanged; downstream measure-pooling code
  stays the same.

## CUE-DETR (vendored at [../cue-detr/](../cue-detr/))

DETR-based model for cue point detection in EDM tracks. Custom COCO-like format
with `position` instead of bounding boxes. Pretrained model `disco-eth/cue-detr`
on HuggingFace; downloaded by default. Consumed here only by
[canonical_cues.py](canonical_cues.py) (full-song @ sensitivity=0.5, keyed by
`track_id` into `canonical_track_cue_points`). Runs only on reference audio with
**identity `stem='regular'`** — not on acappella/instrumental `track_audio` rows
(see root CLAUDE.md "Track identity"). [pipeline.py](pipeline.py) gates Essentia
the same way.

```bash
pip install -r ../cue-detr/requirements.txt
python ../cue-detr/cue_points.py -t /path/to/audio/dir   # Predict cue points
# Flags: -c <checkpoint_dir>, -s <sensitivity>, -r <min_distance>, -p (print)
```

## persistence.py vs core/db.py boundary

[core/db.py](../core/db.py) is the **generic** DB adapter — it converts sqlite
exceptions into `DbError` Results and knows only `core` types. It must stay free
of analysis-domain types. The analysis-result writers (`TrackAnalysisResult` /
`EssentiaFeatures` / `SetAnalysisResult`) depend on [models.py](models.py), so
they live here in [persistence.py](persistence.py) and import `core.db`'s
`connect` primitive for the connection — not the other way around. Keep new
analysis-domain DB writes in `persistence.py`, not `core/db.py`.

## Deploy caveat

A pi-storage systemd unit running `python -m audio_pipeline.vast_worker` (or
similar) must be updated to `python -m analysis.vast_worker` after the
`audio_pipeline/` split, or it won't restart.
