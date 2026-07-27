# Placement Ridge Diagnostic — Design

**Date:** 2026-07-18  
**Status:** approved (conversation lock); ready for implementation plan  
**Home:** `eda/alignment/ridge_diagnostic/` (EDA only — not a chain module, not a new aligner probe)

## Problem

Placement is the documented wall: identity is often right, but the model puts the
track at the wrong time (or wrong piecewise structure). Continuous warp is *not*
the bottleneck (warp prior ≈ N(1, 0.012); wider stretch grids regress). The hard
observation model is superposition under piecewise translation:

\[
y(t)=\sum_i a_i(t)\,x_i(\phi_i(t))+b(t)
\]

Most of a "retrieval + Hough vote + sparse local alignment" stack already exists
in `alignment/` (chunking, fp offset-histogram voting,
path_decode / trajectory Viterbi). The open question is whether the remaining
failures are:

1. **Decoder/voting wall** — a usable diagonal ridge already exists in some
   representation we own, but the reader missed it; or
2. **Representation wall** — superposition destroyed the ridge in every existing
   channel, so a mashup-invariant encoder would be *earned*, not assumed.

## Goal

A one-day diagnostic study that answers that binary with pictures, not faith.
No training. No new probe wired into `infer` / `harness` (sensor phase is
closed — see `alignment/CLAUDE.md`). Output is a verdict
grid + a short FINDINGS note.

## Non-goals

- Building a contrastive mashup-invariant encoder
- Milvus / new retrieval infra
- Raw-spectrogram "object detection" as the primary view
- Claiming statistical findings from n&lt;50 (looking exercise only)
- Hand-typing alignment headline numbers into docs (cite
  `docs/alignment_status.md` only)

## Design

### Hard-case set (model-driven)

Select ~8–12 spans from BB11 (`2nvzlh2k`) + BB12 (`1fsnxchk`) where:

- identity was correct (`identity_hit == 1` / `id_correct is True`)
- placement error is large (`set_start_err_s` / `place_err_s` among the worst)

Primary source: agentic driver timelines (agentic owns placement) scored via
`score_timeline_vs_gt.score_spans`. Fallback if agentic timelines are stale/
missing: `eda/alignment/failure_analysis/out/span_table.csv` (built from the
looptrace predicted timeline). Tag each case with taxonomy labels
(section-jump / loop / buried-acap / key-shift) for reading, not for sampling.

### Artifact: similarity matrix \(M(t,s)\)

For each hard case, compute a mix-chunk × ref-chunk similarity matrix under a
fixed panel of four representations already in-repo:

| Channel | Source | Audio pair |
|---|---|---|
| HuBERT-L9 | `path_decode._ensure_feat` / `trajectory.features.FeatureBank` | mix_vocals ↔ ref vocals (acappella) or mix ↔ ref |
| chroma | `FeatureBank.match(..., "chroma")` / `refine_ref_offsets.chroma` | same pair as above |
| fingerprint-hit map | `landmark_fp` hashes → binned co-occurrence / vote density | same pair |
| instrumental stem-fp | `instr_stem_placement` / stem audio via `recon_probe` | mix_instrumental ↔ ref instrumental |

Pre-stretch the reference by GT `tempo_ratio` (≈1%) before comparing so the
pictures are tempo-locked. Overlay the GT true diagonal(s):

- straight span: \(d \approx \texttt{gt\_placement\_onset} - \texttt{ref\_start\_s}\)
- multiseg/loop: one diagonal segment per `ref_segments` entry

### Delta: ridge contrast

One scalar per (case, representation): energy along a narrow band around the
GT diagonal vs background. Not "is there a ridge somewhere" (MERT's 0.92
self-similar floor teaches that lesson) — "does the *true* diagonal stand out."

### Decision rule (the payoff)

Per case:

- **Ridge present** in ≥1 of the four channels, model still missed placement →
  **decoder/voting wall** (cheap fix lane; no new encoder).
- **Ridge absent** in all four → **representation wall** (encoder is earned;
  this exact failure set becomes its labeled eval).

Aggregate: count of (decoder vs representation) cases + qualitative notes.
No p-values, no feature engineering from n&lt;50.

### Deliverables

1. `eda/alignment/ridge_diagnostic/` package: select → compute → plot → table
2. `out/cases.json` — the hard-case list with provenance
3. `out/heatmaps/<case_id>__<channel>.png` — \(M(t,s)\) with GT diagonal overlay
4. `out/contrast_table.tsv` — ridge contrast per (case, channel)
5. `FINDINGS.md` — per-case verdict + overall recommendation (decoder vs encoder)

## Constraints

- Interpreter: `venvs/audio/bin/python`; run from repo root
- GT: `labeling/fixtures/bb11_ground_truth.yaml` / `bb12_ground_truth.yaml`
- Audio: `~/aligning/<set_id>__*/` (pull if missing)
- Do not mutate pi-storage / canonical DB
- Do not add channels to `infer` / `harness` / `drivers`
- Cite status numbers only from `docs/alignment_status.md`
