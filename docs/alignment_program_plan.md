# Plan: Alignment program — GT export → stem-quality cascade → MERT → aligner

> **Shared cross-agent plan.** Authored 2026-06-05. Companion to
> [alignment_objective.md](alignment_objective.md) (north star / deliverables A·B·C),
> [stem_discovery_playbook.md](stem_discovery_playbook.md), and
> [embedding_backfill_plan.md](embedding_backfill_plan.md). Phase 0 folds the formal
> objective below into `alignment_objective.md`.

A single sequenced program. Phase 1 is the immediate critical path; later phases are the
upstream ingest/analysis work that feeds the aligner. Each phase is independently
shippable.

## Context

BB12 (`1fsnxchk`) manual labeling is done in Ableton. Two forces converged in this
planning: (a) we need to extract that ground truth (`set_ground_truth`), and (b) the
quality ceiling of the whole pipeline is set by **stem source type**, not by the
separator — so stem acquisition should be a *priority cascade*, and MERT embeddings must
be (re)computed **after** stems are fixed. The aligner (deliverable B) trains on the
result.

DAG: `core · scrape → ingest → analysis → labeling → (GT) → alignment`. This program
touches labeling (P1), ingest (P2–P3), analysis (P4), and stands up alignment (P5).

## The formal objective (→ docs/alignment_objective.md, Phase 0)

$f_\theta(A,T,D)\to L$; render invariant $E_a(L)=\hat A,\ d(\hat A,A)<\epsilon$; $L$ =
minimum info to reproduce the set within perceptual $\epsilon$.
- **Perceptual, not Frobenius** metric; $\epsilon$ pinned by a listening study,
  proxied by `set_playback_score.reconstruction_mfcc_distance`. **No numeric $\epsilon$
  in v1 training.**
- **MDL:** $L^*=\arg\min_L[\text{bits}(L)+\lambda\,d(E_a(L),A)]$ — rate term = the
  importance ordering (placement ≫ stem ≫ bpm/key ≫ FX/transitions) and granularity knob.
- $T$ is a constraint (alignment, not transcription); $E_a$ non-differentiable ⇒ render =
  eval/reward, train supervised on $L^*$ first; $L$ admits refs $\notin D$/$=A$
  (`substitute_source: mix_extract`) and the rate term blocks the "play $A$" cheat.
- **Tiered $L$:** span (`set_ground_truth`) → measure (`measure_alignment`, the atomic
  $L_\text{play}$) → eval (`set_playback_score`).
- Keep stem **discovery** out of $f_\theta$; it's ingest (P2).

## Phase 1 — BB12 GT export (critical path; depends on nothing else)

`labeling/export_als_to_gt.py` (new) → `bb12_ground_truth.yaml` → `write_back`.

- **Structure (verified):** track1 `1-mix` (set-time anchor), track2 `2-mix_instrumental`
  (skip), then 40 GroupTracks = mashup sections. **GT = every placed clip past track 2.**
- **Identity outsourced to `~/aligning/`** (user): clip `SampleRef` path →
  `tracks/NNN[wK]__…` (regular) | `stems/NNN__…/candidates/{vocals,instrumental}/…` or
  `…/vocals.flac|instrumental.flac` (acappella/instrumental). Join `NNN[wK]` to
  `manifest.json` `local_path` prefix → **`recording_id`** (manifest `track_id` ≈
  recording_id — verify), artist/title/version. Disambiguates Virtu(154)/SAVI(152);
  unresolved Rvmor `w`-rows → null + flag.
- **Time (warp markers; no tempo integration):** set spans via `1-mix` clip warp markers
  (present: 12/8/15/14 — interpolation is primary, linear fallback only as guard); ref
  spans via each clip's own markers + `Loop/LoopStart`; **bpm** = `tempo_ratio =
  ref_span/set_span`; **key** = `pitch_shift_semi` (`PitchCoarse`+`PitchFine`).
- **Parts-played + clip hygiene (do now, v1):** multiple clips on one ref track → record
  the played regions as `ref_segments` (`is_loop`). Denoise first: (a) **drop parking-lot
  clips** placed past the mix (`set_start_s > mix_end_s + tol`; mix ends ~90:37 yet some
  refs have clips at 116–122 min); (b) **merge sub-threshold slivers** into the abutting
  dominant clip ("take the larger clip" — e.g. Maps 56:15–56:18). Surface both in the
  review table. **Defer to a later version:** exact in/out boundary refinement, and
  whether a genuine rapid chop/stutter (e.g. the 7-clip Blink-182 chain) is an intentional
  effect to model separately vs. an artifact to merge.
- **Schema/write-back patches:** add `tempo_ratio REAL`, `pitch_shift_semi INTEGER` to
  `set_ground_truth`; in `labeling/ground_truth/schema.py` add `slot_label` (the
  `(set_id,label)` PK key, distinct from display title) + the two fields, update
  `dump`/`load`; `write_back_ground_truth.py` writes `recording_id` + `slot_label` (fixes
  current title-as-label collision on the 4 "Honest" slots).
- **Out:** YAML + review table (slot, group, spans, tempo_ratio, semi, recording_id,
  stem, flags). Verify anchors vs Ableton playhead; `--dry-run`; write; confirm row count
  on pi-storage.

## Phase 2 — Stem-quality priority cascade (ingest)

Reframe from "rank candidate stems" to a **cascade where source type dominates**.
Separation is the floor, official-derived is the ceiling. Discipline: **detect-then-
correct, never blanket** (correctness-vs-accuracy); log every change to
`track_audio_correction`; matching is **version-aware** via `core/identity.py`.

**2a — Solvability pre-pass (metadata, no audio).** New `ingest/` querier over
**Discogs + MusicBrainz** APIs: per recording, does an official instrumental/acappella
exist (MB `instrumental` attribute; Discogs instrumental B-sides)? Classify all ~18k
tracks into tier 1 (official pair) / 2 (official instrumental → derive) / 3 (separation
only). This routes retrieval/GPU spend before spending it.

**2b — Tiered acquisition.** Retrieve official audio **losslessly** (FLAC/WAV — lossy
breaks phase-cancel) by tier: stores/Stems packs (Beatport, Bandcamp, Qobuz) →
**paid DJ pools** (DJcity/BPM Supreme/ZIPDJ — the official, curated route) → community
(acapellas4u/voclr/YouTube, quality-scored, throttled/cached, ToS-aware).

**2c — Derive + verify (reuse `~/aligning/phase-cancel/cancel.py` as-is).**
- Phase-derive: official instrumental + official full mix → acappella via `cancel.py`
  (`adaptive --smooth 0.5 --fft 4096 --cap 4`; sub-sample align + gain-match already
  built). Reverse for instrumental from official acappella.
- **Sum-check scorer (net-new, small):** `acappella + instrumental ≈ original` after
  align/gain-match → residual energy score. Verifies both are genuine + same master, and
  is the same residual that *produces* the derived stem.

**2d — Quality gate (gate signals):**
1. **Identity/version** — chromaprint on the full mix + **cross-correlation on a shared
   band (chroma/low-mid)** to confirm a stem belongs to the right *version* (key/BPM
   prefilter via Essentia to cheaply kill mismatches).
2. **Sum-check residual** (2c) — strongest, when a pair/original exists.
3. **Separator-as-judge / bleed** — run the vocal separator on a candidate instrumental,
   measure residual vocal energy (≈0 = clean official); reverse for acappellas.
4. **Watermark/voice-tag** — Whisper ASR + brand/URL keyword list (e.g.
   `digitalmusicpoolstudio.com`) → flag contamination **and** build a per-source
   reputation prior.
5. **Transcode** — spectral-cutoff cliff (≈16 kHz = lossy lineage) rejects fake-FLAC rips.
6. **Coverage** — candidate vocal-activity timeline vs the canonical RoFormer-ensemble
   (P3) vocals (catches truncation).
- **Selection = cascade argmax:** sum-check-verified official pair > phase-derived from
  verified official instrumental > best community candidate that *beats* the separation
  floor > separation. Log when separation wins ("no good external stem").

## Phase 3 — Replace the separation backend with SOTA (ingest/analysis)

**Objective for this phase = maximize separation quality (SDR); cost/speed are not
constraints** (training-data prep, money-no-object). So **replace Demucs as the
separation backend** with the current SOTA **ensemble**, not a single model.

- **Model:** **Mel-Band RoFormer + BS-RoFormer + SCNet-XL** ensemble, staged to limit
  artifacts (~+0.8–1.0 dB SDR over single models; Mel-Band edges BS-RoFormer on vocals;
  BS-RoFormer won SDX23 at ~12.9 dB). Keep the separator **pluggable behind one
  interface** so we can swap to whatever tops the MVSEP leaderboard later.
- **Tooling (verified, with the Mac caveat):**
  - **Vast (CUDA) — primary, run the corpus here:** `pip install "audio-separator[gpu]"`
    (now under the **nomadkaraoke** org; auto-downloads RoFormer weights) **or** ZFTurbo
    `Music-Source-Separation-Training` (`bs_roformer`/`mel_band_roformer`/`scnet`).
  - **Mac (Apple Silicon) — native path is MLX, NOT stock audio-separator:** stock
    CoreML accel targets ONNX, but the best RoFormer weights are PyTorch `.ckpt` →
    they route through MPS, and `PYTORCH_ENABLE_MPS_FALLBACK=1` silently drops unsupported
    ops to CPU (loses the GPU benefit on exactly this path). Use the **MLX port
    `ssmall256/mlx-audio-separator`** for native Apple-Silicon inference. Mac = local/tail
    only; bulk runs on Vast.
- **Role:** this backend powers both the **tier-3 floor** (P2) and the **separator-as-
  judge / bleed** metric (2d.3). Existing Demucs `track_stems` get **re-separated** with
  the ensemble (better stems → better tier-3 floor → better MERT in P4).
- **Validate first:** SDR vs Demucs on a held-out clip + bleed-judge ≈0 on a known
  official instrumental, before bulk re-separation. Independent of P1/P2a — can start now.

**Repo-verification:** grep over `*.py`/`*.md` found **no**
`roformer`/`audio-separator`/`mel_band`/`bs_roformer`/MSST references; separation is
Demucs-based per the docs. So this is net-new.

## Phase 4 — MERT backfill (analysis; AFTER P2 re-sourcing)

Per `embedding_backfill_plan.md`: 330M target = **0 tracks**, existing ~4.5% are stale
95M (unmatchable against 330M). Must be a **clean full 330M recompute** — you can't match
mix-side vs ref-side across model versions.
- **Sequencing:** P2 changes which audio exists and 6c (variant MERT) is gated on stem
  ingest — so embed **after** re-sourcing, else ~100+ GB is wasted on replaced audio.
- **6a** refs (330M all-layer), **6c** variants (acap/instr from P2). **6b set-side**
  (`set_mert_measures`, mix recordings) is stem-independent + an aligner hard-prereq →
  may run in parallel now. Size disk/GPU-hours first.

## Phase 5 — Aligner prototype (workspaces/ → promote to alignment/)

Supervised span aligner: inputs $A$ features + $T$ slots + $\{E(x)\}$; output span-level
$L$ matching `GroundTruthTrack`; loss decomposes by tier (Huber on spans, CE on
`(recording_id, claimed_stem)`, set-multiplicity for mashup layers). Eval = held-out span
error vs $L^*$; later distill to `measure_alignment` + proxy $E_a$ (pyrubberband + stem
mix) + listening study to pin $\epsilon$.

## Critical files

- **New:** `labeling/export_als_to_gt.py`; `ingest/` solvability querier (Discogs/MB) +
  cascade acquisition driver + sum-check scorer + quality gate; RoFormer wrapper.
- **Edit:** `labeling/ground_truth/schema.py`, `labeling/write_back_ground_truth.py`,
  `web_crawler/database/schema.sql`, `docs/alignment_objective.md`,
  `docs/stem_discovery_playbook.md`, `docs/embedding_backfill_plan.md`.
- **Reuse as-is:** `~/aligning/phase-cancel/cancel.py`, `scripts/fetch_candidate_stems.py`
  (promote from labeling-support to corpus ingest), `scripts/ingest_stem_url.py`,
  `core/identity.py`, `scripts/mert_backfill_loop.py`.

## Verification

- P1: review table = 1 row/layer, 0 unresolved ids; anchors ±~1 beat; `schema.load` Ok;
  `write_back --dry-run` clean; pi-storage row count.
- P2: solvability tier counts sane vs spot-checks; sum-check passes on known official
  pairs; gate rejects a seeded watermarked/truncated/transcoded sample; corrections
  ledgered.
- P3: RoFormer ensemble beats Demucs SDR on a held-out clip; bleed-judge ≈0 on an
  official instrumental.
- P4: 330M coverage climbs; both mix and ref sides embedded in the same space.

## Risks

- Live `.als` edits (re-parse); `pyexpat` broken in `venvs/audio` (use lxml/`/usr/bin/python3`).
- **Version disambiguation** is the real hard part of P2 — wrong master ⇒ phase-cancel
  bleed misattributed to technique. Gate on version-aware matching.
- Phase-cancel needs lossless + sample-accurate sources; budget residual.
- Don't blanket re-source/re-embed — detect-then-correct; size MERT cost first.
- ToS/rate-limits on community scraping (throttle/cache); copyright = personal-use only.
