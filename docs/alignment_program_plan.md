# Alignment program plan — GT export → stem cascade → MERT → aligner

> **Canonical shared plan** (Claude + Cursor). Authored 2026-06-05; merged 2026-06-05.
> Edit this file only — PDF via `docs/alignment_program_plan.tex`.
> Companions: [alignment_objective.md](alignment_objective.md),
> [stem_discovery_playbook.md](stem_discovery_playbook.md),
> [embedding_backfill_plan.md](embedding_backfill_plan.md),
> [roformer_separation_plan.md](roformer_separation_plan.md) (P3 detail).

A single sequenced program. Phase 1 is the immediate critical path; later phases are
upstream ingest/analysis work that feeds the aligner. Each phase is independently
shippable.

**Set under labeling:** `1fsnxchk` (Big Bootie Vol. 12)  
**Ableton session:** `~/Desktop/big bootie 12 labeling Project/big bootie 12 labeling_fast.als`  
(always re-parse on disk — never cache a parse)

---

## Context

BB12 manual labeling is done in Ableton. Two forces drive this program:

1. Extract human ground truth $L^*$ into `set_ground_truth` (deliverable A — GT reader).
2. The quality ceiling of the whole pipeline is set by **stem source type**, not the
   separator — stem acquisition is a *priority cascade*, and MERT must be recomputed with
   correct sequencing (see [Phase 4](#phase-4--mert-backfill)).

**DAG:** `core · scrape → ingest → analysis → labeling → (GT) → alignment`

| Phase | Module |
|-------|--------|
| P0 | docs |
| P1, P1.5 | labeling + write-back |
| P2–P3 | ingest / analysis |
| P4 | analysis (MERT) |
| P5 | alignment (workspaces/) |

---

## Phase overview

| Phase | Scope |
|-------|-------|
| **P0** | Fold formal objective into [alignment_objective.md](alignment_objective.md) |
| **P1** | BB12 GT export + schema/write-back (tiers 1–4) |
| **P1.5** | Ingest BB12 placed stem winners into `track_audio` |
| **P2** | Stem-quality priority cascade (ingest) |
| **P3** | RoFormer separation backend (ingest/analysis) |
| **P4** | MERT backfill (split sequencing) |
| **P5** | Span aligner prototype |

---

## Formal objective (P0 → alignment_objective.md)

$f_\theta(A,T,D)\to L$; render invariant $E_a(L)=\hat A$, $d(\hat A,A)<\epsilon$; $L$ =
minimum info to reproduce the set within perceptual $\epsilon$.

| Topic | Decision |
|-------|----------|
| **Metric** | Structured perceptual $d$, **not** Frobenius $\|\hat A-A\|_F^2$. Pin $\epsilon$ via listening study; proxy later with `set_playback_score.reconstruction_mfcc_distance`. **No numeric $\epsilon$ / waveform loss in v1.** |
| **MDL** | $L^*=\arg\min_L[\text{bits}(L)+\lambda\,d(E_a(L),A)]$ — rate term = importance ordering (placement ≫ stem ≫ bpm/key ≫ FX/transitions). |
| **$T$** | Constraint — alignment over finite $D$, not open transcription. |
| **$E_a$** | Non-differentiable → eval/reward only in v1; train supervised on $L^*$ first. |
| **Degenerate cheat** | $E_a(L)$ can paste slices of $A$; rate term + placement-identity supervision block "play $A$" collapse. |
| **Out-of-corpus** | $L$ admits refs $\notin D$ via `ref_source = mix_extract` (and related). |
| **Stem discovery** | **Out of $f_\theta$** — ingest (P2). Aligner selects among existing `track_audio` rows. |

### Tiered $L$

| Tier | Content | Schema home | This program |
|------|---------|-------------|--------------|
| 1 | Placement: `recording_id`, spans, `ref_segments`, `slot_label` | `set_ground_truth` | P1 |
| 2 | Stem select + `ref_source` | `set_ground_truth` | P1 |
| 3 | `tempo_ratio` (span-level bpm) | GT DSL + DB | P1 |
| 4 | `pitch_shift_semi` (span-level key) | GT DSL + DB | P1 |
| 5 | FX, volume, transitions | deferred | later |
| play | Atomic measure grid $L_{\text{play}}$ | `measure_alignment` | P5+ |

---

## Phase 1 — BB12 GT export

**Critical path;** depends on nothing else.

```
.als → labeling/export_als_to_gt.py → bb12_ground_truth.yaml
     → labeling/write_back_ground_truth.py → set_ground_truth
```

### Session structure (verified)

- Track 1: `1-mix` (set-time anchor)
- Track 2: `2-mix_instrumental` (skip)
- 40 top-level GroupTracks = mashup sections (e.g. Honest x Mean)
- GT = every **kept** placed clip past track 2 (after clip hygiene)
- Mashup simultaneity: **one YAML row per layer** (not one row per group)

### Identity (outsourced to `~/aligning/`)

Parse clip **`SourceContext/OriginalFileRef/Path`** (prefer over `Samples/Imported` copy):

| Path shape | Slot | `claimed_stem` | `ref_source` (export) |
|------------|------|----------------|------------------------|
| `tracks/NNN[wK]__…` | `NNN[wK]` | `regular` | `reference` |
| `stems/NNN__…/candidates/vocals/…` | `NNN` | `acappella` | `online_candidate` |
| `stems/NNN__…/vocals.flac` (Demucs) | `NNN` | `acappella` | `demucs` |
| `stems/NNN__…/candidates/instrumental/…` | `NNN` | `instrumental` | `online_candidate` |
| `stems/NNN__…/instrumental.flac` | `NNN` | `instrumental` | `demucs` |
| Phase-cancel derived (future) | per role | per role | `phase_cancel` |
| Official store/pool acquire | per role | per role | `official` |
| RoFormer separation output | per role | per role | `roformer` |
| Mix extract / phantom track | per human | per human | `mix_extract` |

Join `NNN[wK]` to `manifest.json` `local_path` prefix → **`recording_id`**
(manifest `track_id` ≈ `recording_id` — verify against `recording` / `set_track_slots`).
Strip user tags `[NNNbpm KK]` per `_USER_TAG_PATTERN` in `pull_set_for_alignment.py`.
Disambiguates Virtu (154) / SAVI (152). **Legacy BB12 manifest** lacks `label`/`version`/`stem`
— parse slot from `local_path` only. Unresolved Rvmor `w`-rows → `recording_id=null` + flag.

### Time mapping (warp markers; no tempo envelope integration)

- **Set spans:** `content_beat = LoopStart + (arr − CurrentStart)` → SecTime over `1-mix`
  warp markers. Interpolation primary (markers present on current file); linear fallback as guard.
- **Ref spans:** clip warp markers + `Loop/LoopStart`
- **Bpm:** `tempo_ratio = ref_span_s / set_span_s` (flag intra-clip varying warp for measure-level later)
- **Key:** `pitch_shift_semi` from `PitchCoarse` (+ `PitchFine`)

### Parts played + clip hygiene (v1 — do now)

Played regions are captured by `ref_start_s` / `ref_end_s` (and `ref_segments` for loops).

**Before emitting YAML rows:**

1. **Parking-lot drop:** `set_start_s > mix_end_s + tol` (mix ends ~90:37; some refs at 116–122 min)
2. **Abutting sliver merge:** sub-threshold clip glued to dominant neighbor (e.g. Maps 56:15–56:18)
3. **Intentional loops:** comparable-size or overlapping clips on same track → `is_loop: true` +
   one `ref_segments` entry per iteration
4. **Defer:** Blink-182 7-clip chop/stutter (effect vs artifact); sub-clip jumps inside one
   arrangement region (measure-level $L_{\text{play}}$)

Review table lists **kept, dropped, and merged** clips with reasons.
Optional `--include-all-clips` for debugging.

### Schema / write-back patches

**`set_ground_truth` adds:**

- `tempo_ratio REAL`
- `pitch_shift_semi INTEGER`
- `ref_source TEXT` — `reference|official|phase_cancel|online_candidate|demucs|roformer|mix_extract`

**`labeling/ground_truth/schema.py` adds:** `slot_label` (PK key `154`, `154w1`, distinct from
display `track` title), plus fields above. Update `load` / `dump` / `save`.

**`write_back_ground_truth.py`:** use `slot_label` as DB `label` (fixes 4× "Honest" title collision);
write `recording_id`; persist new columns.

Migration (pi-storage, once):

```sql
ALTER TABLE set_ground_truth ADD COLUMN tempo_ratio REAL;
ALTER TABLE set_ground_truth ADD COLUMN pitch_shift_semi INTEGER;
ALTER TABLE set_ground_truth ADD COLUMN ref_source TEXT;
```

### P1 verification

- Review table: one row per **kept** layer after hygiene; 0 unresolved `recording_id` flags you intend to fix
- Anchor-check 3–4 spans vs Ableton playhead (±~1 beat)
- `schema.load(YAML)` ⇒ Ok
- `write_back_ground_truth --dry-run` then live; confirm row count on pi-storage
- Fixture `.als` tests: beat→sec, path→recording_id, tempo_ratio/semi

---

## Phase 1.5 — BB12 winner ingest

After P1 export: batch-ingest placed `candidates/` and volcr.it winners (e.g. Calvin Harris Slide)
via `scripts/ingest_stem_url.py` so corpus $D$ matches $L^*$. Log source in
`track_audio_correction.reason` (`source:volcr.it`, `quality:…`).

`fetch_candidate_stems.py` stays **labeling-support** (audition only) — do not promote it to
canonical DB writes; P2 cascade driver is separate.

---

## Phase 2 — Stem-quality priority cascade

Reframe from "rank candidates" to **cascade where source type dominates**. Separation is the
floor; official/phase-derived is the ceiling.

**Discipline:** detect-then-correct, never blanket; log every change to `track_audio_correction`;
matching is version-aware via `core/identity.py`.

### Cascade (priority dominates within-tier ranking)

| Tier | Action | `ref_source` |
|------|--------|--------------|
| **0** | MusicBrainz + Discogs: official inst/acap exists? Route spend. | — |
| **1** | Official pair (Beatport, Bandcamp, Qobuz, DJ pools, stem packs). Sum-check: `acap + inst ≈ full`. | `official` |
| **2** | Phase-cancel: official inst + full mix → acap (lossless FLAC/WAV only) | `phase_cancel` |
| **3** | Community (voclr, acapellas4u, YouTube) — must **beat** tier-4 floor | `online_candidate` |
| **4** | RoFormer ensemble separation (P3) | `roformer` / `demucs` |
| **5** | Mix extract at GT boundaries | `mix_extract` |

**Selection:** cascade priority first; within tier, argmax quality score.

### 2a — Solvability (metadata, no audio)

New `ingest/` querier over Discogs + MusicBrainz: classify ~18k tracks into tier 1 / 2 / 3
before retrieval or GPU spend.

### 2b — Tiered acquisition

Lossless only (MP3 breaks phase-cancel): stores/stem packs → paid DJ pools (DJcity, BPM Supreme,
ZIPDJ) → community (throttled, cached, ToS-aware).

### 2c — Derive + verify

Reuse `~/aligning/phase-cancel/cancel.py` as-is (`adaptive --smooth 0.5 --fft 4096 --cap 4`).
**Sum-check scorer (net-new):** `acappella + instrumental ≈ original` after align/gain-match →
residual energy score.

### 2d — Quality gate

1. **Identity/version** — chromaprint + cross-correlation (chroma/low-mid); Essentia BPM/key prefilter
2. **Sum-check residual** (2c) — strongest when pair/original exists
3. **Separator-as-judge / bleed** — RoFormer vocal on candidate instrumental (≈0 = clean official)
4. **Watermark/voice-tag** — Whisper ASR + lexicon (e.g. `digitalmusicpoolstudio.com`) + per-source reputation
5. **Transcode** — spectral cliff ≈16 kHz rejects fake-FLAC
6. **Coverage** — vocal-activity timeline vs RoFormer vocals (truncation)

Log when separation wins ("no good external stem"). Human BB12 picks (Slide, All My Friends)
calibrate the ranker — advisory until calibrated; never auto-reject human winners.

### P2 module map

| Component | Path |
|-----------|------|
| Solvability | `ingest/metadata_stems.py` |
| Phase-cancel + sum-check | `ingest/phase_cancel.py` (promote from labeling scratch) |
| Orchestration | `ingest/stem_resolver.py` |
| Scoring | `ingest/stem_quality.py` |
| Canonical write | `acquire_variant.py`, `ingest_stem_url.py` |

---

## Phase 3 — RoFormer separation backend

**Full rollout plan:** [roformer_separation_plan.md](roformer_separation_plan.md)

Maximize SDR; cost/speed not constrained. Replace Demucs floor with **Mel-Band RoFormer +
BS-RoFormer + SCNet-XL** ensemble (pluggable interface; swap when MVSEP leaderboard moves).

- **Vast (CUDA) — primary:** `pip install "audio-separator[gpu]"` (nomadkaraoke) or ZFTurbo MSST
- **Mac (Apple Silicon):** `ssmall256/mlx-audio-separator` — **not** stock audio-separator on MPS
  (`PYTORCH_ENABLE_MPS_FALLBACK=1` silently drops ops to CPU)
- Powers tier-4 floor (P2) and bleed-as-judge (2d.3)
- Re-separate existing Demucs `track_stems` after validation
- **Validate first:** SDR vs Demucs on held-out clip + bleed-judge ≈0 on known official instrumental
- Can start independently of P1/P2a

Repo state: separation is Demucs/UVR-MDX today; RoFormer is net-new (adapter not landed).

---

## Phase 4 — MERT backfill

Per [embedding_backfill_plan.md](embedding_backfill_plan.md): 330M target = **0 tracks**;
~4.5% stale 95M (unmatchable). Clean full 330M recompute required.

**Split sequencing** (do not wait for full P2 before all MERT):

| Task | When | Rationale |
|------|------|-----------|
| **6b** set-side MERT | Parallel with P1 | Stem-independent; aligner prereq |
| **6a** regular refs | Parallel with P1 (BB12 + whitelist) | Exclude wrong-audio via 6d re-source list |
| **6c** variant (acap/inst) | After P2 for those rows | Avoid embedding replaced audio |
| Re-embed changed rows | Incremental post-P2 | detect-then-correct |

Size disk/GPU-hours first (~100+ GB full corpus). Driver: `scripts/mert_backfill_loop.py` on Vast.
P5 aligner needs 6a on BB12 refs + 6b on BB12 mix at minimum.

---

## Phase 5 — Aligner prototype

`workspaces/` → promote to `alignment/`.

Supervised span aligner: inputs $A$ features + $T$ slots + $\{E(x)\}$; output span-level $L$
matching `GroundTruthTrack`.

Loss: Huber on spans; CE on `(recording_id, claimed_stem)`; set-multiplicity for mashup layers.

Later: distill to `measure_alignment`; proxy $E_a$ (pyrubberband + stem mix); listening study pins $\epsilon$.

---

## Critical files

| Action | Path |
|--------|------|
| **New** | `labeling/export_als_to_gt.py`; `ingest/metadata_stems.py`, `stem_resolver.py`, `phase_cancel.py`, `stem_quality.py`; RoFormer wrapper |
| **Edit** | `labeling/ground_truth/schema.py`, `labeling/write_back_ground_truth.py`, `web_crawler/database/schema.sql`, `alignment_objective.md`, playbooks |
| **Reuse** | `~/aligning/phase-cancel/cancel.py`, `scripts/fetch_candidate_stems.py`, `scripts/ingest_stem_url.py`, `core/identity.py`, `scripts/mert_backfill_loop.py` |
| **Tests** | Fixture `.als` — beat→sec, path→recording_id, tempo_ratio/semi |

---

## Verification checklist

- [ ] **P1:** review table complete; anchors ±~1 beat; `schema.load` Ok; write-back dry-run + live; pi-storage count
- [ ] **P1.5:** placed winners ingested; `track_audio_correction` logged
- [ ] **P2:** solvability tiers sane; sum-check on known official pair; gate rejects seeded bad sample
- [ ] **P3:** RoFormer beats Demucs SDR; bleed-judge ≈0 on official instrumental
- [ ] **P4:** 330M coverage climbs; mix + ref in same embedding space
- [ ] **P5:** held-out span error vs $L^*$ on BB12

---

## Risks

| Risk | Mitigation |
|------|------------|
| Live `.als` edits | Re-parse on disk every run |
| `pyexpat` broken in Py3.14 venv | Prefer `lxml`; else `/usr/bin/python3` |
| Version disambiguation (P2) | Wrong master ⇒ phase-cancel bleed; gate on version-aware matching |
| Phase-cancel | Lossless + sample-accurate alignment; budget residual |
| Blanket re-source/re-embed | detect-then-correct only; size MERT cost first |
| Watermarked/truncated acap candidates | Layer C QA; never auto-reject human picks |
| Community scraping ToS | Throttle/cache; prefer paid DJ pools; personal-use only |

---

## PDF export

```bash
cd docs && pdflatex alignment_program_plan.tex && pdflatex alignment_program_plan.tex
```

LaTeX source tracks this markdown. When editing the plan, **update this file first**, then sync
`.tex` if section structure changes materially.
