# Fine placement plan — from 39 s coarse to sub-bar

> Plan of action. Not yet executed. Extends
> [alignment_program_plan.md](alignment_program_plan.md) Phase 5.

## Problem statement

The P5 aligner (`workspaces/alignment_prototype/`) on BB12 held-out spans:

| Column | State | MAE |
|---|---|---|
| identity (`recording_id`+`stem`) | **solved** | 100% acc |
| ref offset (`ref_start_s`) | **solved** | 0.84 s |
| set placement (`set_start_s`) | **coarse only** | **39 s** |

The 39 s comes entirely from the **monotonic tiling prior** (`sequence_decode.py`),
not audio matching. This is measured, not assumed: given the **oracle** ref
segment, pooled-MERT cosine argmax is ~900 s off at *every* layer (0–24), raw or
learned, centered or whitened. **MERT is an identity fingerprint, not a
localizer.** Fine placement therefore needs a different *emission signal* — a
DSP alignment stage, not a bigger model.

## Core idea

We already know, per span, with high confidence: **which** recording
(identity 100%), **which section of it** (ref offset 0.84 s), and **roughly
where** in the set (±39 s window). So the unknown collapses to one cheap
question:

> Which **downbeat of the set** does this known source section land on?

That is a *discrete* search over ~30 downbeats inside the coarse window, snapped
to the beat grid — not a continuous 70-minute search. Sub-bar precision is by
construction once we snap to the right downbeat.

## Assets we already have (don't recompute)

| Need | Source |
|---|---|
| Mix beat grid | `data/analysis/1fsnxchk_measure_times.json` (beat_this); corpus: `set_measures` |
| Source beat grid | `track_analysis.measure_times_json` / `track_measures` |
| Mix Demucs stems | `set_stems` (BB12 has 2); `/mnt/storage/stems/` |
| Source Demucs stems | `track_stems` |
| Identity + ref section | aligner output (`predict_sequence`) |
| Coarse window | aligner `set_start_s` ± ~60 s |
| Eval GT | `labeling/fixtures/bb12_ground_truth.yaml` |

Chroma is cheap to compute on the fly with librosa from rsync'd audio (reuse the
pull pattern in `scripts/mert_backfill_loop.py`) — no pi-side precompute needed
for the spike.

## Method — beat-synchronous chroma, stem-aware, downbeat cross-correlation

1. **Restrict** to the coarse window (aligner `set_start_s` ± 60 s → ~30 set
   downbeats).
2. **Pick the channel by slot stem** (survives mashups — the canonical BB
   pattern is acappella-over-instrumental):
   - acappella slot → mix **vocal** stem vs source acappella
   - else → mix **instrumental/other** stem vs source instrumental
3. **Beat-synchronous chroma**: one CENS/chroma vector per beat for both the
   source section and the set window. Beat-indexing **normalizes tempo away**
   (1 source beat ↔ 1 set beat regardless of the DJ's pitch-ride), so we sidestep
   time-stretching entirely.
4. **Pitch shift**: try 12 chroma rotations (or seed from `pitch_shift_semi` if
   the GT/estimate is available); keep the best.
5. **Slide over downbeat offsets** → normalized cross-correlation. The peak
   offset is the placement; snap `set_start_s`/`set_end_s` to those downbeats.
6. **Confidence** = peak height × peak-to-second-peak ratio. Below threshold →
   **abstain** to the human-review ledger (`track_audio_correction`-style),
   buying effective precision over forced guesses.
7. **Escalate to subsequence DTW** only for spans where cross-correlation is
   weak (local tempo drift, dropped beats) — start simpler.

This is classic audio matching (Müller subsequence DTW / audio thumbnailing),
not novel ML. It exploits every asset above and outputs beat-snapped times that
round-trip to Ableton.

## Phases

**P0 — De-risk spike (½ day, gate).**
One clean held-out span (known identity, in-window, full track not mashup).
Manual notebook: beat-sync chroma of source section vs set window →
cross-correlation → does the peak recover the GT downbeat within ±1 bar (~2 s)?
- **Pass** → build P1.
- **Fail** → the emission signal is wrong; pivot to stretch-tolerant
  fingerprinting (revive `set_fingerprint_hits`, currently empty corpus-wide)
  before investing in a module.

**P1 — `refine_placement.py` module.**
`refine(window, source_section, stems, beat_grids, stem_axis) ->
(set_start_s, set_end_s, confidence)`. Beat-sync chroma, pitch-rotation search,
stem channel select, cross-correlation + DTW fallback. Pure function, audio I/O
at the edge (repo style).

**P2 — Wire as second stage of `predict_sequence`.**
Coarse monotonic decode proposes the window; refinement pins the downbeat.
Keep coarse placement as the fallback when refinement abstains — never regress
below today's 39 s.

**P3 — Eval + abstain curve.**
Measure on BB12 held-out. **Target: median ≤ 1 bar (~2 s)** on confident spans;
report abstain rate and per-type breakdown (acappella vs full, mashup vs solo).
Persist to the findings doc.

**P4 — Cross-set (stretch).**
Run on BB11 once it has any GT (or ear-check blind). The real generalization
test; ties into the BB11 blind-inference item already on the identity side.

## P0 result (2026-06-10) — chroma-class gate FAILED; pivot to fingerprinting

Ran the spike on 4 clean held-out BB12 spans (full tracks, GT-centered ±60–90 s
window = best case). Four emission signals, all snapping to the coarse window:

| Span (ref len) | MERT | chroma DTW | beat-sync bar-chroma | log-mel DTW |
|---|---|---|---|---|
| 086 Pizza (47 s, melodic lead) | ~900 s | **4.0 s** | **3.9 s** | 19.8 s |
| 089 My Window (12 s) | — | 15 s | 14.5 s | **6.1 s** |
| 075 Pinball (12 s) | — | 17 s | 13.2 s | 14.9 s |
| 068 Work From Home (27 s) | — | 67 s | 25.7 s | 60 s |
| **<8 s hits** | 0/4 | 1/4 | 1/4 | 1/4 |

**Verdict: qualified fail.** Chroma/mel reliably localize only a *long,
harmonically-distinctive* section (Pizza). For short, repetitive EDM drops the
match scores are uniformly high (0.9+ chroma) **everywhere** in the window — the
mix is harmonically self-similar, so 12-bin chroma has nothing to discriminate
on, and log-mel is thrown by the DJ's EQ. Stems gave only marginal lift
(Pinball 32→17 s). This beats MERT (~900 s) but is **not** sub-bar and **not**
reliable, and the window here was GT-centered (easiest case) — real coarse
output (39 s MAE, off-center) would be worse.

Two takeaways that redirect P1:

1. **Pivot to stretch-tolerant fingerprinting** (the plan's own fail-branch):
   spectral-peak landmark hashing captures the exact rhythmic/timbral detail
   chroma discards and EQ survives — the thing that breaks self-similarity.
   Home is the empty `set_fingerprint_hits` table. Chroma stays as a
   complementary channel for the melodic cases it already nails.
2. **Fine placement must be JOINT, not per-span.** The spike scored each span in
   isolation (hardest setting). A ±15 s local ambiguity should collapse when the
   monotonic decode forbids overlap with confidently-placed neighbours — so the
   fine stage should be *re-running the `sequence_decode` DP with sharper
   per-span emission curves*, not independent per-span argmax.

Spike scripts: `/tmp/spike_p0{b,d,e}.py` (not committed — throwaway).

## Risks (tied to the domain taxonomy)

- **Beatless acappellas** — no beat grid (open Q in `project_variant_mert`).
  Fall back to continuous chroma/onset subsequence DTW (no beat-sync) for those.
- **Reverb tails / borrowed endings** — span boundaries are fuzzy *in the GT
  itself*; some residual "error" is label noise and caps achievable MAE. The
  abstain lane is the honest handling, not forcing a number.
- **Mashups / overlaps** — stems mitigate but don't fully separate; two vocals
  at once will still confuse the vocal channel. Confidence gating catches these.
- **Pitch + tempo simultaneously** — beat-sync handles tempo, rotation handles
  pitch; the cross term (extreme pitch *and* tempo) may need the DTW escalation.

## Why not just a better MERT head

Already falsified by measurement (oracle ref → ~900 s argmax error, all layers).
Pooled self-supervised embeddings encode *timbre/identity*, which is shift- and
tempo-robust by design — exactly the property that makes them **bad** at
pinning a moment in time. Localization wants the opposite: a feature that *is*
sensitive to precise pitch/onset content. Chroma + beats is that feature.
