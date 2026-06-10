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

### Fingerprinting follow-up (same day) — flashes, not a silver bullet

Built a minimal Shazam-style constellation matcher (sparse STFT-peak landmark
hashes → offset-histogram vote), tempo handled by stretching the source to the
mix's local bar:

| Span | chroma (beat-sync) | fingerprint | **best of the two** |
|---|---|---|---|
| 086 Pizza | **3.9 s** | 18.6 s | **3.9 s** |
| 089 My Window | 14.5 s | **0.3 s** (sub-bar!) | **0.3 s** |
| 075 Pinball | 13.2 s | **2.9 s** | **2.9 s** |
| 068 Work From Home | 25.7 s | 65 s | 25.7 s |
| **<8 s hits** | 1/4 | 2/4 | **3/4 (2 sub-bar)** |

Fingerprinting alone is **not** a silver bullet — its offset histograms are not
sharp (peak barely clears the noise floor; My Window 132 vs 122), because the
dense multi-track EDM mix produces many spurious hash collisions. A tempo-ratio
*sweep* selecting by max votes made it **worse** (0/4) — vote count rewards
diffuse collisions; **sharpness (peak/second), not count, is the right
selector** and the lesson for P1.

**The actual finding — fusion, not a single feature.** No one method dominates,
but *different cheap channels nail different spans to sub-bar*, and each hit is a
true sub-bar lock, not luck. Oracle channel-selection over {chroma, fingerprint}
= **3/4 <8 s, 2/4 sub-bar**. Only Work From Home resists everything — a remix
with vocals where the source tempo estimate blew up and the matched master may
not be the one the DJ used.

### Revised P1 (supersedes the chroma-only module above)

1. **Multi-channel emission** — compute chroma-DTW *and* constellation-fingerprint
   curves per span; the fusion is selection-by-sharpness (the channel whose peak
   most clears its own noise floor wins that span), not averaging.
2. **Confidence = peak sharpness.** Below threshold → abstain to human review
   (Work From Home is the canonical abstain case). This is what makes the stage
   safe rather than a forced guess.
3. **Joint, not per-span.** Feed the fused per-span curves back into the
   `sequence_decode` DP so neighbour monotonic constraints rescue the spans no
   channel locks in isolation — the spike tested the *hardest* (isolated) setting.
4. **Source-master QA upstream.** Work-From-Home-class failures may be an
   *ingest* problem (wrong master), not a placement problem — cross-check before
   blaming the aligner (the correctness-vs-accuracy rule).

Spike scripts: `/tmp/spike_p0{b,d,e}.py`, `/tmp/spike_fp{,2}.py` (throwaway).

## Two ideas evaluated (2026-06-10)

**(A) Three-stream MERT (vocal / instrumental / full).** Sound for *identity*
on mashups (each overlaid layer matches its own stem) and could sharpen the
*coarse* per-bar curve by removing cross-layer contamination — but it **cannot**
reach fine placement: the MERT export is one vector **per bar (~2 s)**, a
resolution wall below the sub-bar target. Measured prior: full-mix MERT can't
localize even against a clean oracle ref (~900 s, all layers 0–24). Stems are
the right lever — but on the **DSP matcher** (chroma/fingerprint run per-stem),
not on MERT. Folded into revised-P1 step 1 as stem-aware channels.

**(B) Bootstrap from scraped tracklist cue times.** *(Corrected 2026-06-10 —
the user was right; my first read was wrong.)* The BB12 cues are **not broken**:
`cue/gt` is a consistent **~2.83× scale** (median ratio 2.83, slots 6–33 all in
2.5–3.3). My earlier "148 s / noise" was a non-robust affine fit poisoned by
outliers (slot 3's bad GT pairing + the zero-cue w-layers dragging the slope).
With proper calibration the cue predicts GT set_start at coarse quality:

| Calibration | median residual | <16 s | <30 s |
|---|---|---|---|
| robust scale (gt = 0.353·cue) | 26 s | 11/38 | 21/38 |
| isotonic (monotonic) | 17 s | 19/38 | 21/38 |
| **bootstrap: 5 anchors → predict 33** | **21 s** | 12/33 | 20/33 |

So the bootstrap works at **coarse** resolution (~20 s, on par with the existing
anchor prior), **not** fine. The ~20 % per-point jitter (and the 2.83× scale,
likely the cue referencing a longer/different mix version) caps it there. Its
real value: a handful of audio-confident anchors calibrate the cue curve, which
then places the ambiguous majority — see the synthesis below.

## Fusion + joint decode result (2026-06-10) — partial, redirects to anchor-and-fill

Built the revised-P1 two-stage pipeline (coarse MERT → per-span stem-aware
chroma+fingerprint emission curves over the coarse window → joint monotonic
re-decode) and ran it on 29 held-out BB12 spans with local audio:

| Method | median | <2 s | <8 s | <16 s |
|---|---|---|---|---|
| coarse (MERT) | 37.7 s | 1 | 3 | 5 |
| chroma-only joint | 51.2 s | 0 | 3 | 3 |
| fingerprint-only joint | 39.0 s | 1 | 3 | 4 |
| **fused per-span argmax** | 35.2 s | **3** | **6** | **7** |
| fused + joint decode | 39.9 s | 1 | 3 | 4 |

Two hard lessons:
1. **Fusion helps only the distinctive minority.** Fused argmax *doubled* sub-bar
   hits (3→6) — but only ~6/29 spans have audio distinctive enough to localize;
   the median barely moved because the other ~23 are self-similar noise.
2. **Joint decode BACKFIRES here.** Forcing a globally-monotonic path through
   mostly-flat curves drags the few good peaks off their true position — worse
   than independent argmax. The neighbour-rescue hypothesis fails when most
   emissions are uninformative.

**Redirect — anchor-and-fill, not localize-every-span.** The ~20 % of spans that
audio nails to sub-bar become **hard anchors**; those anchors *calibrate the cue
curve* (idea B); the calibrated cue + monotonic order + known span durations
place the ambiguous ~80 %. Audio gives precision where it can, the cue gives
global shape, and the anchors are exactly the "few points" the cue calibration
needs — the two ideas compose instead of competing.

## Anchor-and-fill result + the wall (2026-06-10)

Audio-confident spans (fused-curve peak z ≥ THR) → hard anchors → isotonic
calibration of cue→time → calibrated cue fills the rest. On 29 held-out spans:

| Method | median | mean | <8 s | <16 s | <30 s |
|---|---|---|---|---|---|
| coarse (MERT) | 37.7 s | 42 s | 3 | 5 | 11 |
| anchor-fill THR=3.0 | 37.7 s | 106 s | 5 | 7 | 12 |
| anchor-fill THR=4.0 | **31.3 s** | 99 s | 4 | **9** | **14** |

Modest body improvement (<16 s: 5→9), but a fat tail (mean → 99 s) from
cue-fill outliers, and only ~1–3 eval spans are anchors so most predictions ride
the ~20 s cue prior. **Net: roughly coarse-parity, not a breakthrough.**

**The wall, stated plainly.** Every signal we have — MERT (bar-resolution),
chroma/fingerprint (self-similar → flat for ~80 % of spans), scraped cue (~20 s
after calibration), tracklist order — is a *coarse* ~20–40 s prior. Fusing
coarse priors yields a coarse result. Sub-bar placement of the ambiguous
majority is **not reachable by independent per-span matching against any of
them.** Six methods now agree on this.

## Bold method — global program-to-mix alignment (proposed, not built)

Stop matching spans independently. Instead **reconstruct the expected program**
and align the whole thing at once:

1. Concatenate the known source sections in tracklist order, each tempo-warped
   to the mix's local tempo and placed at its `ref_start`, into a **synthetic
   reference timeline** (we have identity 100 % + ref_start 0.8 s — the
   ingredients are reliable).
2. One **global subsequence DTW** between the synthetic-reference chromagram and
   the actual-mix chromagram → a single monotonic warping that maps every source
   position to a mix time, placing **all spans jointly**.

Why this can break the wall where per-span matching can't: a self-similar drop
that's locally ambiguous is **pinned by its neighbours in the global path** — the
warping can't misplace it without breaking the alignment of the tracks around it.
This is how audio-to-score alignment works (whole-piece DTW, not per-note), and
it uses the full harmonic *trajectory* of the set, not isolated snippets. The
anchors and the cue prior become DTW band constraints (limit the warp corridor),
not independent guesses. Higher build cost (tempo-warp + concatenate + one big
DTW), but it's the first method that's *qualitatively* different from "score
each span against the mix."

Fallbacks if global DTW also stalls: (a) phrase-grid snapping — constrain starts
to boundary-probe peaks + 8/16/32-bar spacing; (b) a learned fine head once more
GT sets exist (one BB12 overfits).

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

## RESULT — global program-to-mix DTW (2026-06-10): FAILED catastrophically

Built and ran (`/tmp/spike_global_dtw.py`). Synthetic reference = 135 source
sections footprint-tiled (gap-to-next-predicted-start, so reference length
3211 cols ≈ mix 3734 cols rather than overlapping past it), each loaded at the
predicted `ref_start`, in predicted order. One subsequence DTW (ref matched
within mix, free intro offset) over ~1 s L2-normed CQT-chroma columns, with a
12-roll global transposition search.

| method | n=29 eval | median | mean | <8s | <16s | <30s |
|---|---|---|---|---|---|---|
| coarse (MERT decode) | | **37.7s** | 42.1s | 3 | 5 | 11 |
| global DTW | | 1379.5s | 1407.8s | 0 | 0 | 0 |

The DTW **collapsed**: the entire 3211-col program warped into the mix's
257–644 s band — a ~6:1 compression of a 62-min program into ~6 min. **Smoking
gun:** all 12 key-transposition rolls returned *identical* per-step cost 0.1
(cosine ≈ 0.9 everywhere). The program↔mix chroma cost surface is **flat** — no
harmonic gradient anywhere — so a global DTW has nothing to align to and
produces a degenerate warp. Transposition being irrelevant proves there is no
dominant tonal structure to match; it is the same self-similarity wall, now at
the global scale.

The only untried variant — banding the DTW to a ±60 s corridor around the
coarse diagonal — cannot help: the within-band cost is equally flat, so it
returns the coarse prior (~37 s) at best.

### Verdict — per-mix DSP/DP is exhausted (7 methods)

MERT · chroma (3 variants) · mel-DTW · fingerprint · fusion+joint · anchor-fill
· **global DTW** all bottom out on one fact: BB12's mix audio carries no
per-moment discriminative signal any decoder can exploit. The within-set 0.2
eval split is a single **leaky** split (same mix/DJ/render), not generalization
evidence. **Deliverable holds:** coarse 37–39 s MAE, identity 100%, ref_start
0.8 s.

### The actual unlock — more labeled sets (next action)

Hand-align 5–10 more DJ sets (different DJs/styles) → leave-one-**set**-out CV
+ a learned placer become meaningful (a learned placer on n=1 only memorizes
BB12). Audio acquired via the **new** pipeline: full-track download + Demucs
all-stems + identity-correct, logged to `track_audio_correction` — NOT the old
YTM-search-`hits[0]` rescue path (mashup/bootleg mismatch). Stage into
`~/aligning/`.
