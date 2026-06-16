# Inference-time gain estimation — scope

**Status:** scoping (2026-06-16). **Owner:** alignment_prototype.
**Prereq context:** [[project_gt_gain_curve]], `workspaces/alignment_prototype/{joint_decode_probe,transition_probe}.py`.

## Why this exists

The **dominance-window decode** recovers a bed buried under a crossfade (BB12
slot 111: single-bed 0% → dominance 100% exact<2s; IN-overlap 59%→71%). It works
by decoding each bed on the window where it is the *loudest* bed, then
propagating. To pick that window it needs, per bed, a **gain envelope** over the
mix timeline `gain_i(t)` — when each bed is audible/dominant.

Today that comes from the **GT `gain_curve`** (regime 1). A deployed aligner has
no GT. **Estimating `gain_i(t)` from the mix alone is the gate** between the
validated mechanism and a usable aligner. This doc scopes it.

## Goal

Given the mix (full + Roformer stems `mix_instrumental`/`mix_vocals`), the
candidate ref tracks (identities from the tracklist) and tentative placements,
output a per-bed gain envelope `ĝ_i(t)` over the mix timeline — enough to rank
**dominance** (which bed is loudest at t) and find each bed's clean window.
Absolute calibration is not required; relative dominance is.

## What was ruled out (measured 2026-06-16 — do not retry these)

Three cheap analysis signals were tested on the BB12 111-under-112 crossfade
(111 plays buried at GT gain 0.16 under 112 at 1.0). **All fail to track gain:**

| Signal | Buried region (GT gain 0.16) | Loud window (1.41) | Verdict |
|---|---|---|---|
| Best-offset chroma matched-filter | 0.75–0.93 | 0.90–0.96 | saturated — chroma self-similarity matches *some* offset everywhere |
| Chroma corr @ GT-trajectory offset | 0.68–0.90 | 0.90–0.96 | still high — **chroma is L2-normed per frame, so it discards level**, the one thing "buried" is about |
| Per-frame LS mel scale (α = ⟨mix,ref⟩/⟨ref,ref⟩) | 1.3–18.8 (wild) | 2.1–5.3 | contaminated — two beds share spectral bands; frame energy can't be attributed to one bed |

Root cause: at the frame level, two simultaneous beds occupy overlapping
harmonic/spectral space. You cannot attribute energy to bed A vs bed B **without
a model of what each bed sounds like in this mix.** The same wall sank oracle
spectral *cancellation* earlier (warp/master mismatch → 87% residual). Hand-rolled
signal processing is not the path.

## Approaches

### A. Analysis-based (per-frame gain from DSP) — **rejected**
Documented above; left here so it isn't re-attempted. Chroma is level-blind;
mel attribution is defeated by band-sharing + master/warp mismatch.

### B. Learned gain estimator — **the target**
A small model that *learns* what each bed contributes:
- **Input:** mix window features (log-mel + the Roformer stem it routes to) +
  candidate ref features at the tentative placement (so the model sees "this is
  the ref; how much of it is in the mix here?").
- **Output:** per-bed gain envelope `ĝ_i(t)` (or a softmax dominance map over the
  beds active at t).
- **Supervision:** the GT `gain_curve` we now persist — directly the regression
  target. Dominance map = `argmax_i gain_i(t)` from GT.
- **Loss:** envelope regression (Huber) + a dominance cross-entropy; or frame-wise
  ranking loss (only relative dominance matters).
- **Why it can work where DSP can't:** it learns the timbral signature of each
  ref *in context*, so it can attribute shared-band energy — exactly the thing
  per-frame LS can't do.

### C. Human-in-the-loop bridge — **pragmatic, available now**
The labeling workflow already produces gain curves. Until B is trained, the
aligner can: (1) run dominance decode with gains from a quick human fader pass,
or (2) **abstain** on heavy-overlap regions and flag them for review (margin is
near-zero there — ties into [[project_abstention_margin]] and the
[[project_open_set_alignment_endstate]] human-confirm loop). Overlap regions are
exactly where the aligner *should* be least confident.

## The real bottleneck: training data

B is gated on **gain-curve GT across many sets** — today it exists for **BB12
only** (n=1). The unlock is cheap and already built: **the GT export computes
`gain_curve` from any labeled `.als`** ([labeling/export_als_to_gt.py]). So the
next step is not more DSP — it is:

1. **Harvest gain curves** by re-exporting every already-labeled `.als` in
   `~/aligning/` through the updated exporter — free training data from existing
   labels. As of 2026-06-16 there are **4 labeled sessions beyond BB12** (BB11
   `2nvzlh2k`, BB10 `w1mgcjt`, murph `pwgrrb1`, + BB12), so a one-off re-export
   sweep takes gain-curve GT from n=1 to ~4 sets. Mind the enrich-then-merge
   rule (never bare-overwrite an enriched GT — see [[project_gt_gain_curve]]).
2. Once ≳ a handful of sets carry gain curves, train estimator **B** and validate
   it the same way we validated the decode: does dominance-from-`ĝ` recover the
   buried beds that dominance-from-GT-gain does, **without** GT?

## First experiment (when picked up)

1. Re-export gain curves for all labeled sets; report how many bed-overlap spans
   that yields (the effective training/eval set size — guard against the
   small-n trap, [[feedback_small_sample_regressions]]).
2. Train a minimal estimator B (start: log-mel + ref-at-placement → per-frame
   dominance logit).
3. **Metric:** (a) frame dominance accuracy vs GT; (b) end-to-end — swap `ĝ` for
   GT gain in `transition_probe`'s dominance decode and check IN-overlap exact<2s
   stays near the 71% the oracle gains achieved. Success = within a few points.

## Open questions

- Two beds in the **same** stem (`mix_instrumental`) don't separate via Roformer —
  the estimator must disambiguate them from ref identity alone.
- >2 simultaneous beds (BB12 is 2–3 layers deep, [[project_bb12_gt_state_taxonomy]]).
- Does estimated gain need to be calibrated, or is the dominance *argmax* enough
  for the decode? (Likely the latter — the decode only uses the window choice.)
