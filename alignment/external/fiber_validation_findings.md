# External fiber validation — HuBERT fibers vs SALAMI human structure

> **Headline numbers: [docs/alignment_status.md](../../../docs/alignment_status.md)** (canonical).
> This doc owns the SALAMI validation detail (P .88 / R .06, per-arm P/R). The
> canonical doc frames the +20pp fiber-aware lift as a contribution and cites
> this file for the precision proof.

**Question.** Do this repo's self-repeat classes (`ref_fibers.compute_fibers`,
HuBERT-L9 + silence-gate) actually recover *human-annotated section repeats* —
independent of our tiny Big Bootie GT? The fiber-aware trajectory metric and the
`fiber_ambiguity` abstain signal both rest on the fibers being real; this is the
first external check that they are.

**Setup.** `fiber_validation.py` (this dir). Runs the REAL interpreter
(`continuity_refine._compute_hubert(path, layer=9)` → `compute_fibers(feat, FPS,
audio_path=...)`, exactly as `path_decode.fibers_for`), converts fiber per-frame
labels and the human annotation to a common dense grid, scores with
`mir_eval.segment.pairwise` + `.nce`.

```
venvs/audio/bin/python -m alignment.external.fiber_validation --n 30
```

## Dataset & audio (honest accounting)

- **SALAMI** via mirdata. Annotations are the uppercase (large-scale) section
  letters — a segment tagged `A` recurring = a repeat, which is precisely what a
  fiber claims to find.
- **Audio: SALAMI-Internet-Archive subset** (`id_index_internetarchive.csv`, 477
  tracks with archive.org URLs). The 2012-era direct derivative filenames have
  **rotted** — only ~5% resolve by literal URL (404/403). But the *items* still
  exist, so we resolve each via the **archive.org metadata API** (item-id + track
  stem → current `.mp3`): **~96% resolve** that way. This is why the fetch worked
  where a naive URL loop would report "audio unavailable".
- **n = 30 scored** (target 30, hit exactly). 31 attempted, **1 fetch fail**, 0
  HuBERT failures. 5 candidates skipped for having no repeated section in the GT
  (nothing to recover). No silent truncation.
- **CAVEAT (matters for interpretation):** SALAMI-IA is **live jam-band / improv**
  audio (long solos, sprawling non-repeating stretches) — structurally looser
  than the studio pop/EDM our aligner's refs actually are. So the absolute number
  here is a **pessimistic floor**, not an estimate of fiber quality on our corpus.
  It still cleanly answers the *direction* question (over- vs under-merge).

## Timebase / interval conversion (the fps-bug-prone part)

Fibers return per-frame labels at `label_hz` (~8 Hz, the compute_fibers downsample
grid — NOT the 43 Hz HuBERT grid). GT is (intervals seconds, letters). To compare
with zero fps/offset drift:

1. common dense grid at `GRID_HZ=10` over `[0, min(dur, gt_end, fiber_end))`;
2. sample BOTH labelings at cell centers — fiber via `round(t·label_hz)`, GT via
   interval containment;
3. emit **identical** unit-width intervals for est and ref, so mir_eval sees a
   pure clustering comparison (not a boundary comparison).

Fiber silence (`-1`) → a **unique singleton label per frame** (so "ungrouped" is
never rewarded as one big cluster). This is the faithful encoding: a `-1` frame
that the human put in a repeated section correctly earns 0 recall credit for the
pairs it failed to join.

## Results (n = 30)

| metric | mean | median |
|---|---|---|
| **pairwise F** | **0.104** | **0.056** |
| pairwise **precision** | **0.882** | **1.000** |
| pairwise **recall** | **0.065** | **0.029** |
| V-measure (nce) | 0.402 | 0.388 |
| — S_over (completeness/recall side) | 0.272 | 0.241 |
| — S_under (homogeneity/precision side) | 0.941 | 1.000 |
| fiber coverage (frac frames in any fiber) | 0.179 | 0.146 |

Sanity baselines (same ref/grid): **all-one-cluster F = 0.521**, **each-own-frame
F = 0.001**. GT-vs-GT F = 1.0 by construction.

- **5 of 30 tracks produced ZERO fibers** (coverage 0.00) despite the human
  marking 6–23 sections with 4–8 unique letters.
- **corr(coverage, pairwise-F) = 0.83** — F is almost entirely determined by how
  much of the track the fibers cover. When coverage is decent (0.4–0.6, e.g.
  track 1188 cov=0.51 F=0.625, 1303 cov=0.48 F=0.320) the score is respectable;
  most tracks sit at cov ~0.1 → F ~0.05.

## Verdict

**Our HuBERT fibers recover human repeat structure POORLY on this material, and
the failure is unambiguously UNDER-MERGING (low recall), not over-merging.**

- **Precision is excellent (median 1.0, mean 0.88).** When a fiber groups two
  frames as the same section, the human almost always agrees. The `s_under=0.94`
  homogeneity confirms it. So the fibers are **trustworthy when they fire** — the
  design intent ("real repeat classes, never false merges") holds up externally.
  This directly validates the *precision* premise the `fiber_ambiguity` /
  fiber-aware credit relies on: a same-fiber verdict is rarely a lie.
- **Recall is very low (median 0.03) and coverage is ~18%.** The fibers detect
  only a small fraction of the repeats the human annotates, and skip 5/30 tracks
  entirely. `s_over=0.27` (completeness) says the same. The `min_repeat_s=6`,
  `verify_thresh=0.5` diagonal + average-linkage pipeline is a **conservative
  detector** — by design it declines rather than risks a false merge (this is the
  documented v1→v2 fix in `ref_fibers.py`: the old spectral version over-merged and
  John heard false merges; the current version swung hard to the precision side).

**Two forces mixed in the low recall, and this test can't fully separate them:**
1. genuine conservatism of the detector (the same thing that buys the 1.0
   precision), and
2. the jam-band material simply having long non-repeating improv stretches +
   HuBERT being blind to purely *melodic/harmonic* repeats with no lyric content
   (`ref_fibers.compute_fibers_fp`'s docstring already notes HuBERT misses melodic
   repeats — Love On Me 0:00≈2:32 scores 0.11). SALAMI-IA is instrumental-heavy,
   which plays to that blind spot.

**What this means for the repo's fiber-dependent machinery.** The fiber-aware
metrics and `fiber_gate` rest on a **sound but narrow** foundation: fibers are
high-precision, so crediting a within-fiber decode pick as correct will rarely
launder a wrong answer — good. But fibers are **low-recall / low-coverage**, so
fiber-awareness only helps on the minority of spans where a repeat was actually
detected; it is *not* a general safety net, and "not in any fiber" carries almost
no information (most real repeats also aren't). This is consistent with the
in-repo BB numbers (fiber-aware lifts traj-acc only +6pp, 53→59%) — the lift is
small precisely because coverage is low. The `fiber_ambiguity` abstain signal is
safe to trust *when it fires* but will stay silent on most genuinely-ambiguous
repeat placements.

---

# Phase 2 — Multimodal complementarity (chroma arm + union)

**Question.** The phase-1 under-merge is (hypothesized) HuBERT being *phonetically*
blind to melodic/harmonic-only repeats. If so, a **harmony** modality (chroma)
should catch the repeats HuBERT misses, and their UNION should recover far more
than either alone. Tests the axis-decomposition claim (song ≈ timbre × harmony ×
language) on external GT.

**Setup.** SAME 30 SALAMI-IA tracks, SAME mir_eval scoring, SAME
`ref_fibers.compute_fibers` — only the feature changes: `refine_ref_offsets.chroma`
(chroma-CQT, 12-dim, SR/HOP grid, FPS≈43.07) instead of HuBERT-L9. For the union
("a repeat pair is caught if EITHER modality merges it") mir_eval's `pairwise`
can't score a non-partition, so complementarity uses **frame-pair set algebra**
(identical P/R/F definition, works on the union mask). All three arms scored
against the SAME GT same-section pair mask.

## Results (n = 30, same tracks)

| arm | pairwise-F | precision | recall | coverage | # fibers |
|---|---|---|---|---|---|
| **HuBERT** | 0.104 | **0.882** | 0.065 | 0.18 | 0–6 |
| **chroma** | **0.509** | 0.444 | **0.706** | 0.76 | **1 on 30/30** |
| baselines | all-one **0.521**, each-own 0.001 | | | | |

**Complementarity (pair-based, apples-to-apples):**

| | precision | recall | F |
|---|---|---|---|
| HuBERT | 0.715 | 0.064 | 0.104 |
| chroma | 0.444 | 0.706 | 0.509 |
| **HuBERT ∪ chroma** | 0.444 | **0.706** | 0.509 |

- **Union recall gain over HuBERT: +0.642** (0.064 → 0.706).
- Micro-averaged over all frame-pairs: **chroma recovers 67.1% of all GT-true
  repeat pairs that HuBERT missed.**
- **All 5 tracks where HuBERT found ZERO fibers got chroma coverage** (chroma
  recall on them: 0.90, 0.42, 0.27, 0.78, 0.34). Chroma rescues the tracks HuBERT
  completely whiffs.

## Verdict — the modalities ARE complementary, but chroma's arm is DEGENERATE

**The hypothesis is confirmed in direction and refuted in usefulness.**

1. **Direction: yes, exactly the predicted mirror image.** HuBERT = high-precision
   / low-recall (under-merge); chroma = low-precision / high-recall (over-merge).
   They fail on *different* repeats — chroma catches 67% of HuBERT's misses,
   including 5/5 zero-HuBERT tracks. This externally validates the repo's
   axis-decomposition premise: harmony and language see different structure.

2. **Usefulness: NO — the union is not worth building as-is.** The blocker is that
   **chroma collapsed to a single fiber on 30/30 tracks.** Its recall is high
   because it merges (nearly) the *whole track* into one class — the exact
   "chroma over-merges / blobs to one fiber" failure `ref_fibers.py` documents and
   forbids ("fibers MUST be HuBERT + silence-gated, never chroma; computing fibers
   on the chroma decode feature blobs to one fiber → fake 100%"). The external
   data confirms that rule quantitatively:
   - Union **mean F (0.509) is BELOW the all-one-cluster baseline (0.521)**, and
     beats all-one on only **16/30 tracks**. A modality whose F barely reaches
     "put everything in one bucket" is contributing trivial recall, not
     discriminative repeat detection.
   - Union precision drops to **0.444** — every union merge is now roughly as
     likely wrong as right. For the repo's use (crediting a within-fiber decode
     pick as correct), that precision is **not usable**: it would launder wrong
     placements about half the time, destroying the one property that made HuBERT
     fibers safe (phase-1 median precision 1.0).

**So: a naive HuBERT ∪ chroma union is a NO-GO** — it trades HuBERT's trustworthy
0.88 precision for chroma's degenerate 0.44, buying recall that is mostly the
all-merge trivial floor. This externally grounds the in-repo decision to keep
fibers HuBERT-only.

**What WOULD be worth building** (the real lesson): chroma's signal is real but
its *segmentation* is broken — `compute_fibers`'s diagonal + average-linkage
pipeline, tuned for HuBERT's sparse sharp diagonals, cannot cut chroma's
everywhere-high self-similarity into discrete repeats (chroma's diagonal is high
at *every* lag because harmony is uniform — the `compute_fibers_fp` docstring says
exactly this). A useful harmony arm needs a *different* detector (adaptive /
normalized SSM, `compute_fibers_fp`'s constellation match-density, or a learned
embedding), NOT `compute_fibers(chroma, ...)`. The complementarity is there to
exploit; running chroma through `compute_fibers` is not the way to exploit it.
Same jam-band caveat as phase 1 (chroma over-merge is plausibly worse on
harmonically-static improv than on studio pop).

## LYRIC arm (DALI) — SKIPPED, with reason

Not run. Two blockers, both flagged as acceptable-to-skip by the task:
1. **DALI needs a separate package** (`pip install 'mirdata[dali]'` — the `DALI`
   module; confirmed `ModuleNotFoundError: No module named 'DALI'` in our venv),
   and its audio is **not distributed** — every track is a YouTube fetch (flaky /
   slow, the exact risk we were told not to burn time on).
2. **DALI has no structure GT.** The best it could yield is lyric-line-repeat vs
   HuBERT-fiber *agreement* — a proxy, not a validation against human structure,
   and far weaker than the chroma-vs-SALAMI-GT measurement already in hand. Since
   phase 2 already answers the multimodal question decisively, DALI was not worth
   the download risk. It remains the right vehicle for the lyric arm if pursued
   later (reuse `ingest/` yt-dlp; expect the *chroma-like* profile — high-recall /
   low-precision — because lyrics recur across musically-different sections,
   consistent with the in-repo `compute_fibers_multi` over-merge verdict).

## Extensions (noted, NOT built)

- **LYRIC-fiber arm (DALI).** DALI (`mirdata dali`, needs `pip install
  'mirdata[dali]'`, audio via YouTube — reuse `ingest/` yt-dlp) has line/paragraph
  annotations with recurring lyric text. That's the natural GT for
  `ref_fibers._lyric_labels` / `compute_fibers_multi`'s lyric axis. Score identically
  (recurring line = repeat label). This would directly test the 2026-07-07 verdict
  that the lyric axis *over-merges* (~45% false new credits) — expect the mirror
  image of the HuBERT result: higher recall, **lower** precision, because lyrics
  recur across musically-different sections.
- **KEY/harmony-fiber arm (McGill Billboard / Isophonics chords).** mirdata
  `billboard` and `beatles` carry chord annotations; a repeated chord progression =
  a harmonic repeat. This is the GT for a *chroma*-fiber (the axis `ref_fibers`
  deliberately does NOT use for fibers because chroma blobs to one class). Scoring
  a chroma-fiber here would quantify exactly how badly chroma over-merges — the
  empirical justification for the "fibers MUST be HuBERT, never chroma" rule.

Together the three arms would map the axis-decomposition claim (timbre×harmony×
language) onto external GT: HuBERT=high-precision/low-recall (shown), lyric=high-
recall/low-precision (hypothesized), chroma=over-merge (hypothesized).

## Files

- `fiber_validation.py` — the harness (new; read-only reuse of `continuity_refine`,
  `ref_fibers`, `refine_ref_offsets`).
- `out/fiber_validation.json` — per-track + aggregate metrics.
- `.salami_data/` — mirdata annotations (auto-downloaded). `.salami_audio/` —
  resolved archive.org mp3 cache. `.feat_cache/` (repo-shared) — HuBERT arrays.

---

# Phase 3 — `harmony_fibers` detector (the "different detector" phase 2 called for)

**Question.** Phase 2's lesson was that the harmony *signal* is complementary but
`compute_fibers(chroma)` is the wrong detector (degenerate 1-fiber blob on 30/30).
`harmony_fibers.py` is the response: CENS chroma → diagonal-enhanced SSM →
time-lag stripe extraction with an **adaptive per-track threshold** + a
uniform-track abstain gate + matched-filter span verification. Same output
contract as `compute_fibers`. Does it (a) avoid the blob, (b) preserve the ≥~0.8
precision the fiber design rule demands, (c) add coverage/recall over HuBERT —
at its **shipped defaults** (SALAMI is validation, not a tuning set)?

**Setup.** SAME 30 SALAMI-IA tracks, same harness/scoring
(`fiber_validation.py --harmony`, seed 0). Post-hoc pooled/collapse analysis:
`fiber_validation_phase3.py` (sibling module, reuses the harness's loaders and
pair algebra; writes `out/fiber_validation_phase3.json`).

## Results — per-track means (mir_eval partition, apples-to-apples w/ phases 1–2)

| arm | pairwise-F | precision | recall | coverage | # fibers |
|---|---|---|---|---|---|
| HuBERT | 0.104 | 0.882 | 0.065 | 0.18 | 0–6 (0 on 5/30) |
| chroma via `compute_fibers` | 0.509 | 0.444 | 0.706 | 0.76 | **1 on 30/30** (blob) |
| **`harmony_fibers`** | 0.089 (med 0.001) | 0.928* | 0.070 | **0.10 (med 0.00)** | **0 on 22/30**, 1–2 else |
| baselines | all-one 0.521 · each-own 0.001 | | | | |

\* mir_eval returns P=1.0 on the 22 empty tracks — the per-track mean is not the
honest precision readout. Use the pooled numbers below.

## Pooled (micro) frame-pair P/R — the honest gate readout (56.1 M GT-true pairs)

| arm | precision | recall | F | grid coverage (mean / med) |
|---|---|---|---|---|
| HuBERT | 0.866 | 0.056 | 0.105 | 0.179 / 0.146 |
| `harmony_fibers` | 0.797 | 0.039 | 0.074 | 0.097 / 0.000 |
| **HuBERT ∪ harmony** | **0.832** | **0.091** | **0.165** | **0.252 / 0.217** |

- Fired-only (the 8/30 tracks where harmony produced ≥1 fiber): pooled P 0.797,
  R 0.285, mean coverage 0.36. Per-track fired pair-precision is **bimodal**:
  {1.00, 1.00, 0.92, 0.82} vs {0.57, 0.52, 0.50, 0.49} — half the fired tracks
  are coin-flip.
- The pairs the union **adds** over HuBERT (harmony-only mass) are ≈0.785
  precise (derived from the pooled counts) — right at, slightly under, the 0.8
  bar. Micro harmony-only-recovered = **3.6%** of all GT-true pairs (vs chroma's
  degenerate 67%).
- HuBERT-zero-fiber tracks rescued: **1 of 5** (1491: harmony R 0.21 at P 0.49;
  the other four stayed empty).

## Degenerate-collapse check — PASSED (the chroma failure mode is fixed)

| | `harmony_fibers` | chroma via `compute_fibers` (reference failure) |
|---|---|---|
| # fibers per track | **0 on 22/30**, 1 on 4, 2 on 4 | 1 on **30/30** |
| max-fiber-share of track | mean 0.09, max 0.81 | mean 0.76, min 0.49 |

The failure mode inverted, exactly as designed: instead of merging everything,
the uniform-track gate **abstains** (returns zero fibers) on 22/30 — the honest
answer on harmonically-static jam material. The one large-share case (1216,
one fiber = 81% of track) is *correct*, not a blob: that track's GT is 16
sections with only 3 unique letters (one dominant repeat class), and the fiber
scores P 0.92 / R 0.69 against it. Within fired tracks the largest fiber is
0.57–1.0 of the fibered mass, but the fibered mass itself is section-scale
(cov 0.08–0.81), not the whole track.

## Sensitivity sweep (pre-registered, one knob at a time — robustness check ONLY)

Declared before running: `local_prom ∈ {0.04, 0.10}`, `glob_pct ∈ {85, 97}`
around the shipped defaults (0.07 / 92). Pooled micro:

| config | harm P | harm R | cov | fired | union P | union R |
|---|---|---|---|---|---|---|
| **defaults** | **0.797** | 0.039 | 0.10 | 8/30 | **0.832** | 0.091 |
| local_prom=0.04 | 0.582 | 0.024 | 0.10 | 8/30 | 0.746 | 0.075 |
| local_prom=0.10 | 0.708 | 0.019 | 0.07 | 7/30 | 0.818 | 0.071 |
| glob_pct=85 | 0.759 | 0.037 | 0.09 | 8/30 | 0.818 | 0.090 |
| glob_pct=97 | 0.714 | 0.045 | 0.11 | 9/30 | 0.791 | 0.095 |

Two readouts: (1) the shipped defaults are the best operating point in the
neighbourhood — every perturbation *lowers* precision without buying coverage,
so the result is not a lucky threshold; (2) coverage is pinned at ~0.10 across
all configs — the low recall is **structural** (the abstain gate + this
material), not a threshold artifact. No config reaches higher coverage at
gate-passing precision, so there is no better SALAMI operating point being
hidden by the defaults.

## Verdict — standalone: NOT usable. Union with HuBERT: usable-with-caveats (marginal).

**Against the gate** ("precision ≥ ~0.8 at coverage meaningfully above HuBERT's
R 0.06 / cov 0.18"):

- **Standalone harmony arm: FAILS.** Pooled P 0.797 is borderline-at-the-bar,
  but coverage (0.097, median **0**) is *below* HuBERT's, not above. It cannot
  replace or stand beside HuBERT as an independent arm.
- **HuBERT ∪ harmony: passes, barely.** Union precision 0.832 stays over the
  bar, coverage 0.179 → 0.252 (+41% rel), micro recall 0.056 → 0.091 (+63%
  rel). But the increment is concentrated on 8/30 tracks (22 unchanged), the
  added pairs are ≈0.785 precise (at/under the bar), and fired-track precision
  is bimodal — 4 of the 8 fired tracks would launder ~half-wrong merges into
  any fiber-credit that consumed them per-track.

**Direction (the trustworthy readout on this material):** the phase-2 blocker is
genuinely fixed — `harmony_fibers` never blobs; it inherits (and amplifies) the
HuBERT-style under-merge/abstain profile instead. That is the *safe* failure
direction under the "fibers must never launder wrong answers" rule. The
jam-band caveat cuts in the detector's favour here: the abstain gate zeroed
22/30 tracks precisely because harmonically-static improv has no separable lag
structure; studio pop/EDM (our corpus) has sharper sectional repeats, so fire
rate and recall should be higher there — but SALAMI cannot prove that.

**Recommendation.** Do NOT wire harmony_fibers as a standalone fiber source.
The only defensible configuration is **union-with-HuBERT, opt-in** (like
`fiber_gate`), and only after re-measuring the two things SALAMI leaves open on
our own GT: (1) fire rate / coverage gain on BB material, (2) whether the
per-track bimodal precision persists — if it does, add a per-track acceptance
check (e.g. require the matched-filter verification margin to clear a floor)
before letting harmony merges into fiber credit. Expected value: modest — even
a favourable BB fire rate leaves the union recall far from the multimodal
unlock phase 2 hoped for; the melodic-repeat blind spot is narrowed, not
closed.

## Files (phase 3)

- `harmony_fibers.py` (module root) — the detector (CENS + time-lag stripes +
  adaptive threshold + abstain gate). `ref_fibers.py` untouched.
- `fiber_validation.py` — `--harmony` arm (adds the arm + HuBERT∪harmony union;
  HuBERT/chroma arms unchanged).
- `fiber_validation_phase3.py` — pooled micro P/R, collapse check, pre-registered
  sweep; writes `out/fiber_validation_phase3.json`.
- `out/fiber_validation.json` (now includes harmony aggregates),
  `out/fiber_validation_phase3.json`, `out/phase3_defaults.log`,
  `out/phase3_sweep.log`.
