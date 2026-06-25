# Alignment as a research project — plan

**Status:** draft (2026-06-25). Living doc. The prior-art section will be
backfilled by the running `deep-research` report; the framing below already
accounts for the closest ancestor.

## 0. The one-paragraph thesis

DJ-mix reverse-engineering is **not** open territory — and that's the most
important correction the prior-art sweep produced. Kim et al. (ISMIR 2020) already
do mix→track subsequence alignment + cue/transition stats at corpus scale; Schwarz
& Fourer (UnmixDB) already recover *editable* parameters (offset, time-stretch,
cue points, fade curves); and **André/Schwarz/Fourer 2024 (multi-pass NMF)** is
the current SOTA — it jointly recovers *arbitrary warp (loops & jumps) + gain
curves* on the UnmixDB benchmark. So "editable reconstruction" is taken, and our
component numbers are **behind** that SOTA, not ahead. The genuinely open axes,
after subtracting their work, are narrower and must be the contribution:

1. **Stem-decomposed** identity+placement (vocal/instrumental/full layering — the
   acappella/instrumental axis), which the NMF-unmixing line does not model as a
   first-class identity dimension.
2. **Learned** alignment (their SOTA is signal-processing NMF/DTW) pretrained on
   synthetic mixes and **finetuned on real human Ableton GT**.
3. **Real-world scraped corpus at scale** (1001Tracklists, ~16k tracks) with a
   **DAW round-trip** (actual `.als`) as the representation — aimed at the
   downstream *generation* goal, not just analysis.

We are **not SOTA today** on identity or placement. The bet is that (1)+(2)+(3)
together — stem-aware, learned, real-corpus, DAW-round-trippable — is the unclaimed
quadrant. Everything below must beat the André-2024 NMF baseline on UnmixDB to be
credible.

## 1. Positioning vs prior art (verified, deep-research 2026-06-25)

| Work | Task | Reported result | Output | Relation to us |
|---|---|---|---|---|
| **André, Schwarz, Fourer 2024** — *DJ Mix Transcription w/ multi-pass NMF* (arXiv 2410.04198) | reference-conditioned unmix: warp + gain | warp incl. **loops/jumps** + gain, MAE on UnmixDB v1.1 (1931 mixes) | **editable params** | **THE bar to beat.** Ahead of our components; signal-processing not learned; no explicit stem-identity axis |
| **Schwarz & Fourer 2018–21** — UnmixDB + 3-stage pipeline | rough align (start+stretch) → sample-precise offset → cue/fade (TF-domain) | editable offset/stretch/cue/fade | **editable params** | closest ancestor to "editable"; closed-set; the benchmark + GT we adopt |
| **Kim, Choi, Sacks, Yang, Nam 2020** (ISMIR) | mix→track subsequence DTW, key-invariant beat-sync CENS+MFCC | cue median **~4–11 s** (cue type/feature dependent); corpus stats over 1,557 mixes | timestamps + statistics | direct placement yardstick — our 7.9 s is **comparable-to-worse**, not a win |
| **Sonnleitner & Widmer (Qfp)** 2014/16 — landmark quad fingerprint | which-track-when in mixes | disco 74.1% / mixotic 87.6% acc; 92–96% precision per-second; robust to 20k DB + ±20% tempo | track ID + time | identity SOTA; **does not** score within-ref placement ("not meaningful for repetitive content") |
| **Schwarz & Fourer NIME 2021** | transition gain/EQ recovery | sub-band convex opt; listening test n=14 (no timing metric) | gain/EQ trajectories | the EQ/gain piece of P4 |
| **Werthen-Brabants 2018** (Ghent MSc + GitHub extractor) | reverse-engineer DJ actions vs sources | JSON of mix actions | params (JSON) | engineering precedent; fingerprint backbone |
| **DJtransGAN 2022** (arXiv 2110.06525) | *generate* transitions w/ differentiable EQ+fader | GAN transition synthesis | audio | differentiable-mixer idea for the generation north star |
| **Audio-to-audio sync** (Müller/Zeitler; RNN-AMT) | warp same-piece | **10–20 ms** median (clean) | dense path | shows "good" alignment = ms; our seconds reflect the *harder* mix task |

**Benchmark standard:** **UnmixDB** (Schwarz & Fourer) — auto-generated mixes,
sample-accurate per-track start/end/cue + BPM/speed GT; 3 warp types × 4 effects;
v1.1 = 1,931 mixes. André 2024 reports **MAE on warp + gain** here. This is the
table we must enter.

**Must-read, in order:** (1) André/Schwarz/Fourer 2024 NMF — the SOTA + method to
beat; (2) Schwarz & Fourer UnmixDB/pipeline — benchmark + editable-param framing;
(3) Kim et al. 2020 — placement yardstick + corpus method; (4) Sonnleitner &
Widmer Qfp — identity SOTA + tempo-robust fingerprint; (5) Werthen-Brabants —
engineering precedent; (6) DJtransGAN — differentiable mixer for generation.

**Measured on real UnmixDB (2026-06-25, `eval_bench.py`, warp-varied slice):**

| method | set_start median | tempo % | identity rank@1 |
|---|---|---|---|
| grid_mf (matched filter, ours) | **~2.0–2.7 s** | 2–6% | 46% (chroma) |
| **fingerprint identity** | — | — | **89.7%** (SOTA band 74–88%) |
| NMF v0/v1 (André-style baseline) | ~25 s | 10% | (cross-talk limited) |

Reads: (1) our matched-filter placement is **Kim-comparable** (~2 s median);
(2) **fingerprint nearly doubles identity (46→90%)** and reaches the Qfp SOTA
band — the clean win; (3) our **NMF reproduction is behind** — real placement is
~25 s, bound by **cross-talk between spectrally-similar sequential tracks**, the
hard part André's full multipass method is built to handle. So on UnmixDB we are
**Kim-comparable on placement, SOTA-band on identity (via fingerprint), and not
yet matching the NMF SOTA on joint warp+placement.** The unclaimed quadrant (§0)
remains the shot; near-term, matched-filter + fingerprint beats chasing NMF.

## 2. Problem decomposition (four tasks + one output)

Keep these **separate in every eval** — they need different priors and conflating
them hides which one broke (the lesson of the collapse ladder:
[[project_collapse_ladder_findings]]).

1. **Identity** — which reference track is playing. Prior: closed tracklist pool
   (verify-not-search) + open-set abstain. Tools: landmark fingerprint, HuBERT-L9
   verification.
2. **Placement** — where in the reference each played segment starts. Prior: beat
   grid (the lever) + banded subsequence DTW. Metric: ref-offset MAE/median.
3. **Segmentation / transition** — span boundaries, transition length, **stem
   layering / superposition** (vocal-over-instrumental). Metric: boundary F1,
   transition-length error, stem-presence recall.
4. **Warp / unmix** — per-segment tempo ratio + non-linear warp + gain curve
   (+ EQ later). Metric: stretch % error, gain-curve error, marker-offset error.
5. **Editable output (the differentiator)** — a `.als` (or neutral structure)
   that round-trips: `parse ∘ print = id`. Metric: round-trip fidelity, % clips a
   human must fix.

## 3. The benchmark (cornerstone deliverable — build first)

`workspaces/alignment_prototype/eval_bench.py`: score any method across sets on
the §2 metrics, with **abstain-aware coverage-vs-accuracy curves**, not single
numbers. Datasets, in order of what they buy:

- **UnmixDB** (synthetic; exact GT warp/timing/gain) — already have
  `external/unmixdb.py`. The only source that can measure *sub-second* placement
  and *warp* honestly, at scale. Primary quantitative benchmark.
- **Hand-GT sets** (BB12 `1fsnxchk`, BB11 `2nvzlh2k`) — real, *editable* GT from
  the `.als` (warp maps, stems, gain curves). Our differentiator as a benchmark:
  Kim et al.'s GT was timestamps; ours has the warp map. Grow via the seeder →
  listen → correct flywheel (`render_review_snippets` → `seed_als_from_timeline`).
- **1001Tracklists slice** (the Kim et al. regime) — large, weak GT (tracklist +
  community cue points) for identity/cue at scale and for the descriptive-stats
  reproduction baseline.

Reuse: `trajectory_acc`, `labeling/ground_truth/schema.py`, `labeling/als_io.py`,
the GT YAML, and the `dtw_failure_eda.ipynb` metric functions.

**This is itself a paper contribution**: the first DJ-mix benchmark with
*editable / warp-level* ground truth.

## 4. Method roadmap (phased; reuse-first)

- **P0 — Harness + baselines.** §3 bench on **UnmixDB first** (it's the standard
  and where the SOTA lives). Reproduce three baselines so every later number has an
  honest floor and a real ceiling: **André-2024 multi-pass NMF** (the SOTA to
  beat — port or re-implement; their MAE on warp+gain is the target line), a
  **Kim-style** subsequence-DTW alignment, and a **fingerprint** identity baseline
  (Panako / `landmark_fp`). Until we match André on UnmixDB, we have no claim.
- **P1 — Identity.** Fingerprint shortlist → HuBERT-L9 verify → open-set abstain
  (margin-calibrated). Fixes the L3 instrumental-presence failure (fingerprint,
  not chroma). Reuse `landmark_fp.py`, `similarity_probe.py`.
- **P2 — Placement.** Beat-grid-anchored, identity-banded **subsequence DTW**;
  piecewise-linear **Viterbi warp** (`path_decode.decode_path`) for the marker
  set. Grid is the lever (24→8 s); target sub-second on UnmixDB.
- **P3 — Segmentation / superposition.** Boundary + transition-length detection;
  the **stem-presence pair** done right (fingerprint presence per stem). Reuse
  `continuity_refine` cross-channel scan, fixed to independent thresholds.
- **P4 — Unmix.** Gain curve (GT already has `gain_curve`), then EQ / non-linear
  warp residual. Borrow Schwarz–Fourer formulation.
- **P5 — Editable output.** `.als` round-trip via `labeling/als_io` (Law A:
  `parse ∘ print = id`, already 152/152 on BB12). The thing nobody else produces.
- **P6 — Learned aligner (north star).** Attention-over-catalog model
  ([[project_aligner_attention_design]]); **synthetic-mix pretrain on UnmixDB →
  finetune on hand GT** ([[project_alignment_bootstrap_flywheel]]). Probes plateau
  ~55–60%; the learned model has to beat both the stacked-prior baseline (P0–P5)
  *and* the André-2024 NMF SOTA or it isn't earning its keep.
  **Architecture template — Flamingo** (Alayrac et al. 2022, arXiv 2204.14198):
  useful *as a structural blueprint, not a method to copy*. Flamingo bridges
  **frozen** pretrained encoders with **gated cross-attention** and a
  **Perceiver-Resampler** that maps a variable-size set of encoded items to a
  fixed set of latents. Map onto our problem: freeze the audio encoders
  (HuBERT/MERT) for the catalog tracks, resample each candidate to a few latents,
  and let the mix sequence cross-attend over the candidate-set latents to emit
  identity+offset per frame — i.e. "few-shot align the mix against a provided set
  of songs." This is the concrete realization of attention-over-catalog and the
  natural way to ingest a variable-length tracklist without retraining. The
  generation end (DJtransGAN's differentiable mixer) is the eventual decoder.

## 5. Experiments / ablations (the science)

- **Oracle ceilings** — feed oracle identity / oracle grid / oracle stem
  separately; whichever oracle most collapses error is the binding constraint.
  Today's read: grid >> feature. Re-confirm per task, per set.
- **Feature ablation** — chroma / MFCC / HuBERT / MERT / fingerprint, scored
  separately for identity vs placement (they diverge — HuBERT helps ID, not
  placement).
- **Prior ablation** — grid-lock, closed pool, monotonic order, stem routing;
  the collapse ladder, run across ≥3 sets.
- **Cross-set generalization** — train/calibrate on BB12, test on BB11 +
  UnmixDB-held-out. *Nothing is believed in-domain on one set.*
- **Abstain economics** — coverage vs accuracy; what fraction can ship
  auto-confirmed vs human-in-the-loop.

## 6. Milestones

| Phase | Deliverable | "Done" = |
|---|---|---|
| M0 (wk 1–2) | `eval_bench.py` + Kim/fingerprint baselines on BB12+BB11+UnmixDB | one table, 4 metrics, abstain curves, ≥3 sets |
| M1 (wk 3–4) | Identity stack (fp+HuBERT+abstain) | open-set ID with calibrated abstain beats fingerprint-only |
| M2 (wk 5–7) | Placement (grid + banded DTW + Viterbi warp) | sub-second median on UnmixDB; <few-s on hand GT |
| M3 (wk 8–9) | Segmentation + stem superposition fixed | instrumental-presence recall ≫ 0; boundary F1 reported |
| M4 (wk 10–11) | Editable `.als` round-trip end-to-end on a *predicted* set | human fixes < X% of clips |
| **MVP-paper** | M0–M4 = "editable DJ-mix reconstruction + benchmark" | submittable to ISMIR even without P6 |
| M5+ | Learned aligner (synthetic pretrain → finetune) | beats the M0–M4 stacked-prior baseline |

## 7. What makes it publishable (and where)

- **Venue:** ISMIR (home of the ancestor) primary; DAFx / workshops secondary.
- **Contributions, in defensibility order:** (1) editable/warp-level **benchmark**
  + metrics; (2) **piecewise-linear warp recovery** that round-trips to a DAW
  (with the DTW+RDP-fails ablation as evidence); (3) **stem-aware, abstaining**
  open-set alignment; (4) **learned aligner** w/ synthetic pretrain (if it beats
  the baseline). Any one of (1)–(3) is a paper; (4) is the ambitious version.
- **Reproducibility:** release the benchmark + UnmixDB recipe + code; hand GT
  where licensing allows.

## 8. Risks & mitigations

- **GT scarcity** (2 hand sets) → synthetic pretrain (UnmixDB) + the labeling
  flywheel; treat hand GT as finetune/eval, not the main train set.
- **Separator noise asymmetry** (mix-side stems dirtier than ref) → measure it
  (clean-vs-clean control already in the notebook); prefer fingerprint/HuBERT
  over chroma where robust.
- **Overfitting to Big Bootie** → cross-set + UnmixDB held-out gate every claim.
- **Scope creep** (unmix/EQ is deep) → P4 is optional for the MVP paper; ship
  M0–M4 first.
- **"Not novel vs André 2024 / Schwarz-Fourer"** (the real risk — they own
  editable unmix) → defend on the unclaimed quadrant: **stem-identity axis +
  learned model + real scraped corpus + DAW round-trip for generation**. If we
  can't beat their UnmixDB MAE, pivot the contribution to stems + real-world +
  the generation application, not raw warp/gain accuracy.

## 9. Next two weeks (concrete)

1. **Get UnmixDB v1.1 + run our pipeline through it on the standard MAE metric** —
   this is the single most important move: it tells us exactly how far behind
   André 2024 we are, on the field's benchmark, in their units.
2. `eval_bench.py` over {UnmixDB, BB12, BB11}, 4 metrics + abstain curve.
3. Baselines: André-2024 NMF (target line) + Kim-style subseq-DTW + `landmark_fp`.
4. Oracle-ceiling table (oracle identity/grid/stem) → pick the binding constraint.
5. ✅ Done: prior-art folded into §1 (verified numbers + must-reads).

## Anchors (reuse, don't reinvent)

`path_decode.decode_path` (warp), `labeling/als_io` (round-trip),
`landmark_fp.py` (fingerprint), `external/unmixdb.py` (synthetic GT),
`similarity_probe.py` (HuBERT), `continuity_refine.py` (grid + cross-channel),
`mert_store.load_bb12_mert` (grids), `dtw_failure_eda.ipynb` (metrics + collapse
ladder), `docs/alignment_objective.md` + `docs/alignment_program_plan.md`
(existing specs).
