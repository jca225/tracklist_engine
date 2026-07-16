# Neuro-inspired alignment plan

Status: DRAFT (2026-06-29). Three neuroscience-derived levers for the aligner's
known weak axes, scoped to **not collide** with the parallel aligner agent.

## Results & pivot (2026-06-29) — read this first

All work lives in `workspaces/alignment_prototype/neuro/` (no harness/infer edits).

- **WS1 (precision fusion): PARTIAL / mostly negative.** Curve-precision predicts
  correctness (AUC z=0.75 > raw-peak 0.64) and gives a clean monotonic ABSTENTION
  knob (the one keeper). But as an arbiter it does NOT beat single-channel: 2-ch
  ties hand-set hubert priority; 3-ch (with lyrics) is the WORST strategy (16.1s)
  while oracle is 4.6s. Channels are strongly complementary but unsupervised
  precision can't pick the winner per span → needs a LEARNED gate (C1/C2), blocked
  on cross-set GT. See [[project_ws1_precision_fusion]]. Files: `precision.py`,
  `precision_fusion_eval.py`, `precision_fusion_lyrics_eval.py`.
- **WS2 (phrase-entrainment): FALSIFIED.** GT segment boundaries are NOT
  phrase-locked (4/8/16-bar at parity with a random null). Don't build phrase-snap.
  File: `phrase_premise_check.py`.
- **Trajectory decomposition** (`path_decode --eval`, GT placement, fiber-aware):
  decode CEILING 59% ALL / 62% multiseg / acappella 33%(chroma) / oddratio 29%.
  Half the 26% headline loss is decode-bound, half placement.
- **PIVOT → Lever A (chosen by John): lyrics-anchor → segments.** Acappella
  placement is good but lyrics emits a SCALAR (prod traj-acc 3%); route
  `decode_path` on the lyrics-placed window to produce segments. File:
  `lyrics_segment_eval.py` (GT-placement ceiling vs lyrics-placement shippable).
- **WS3 (temporal-coherence binding): not started** — the last standing neuro
  idea, parked for instrumental + dense-overlap crosstalk.

## North star

Map auditory-cortex principles to the aligner's measured weaknesses. Each idea
ships first as a **standalone read-only eval probe** scored against BB12 GT, and
only wires into the default `infer.py` path *after* it wins **and** the parallel
agent's in-flight work (`infer.py`, `pretrain.py`, `synthetic_mix/`) has landed.

The two-streams framing (dorsal *where/when* vs ventral *what*) is the spine:
- never let a timing-invariant feature (MERT/HuBERT) vote on **when**;
- never let a content-agnostic feature (beat grid / novelty / fingerprint) vote
  on **what**.
This is the principle behind the [placement-wall decomposition error] finding.

## Isolation contract (non-negotiable)

The parallel agent is actively editing — per `git status` — `infer.py`,
`pretrain.py`, and `synthetic_mix/`. Therefore:

1. All new code lands in a **new** subdir: `workspaces/alignment_prototype/neuro/`.
2. **Do NOT edit** `infer.py`, `pretrain.py`, `synthetic_mix/`, the notebook, or
   (to be safe) any currently-modified file. Compose, don't mutate.
3. New probes conform to `harness/contract.Probe` but are first exercised through
   **own eval scripts** under `neuro/`, not by registering into
   `harness/driver.py` or editing `harness/merge.py` / `harness/axes.py`.
4. Read-only imports only from existing harness modules.
5. Namespaced outputs: write to `neuro/out/` and `.cache/neuro/`, never the
   shared `out/` or `.cache/mert/`.
6. `git pull` + re-scan the parallel agent's diff before each session; never
   revert workspace files. (See [parallel-aligner-agent] memory.)

Integration (registering a probe in `driver.py`, extending the `AXES` dict, or
swapping the `merge()` arbiter) is a **separate, later** step taken only once the
other agent has merged and a probe has cleared its win bar.

## Harness seams (from read-only map, 2026-06-29)

- Probe interface: `harness/contract.Probe.run(MixContext, RefContext,
  CandidatePool) -> AlignmentResult`.
- Arbiter: `harness/merge.merge(results, *, offset_tol_s, min_confidence,
  agreement_bonus, source_priority) -> AlignmentResult` (picks earliest-priority
  non-abstaining; agreement bonus on matching `recording_id`+offset).
- Routing: `harness/axes.AXES[stem] = AxisRoute(mix_file, ref_stem,
  invariant_feature, placement_probes)`.
- Acappella placement: `stem_placement.place_joint(mix_feat, ref_feat, prior_ss,
  span_dur, *, band_s, w_s, hop_s, stretches, tol_s, gap_s) -> (set_start_s,
  ref_start_s, peak)`.
- Landmark placement: `mix_fp_hits.span_from_offset_votes(...) -> (set_start_s,
  set_end_s, votes, offset_s)`; `mix_fp_hits.offset_candidates(..., topk=6)`.
- Features: `continuity_refine._compute_hubert(path, layer) -> (768, T)` (cached);
  `refine_ref_offsets.chroma(y)`; `landmark_fp.fingerprint_from_audio(y)`;
  `mert_store.MertSeries.pool(start_s, end_s)` (beat-grid aligned).
- Beat grid: `set_measures` / `track_measures` tables (mix & ref downbeats);
  MertSeries is already measure-aligned → phrase indices available.
- Per-axis eval (read-only, score vs GT):
  - `python -m workspaces.alignment_prototype.acappella_ref_offset_eval --workers 8`  (n=24 acappella)
  - `python -m workspaces.alignment_prototype.instrumental_ref_offset_eval --workers 8`  (n=5)
  - `python -m workspaces.alignment_prototype.eval_placement [--feature chroma|fp]`  (BB12 regular)
  - `python -m workspaces.alignment_prototype.eval_ref_detection --eval [--stems acappella] [--feature hubert]`

---

## WS1 — Reliability-weighted fusion arbiter  (cheapest, highest certainty)

**Neuro principle.** Optimal multisensory cue integration (Ernst & Banks):
the brain fuses cues weighted by **inverse variance** (precision), and collapses
to abstention when all cues are unreliable. Raw response magnitude is *not* the
weight.

**Weakness it attacks.** [fusion-needs-axis-prior] (merge-by-raw-peak is unsound
cross-axis; chroma is high-everywhere) and [abstention-via-margin] (margin, not
absolute cosine, is the confidence signal). Current `merge()` uses a fixed
`source_priority` ordering + a flat confidence scalar.

**Method.** Replace the fixed priority/confidence with a per-result **precision**
estimate computed from each probe's own score curve:
- precision proxy = peak **margin** (top offset peak − runner-up) / local
  background std, on the probe's matched-filter / vote-histogram output;
- fuse candidates as a precision-weighted vote in (recording_id, offset) space;
- abstain when total precision < floor (all channels flat).

**Build.**
- `neuro/precision.py` — `precision_from_curve(scores) -> float` and
  `precision_merge(results_with_curves, ...) -> AlignmentResult` (drop-in shape
  of `harness/merge.merge`, never edits it).
- `neuro/precision_fusion_eval.py` — read-only: import `continuity_refine`,
  `stem_placement`, `chroma`/`fp` probes to recover per-channel score curves on
  BB12, run both the stock `merge()` and `precision_merge()`, print deltas.

**Eval.** BB12 all axes via the per-axis eval scripts; headline = set_start
median/p90 + identity acc + **abstention precision/recall** (does it correctly
decline the frames the stock merge gets wrong?).

**Win bar.** ≥ stock merge on set_start median *and* strictly better abstention
precision (fewer confident-wrong) at equal coverage. **Kill** if precision proxy
doesn't separate right-from-wrong frames (AUC ≤ 0.6).

**Cost/risk.** Pure post-processing over existing curves — smallest collision
surface, no new feature extraction. Main risk: recovering clean per-channel
curves without touching probe internals; mitigate by adding optional
curve-return wrappers in `neuro/`, not in the harness probes.

---

## WS2 — Phase-aware entrainment decode  (acappella ref_start)

**Neuro principle.** Dynamic Attending Theory / oscillatory entrainment (Large,
Jones): cortical oscillators phase-lock to metrical structure and sample
attention at **expected phrase phases**, not continuously. This is the *why*
behind the [collapse-ladder] finding that **grid-lock is THE placement lever**
(24.4 → 7.9s). The unexploited part is **phrase phase**, not just the grid.

**Weakness it attacks.** Acappella **ref_start** is the worst, un-owned lane
([bb12-per-axis-baseline]): repeat-ambiguity (repeated choruses) defeats argmax;
[continuity-stack is a no-op] because 93% of acappella spans are non-linear.
DSP can't fix it; phase can constrain it.

**Method.** Build a phrase-phase grid from `set_measures` (mix) and
`track_measures` (ref): downbeat → 4/8/16-bar phrase indices. Then:
- restrict `place_joint` candidate set_start/ref_start to metrical phases
  (snap-to-phrase), shrinking the search space;
- among repeat-ambiguous ref_start candidates (near-equal HuBERT peaks), break
  ties by **phrase-phase agreement** between mix-side and ref-side phrase index.

**Build.**
- `neuro/phase_grid.py` — `phrase_grid(measures, bars=(4,8,16)) -> grid`;
  `snap_to_phase(t, grid)`.
- `neuro/phase_entrain_eval.py` — wrap `stem_placement.place_joint` with
  phase-snapped candidates (no edit to `stem_placement.py`; pass candidates in or
  re-rank its `(set_start, ref_start, peak)` output), score on BB12 acappella.

**Eval.** `acappella_ref_offset_eval` (n=24) — ref_start error median/p90 and
**repeat-ambiguity resolution rate** (fraction of multi-chorus spans where the
correct chorus is chosen).

**Win bar.** Lift ref_start <15s rate vs the [per-stem HuBERT set_start] baseline
without regressing set_start. **Kill** if beat grids are too noisy on
vocals-only mixes to define phrases (check `set_measures` coverage on acappella
mix stems first).

**Cost/risk.** Medium. Depends on beat-grid quality on vocal stems — verify
before building. Honest caveat: phase constrains *placement*, it does not
identify the right repeat if the artist literally repeats a chorus verbatim at
the same phrase phase — those stay ambiguous and should abstain (feeds WS1).

---

## WS3 — Temporal-coherence stream binding  (identity under crosstalk)

**Neuro principle.** Auditory Scene Analysis (Bregman) + temporal coherence
(Shamma): the cortex segregates a mixture **without** source separation by
binding feature channels that **co-modulate over ~200ms** into one stream
(plus harmonicity / common-onset grouping).

**Weakness it attacks.** [stem-match-bootstrap] — identity under crosstalk is the
real problem (mixes 2–3 layers deep; id 0–14% raw → 84% with stem routing).
Current fix is full Roformer separation: lossy, offline, and a heavy dependency.

**Method.** Compute a cochleagram (gammatone / mel) envelope per channel; build a
**coherence matrix** = pairwise envelope correlation over a sliding ~200ms
window; cluster co-modulating channels into streams; use each stream's soft
spectral mask as a **pre-identity cue** feeding the MERT/HuBERT identity pathway
— compared against Roformer-separated and raw-mix baselines.

**Build.**
- `neuro/coherence_bind.py` — `cochleagram(y)`, `coherence_matrix(env, win_s=0.2)`,
  `bind_streams(C) -> masks`.
- `neuro/coherence_id_eval.py` — feed bound-stream masks into the existing
  identity matcher (`stem_match_probe` path, read-only) and compare id-acc:
  coherence-bind vs Roformer-separated vs raw mix.

**Eval.** Identity accuracy under crosstalk on BB12 overlap spans (the
stem_match litmus). Target: close the gap toward the 84% Roformer-routed number
**without** running separation.

**Win bar.** Beat raw-mix id-acc by a wide margin and reach within ~10pts of
Roformer at a fraction of the compute. **Kill** if binding can't beat raw mix —
ASA without harmonicity grouping may be too weak for dense EDM; if so, document
and stop (don't chase a learned separator, that's ingest's lane).

**Cost/risk.** Highest. This is genuine research with real odds of failure;
sequence it last and timebox it.

---

## Sequencing & shared rules

1. **WS1** first — days, pure post-proc, unblocks principled abstention that the
   other two feed into.
2. **WS2** second — gated on a 30-min beat-grid-coverage check on vocal stems.
3. **WS3** last — timeboxed research spike.

Shared:
- Every probe eventually conforms to `harness/contract.Probe`, but lands as a
  standalone `neuro/*_eval.py` first.
- Report deltas vs the documented per-axis baselines ([bb12-per-axis-baseline]),
  not absolute numbers.
- No silent caps: if a probe drops spans (coverage), log how many and why.
- Small-n honesty: instrumental n=5, acappella n=24 — treat single-set deltas as
  directional, confirm on BB11 before any wiring claim ([small-sample-regressions]).

## Open questions before coding

- WS1: can per-channel score **curves** be recovered cleanly without editing the
  harness probes, or do we need thin curve-returning wrappers under `neuro/`?
- WS2: is `set_measures` populated/reliable for **vocals-only** mix stems? (Verify
  on pi-storage first.)
- WS3: cochleagram backend — reuse librosa mel, or add a gammatone filterbank?
