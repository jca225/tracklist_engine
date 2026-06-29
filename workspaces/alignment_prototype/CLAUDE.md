# alignment_prototype — P5 span aligner (offline)

Incubates in `workspaces/` per [alignment_program_plan.md](../../docs/alignment_program_plan.md).
Promote to top-level `alignment/` when stable.

## Current scope

- Load exported `*_ground_truth.yaml` → `SpanTarget` rows
- Held-out split by base slot (`split.py`)
- Eval metrics + baselines (`eval.py`, `model.py`)
- `CopyGTBaseline` sanity model (loss should be 0 on eval)
- Huber placement + identity CE loss stubs
- `MertAlignHead` seed ensemble (`TrainConfig.n_heads`) + joint slot decoding:
  identity = max over (mix window, ref window) pairs in the search band; a
  slot's k spans assign to top-k candidates, and `predict_sequence` then
  re-scores each multi-span slot's span→candidate assignment by global
  decode total (`_sweep_slot_assignments`) — the anchor-band match-time
  ordering swapped slots 058/059 (spans minutes apart, band covers neither)

**BB12 held-out eval (2026-06-09):** identity 100% (30/30), ref_start MAE
0.84 s, set placement MAE 39 s (median 37 s, p90 78 s) via
`predict_sequence` — a whole-mix monotonic DP (`sequence_decode.py`)
replacing the per-slot anchor band. Candidates without MERT embeddings are
logged loudly, never silently zero-filled (that hid the slot-039 miss).
All-span identity (train+eval) is 147/147 after the assignment sweep
(2026-06-10) and printed by `train.py --eval --train-mert` as a `MISS`
report — watch it on new sets; within-slot swaps don't show in held-out
metrics.

> **⚠ Label corruption (fixed 2026-06-11, commit a450005):** the GT export
> read the warp anchor instead of the clip trim, so most ref_start labels
> were ≈0 — the 0.84 s ref MAE above was self-consistent-but-wrong, and any
> head trained before the bb12_ground_truth.yaml regeneration learned to
> predict ~0 ref offsets. **Retrain on the regenerated yaml** before
> trusting decode ref offsets. Set placement + identity metrics unaffected.
> Detector eval vs corrected GT (`eval_ref_detection.py`, n=20 straight
> clips): regular 42% exact <2 s / 67% repeat-equivalent, acappella 0%/14%
> (vocal chroma weak — needs a vocal-specific signal), stretch err 1.2%
> median (grid heuristic validated). Loops/segments (83/166 GT rows!) are
> outside linear scoring — span output must become segment lists.
>
> End-to-end pipeline on BB12 vs corrected GT (`score_timeline_vs_gt.py`,
> retrained head, in-domain upper bound, 2026-06-11): identity 66% (time-
> overlap matching — placement error >5 s can mark correct ids wrong),
> set placement median 30 s / p90 76 s (cue-anchored), ref offsets median
> 50 s on straight clips (repeat ambiguity dominates; needs continuity
> decode). The mix-side timebase bug (c43fa62) is fixed in this number.

**Measured limitation (MERT):** pooled-MERT cosine does not *localize* content
in the mix — with the oracle ref segment, the unconstrained argmax is ~900 s
off at every layer (0–24). MERT is identity, not placement. (The old ~39 s
"placement wall" was read off the monotonic tiling prior.)

**PLACEMENT REFRAME (2026-06-28) — the ~30 s "wall" was a decomposition error.**
The landmark fingerprint localizes the mix↔ref *alignment diagonal* to **0.2 s
median / 76 %** (BB12 regular). set_start looked stuck at ~30 s only because we
measured it as the alignment offset, but DJs start tracks mid-song (GT ref_start
median ~56 s), so **set_start = ref_start + d**. The fingerprint's own
vote-density extent along the diagonal gives the span directly:
`mix_fp_hits.span_from_offset_votes` (single ref, set_start median 5.7 s) →
`offset_candidates` (top-K diagonals) → `decode_placements` (monotonic decode
over tracklist order, rejects out-of-order wrong-diagonal/repeat picks): **BB12
regular set_start median 4.1 s, <15 s 73 %.** Run: `python -m
workspaces.alignment_prototype.eval_placement`. Requires the fingerprint backfill
(`scripts/backfill_track_fingerprints.py`, done corpus-wide). The ~27 % outliers
are weak-fp / repeat spans (true diagonal absent from top-K) → per-stem + fibers;
cheap post-hoc fixes (cluster-strength, isotonic, boundary-snap) were all measured
and REJECTED — see the `project_placement_wall_was_decomposition_error` memory.

**Axis decomposition (the unifying principle).** song ≈ timbre × harmony ×
language, near-orthogonal: timbre=MERT (identity only), harmony=chroma,
language=HuBERT. Match/fiber on the NUISANCE-INVARIANT axis per stem — vocals →
HuBERT ("lyrics don't transpose"; key-invariant, beats chroma on acappella
ref-offset 2.1 s vs 39.6 s median), instrumental → chroma+fingerprint. `harness/
axes.py` routes stem → (mix_file, ref_stem, invariant_feature, placement_probes,
in priority order). Key changes break chroma (31 % of BB11 acappellas re-pitched;
transposition search adds spurious peaks) — HuBERT sidesteps it. Fusion must use
the axis prior, not raw cross-probe confidence (`harness/merge.py` source_priority).

**Design decision (2026-06-11) — stem-wise alignment.** A mix moment is a
sum of layers (host instrumental + overlaid acappellas), so full-mix-only
matching entangles them and matches no single ref — the root cause of the
localization failure above, on BB12 too. Alignment is computed per stem
channel (mix_vocals↔ref vocals stem, mix_instrumental↔ref instrumental
stem) AND on the full mix, as separate channels fused at decode. Division
of labor: identity = MERT head (100% BB12 held-out); ref-offset placement =
matched-filter correlation (`refine_ref_offsets.py` — BB11 151/151
relocated, median move 100 s vs decode, peak median 0.83); tempo = the
instrumental-BPM-anchor heuristic (host grid never changes within a span;
acappellas beat-synced to it ⇒ stretch = ref_bpm / mix_local_bpm from
set_measures × track_measures, search in beat space with bar-quantized
offsets — v1's seconds-space stretch grid saturated at its 0.92/1.08 edges).

## Harness + new modules (2026-06-28)

The unified-aligner consolidation (plan: `.claude/plans/and-then-can-we-cuddly-sparrow.md`):
- `harness/` Probe/AlignmentResult/DeterministicDriver contract; probes:
  `chroma_probe`, `fingerprint_probe`, `path_decode_probe`, **`hubert_probe`**
  (vocal/language axis), **`continuity_probe`** (repeat-robust stack). `axes.py`
  = the stem→axis routing. `merge.py` `source_priority` = axis-priority arbitration.
- `ref_fibers.compute_fibers_soft` (μ membership + per-fiber confidence) +
  `fiber_ambiguity` (instance-abstain signal). Fibers are HuBERT+silence-gate, never chroma.
- `mix_fp_hits.{span_from_offset_votes, offset_candidates, decode_placements}` =
  the placement pipeline; `eval_placement.py` runs it vs GT.

## Not wired yet

- `decode_placements` is a standalone module — NOT yet called by `infer.py` (the
  cross-set inference still uses `predict_sequence`). Wiring it in is the next integration.
- Per-stem placement: only `stem='regular'` fingerprints are backfilled; acappella/
  instrumental set_start needs their stem fingerprints (or HuBERT) backfilled.
- B3 live decode: `fiber_ambiguity`/μ computed but not yet fed into the live decode.
- Learned fusion arbiter (C1/C2): probe-feature extractor + small head over
  {axis scores, fiber conf, fp sharpness} — needs more GT (leave-one-set-out CV).
- PyTorch training loop beyond the small `MertAlignHead`; learned weighted-sum over MERT layers.

## Commands

```bash
venvs/audio/bin/python -m workspaces.alignment_prototype.train --dry-run
venvs/audio/bin/python -m workspaces.alignment_prototype.train --eval
venvs/audio/bin/python -m workspaces.alignment_prototype.train --eval --train-mert
# cross-set inference (train on BB12 GT, predict target set)
venvs/audio/bin/python -m workspaces.alignment_prototype.infer --set-id 2nvzlh2k --band-s 45
# human verification of a predicted timeline (after infer):
venvs/audio/bin/python -m workspaces.alignment_prototype.render_review_snippets --set-id 2nvzlh2k
venvs/audio/bin/python -m workspaces.alignment_prototype.seed_als_from_timeline --set-id 2nvzlh2k
```

## Human review loop (predictions → GT)

1. `infer` writes `out/<set_id>_predicted_timeline.json`.
2. `render_review_snippets` renders per-span A/B clips (mix at predicted
   position vs ref at predicted offset; acappella/instrumental spans use the
   Demucs stem) + `out/review/<set_id>/review.html` — keyboard verdicts,
   worst-suspicion-first. Listening pass ≈ 30–40 min for ~150 spans.
3. `seed_als_from_timeline` writes a pre-seeded Live project to the Desktop
   (60 BPM = 1 beat/s; clips warped onto predicted ref segments; color =
   suspicion). Self-validates by round-tripping its own output through
   `labeling/als_io` (placement, identity, stem). Human fixes failures only.
4. The corrected `.als` exports via `labeling/export_als_to_gt.py` → the
   target set becomes new GT; diff vs the predicted timeline = honest
   placement/identity scorecard. Requires the set pulled via
   `labeling/pull_set_for_alignment.py` (slot-spine fix ed7f121 — older
   pulls silently dropped Rvmor-gap slots).

## UnmixDB pretrain (external)

Download [UnmixDB](https://zenodo.org/records/1422385), then:

```bash
# label parse smoke (no audio)
venvs/audio/bin/python -m workspaces.alignment_prototype.pretrain --dry-run \\
  --unmixdb-root ~/data/unmixdb-v1.1

# chroma pretrain (pipeline validation; dim=12, no BB12 weight transfer)
venvs/audio/bin/python -m workspaces.alignment_prototype.pretrain \\
  --unmixdb-root ~/data/unmixdb-v1.1 --features chroma --max-mixes 50

# MERT pretrain → BB12 ablation (use --features mert for weight transfer)
venvs/audio/bin/python -m workspaces.alignment_prototype.pretrain --ablation \\
  --pretrain-checkpoint workspaces/alignment_prototype/.cache/pretrain_mert.pt
```

Loader: `external/unmixdb.py`. Features: `external/feature_series.py` (chroma
1 s bins or cached MERT 2 s bins). Checkpoints: `external/checkpoint.py`.

## Landmark fingerprint index

Backfill reference rows, then `refine_ref_offsets` reads the local cache:

```bash
venvs/audio/bin/python scripts/backfill_track_fingerprints.py --dry-run
venvs/audio/bin/python scripts/backfill_track_fingerprints.py --limit 100
venvs/audio/bin/python -m workspaces.alignment_prototype.refine_ref_offsets \\
  --set-id 2nvzlh2k
```

Writes `track_fingerprints` (kind=landmark JSON) + `.cache/fp_index/`.
Spans with weak chroma **and** weak fingerprint get `abstain_ref_offset: true`
on the timeline JSON. This improves **ref_offset** recovery and safety; it does
**not** fix the ~30–39 s **set_start** placement bottleneck by itself.

## Mix fingerprint hits + placement refine

Scan the mix per slot (cue ± band) into `set_fingerprint_hits`, then optional
sharpness-gated per-span argmax on coarse decode:

```bash
venvs/audio/bin/python scripts/cache_set_fingerprint_hits.py --set-id 1fsnxchk --dry-run
venvs/audio/bin/python scripts/cache_set_fingerprint_hits.py --set-id 1fsnxchk
venvs/audio/bin/python -m workspaces.alignment_prototype.infer \\
  --set-id 2nvzlh2k --fp-refine --fp-band-s 45 --fp-gate-z 1.0
```

Module: `mix_fp_hits.py` (scan + placement curves), `fp_placement_refine.py`
(per-span override — **not** joint re-decode; see `docs/fine_placement_plan.md`).
