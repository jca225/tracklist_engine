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

**Measured limitation:** pooled-MERT cosine does not *localize* content in
the mix — with the oracle ref segment, the unconstrained argmax is ~900 s
off at every layer (0–24), raw or learned, centered or whitened. The 39 s
placement comes from the monotonic tiling prior, not audio matching.
Sub-bar placement needs a different emission signal (stem-aware chroma /
DTW or stretch-tolerant fingerprinting — `set_fingerprint_hits` exists but
is empty corpus-wide), not a better MERT head.

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

## Not wired yet

- PyTorch training loop beyond the small `MertAlignHead` (`--train-mert`)
- Learned weighted-sum over all MERT layers (still layer-6 probe)
- Full-corpus ref candidate pool (slot pool = GT-distinct ids per slot today)

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
