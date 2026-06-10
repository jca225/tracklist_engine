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

**Measured limitation:** pooled-MERT cosine does not *localize* content in
the mix — with the oracle ref segment, the unconstrained argmax is ~900 s
off at every layer (0–24), raw or learned, centered or whitened. The 39 s
placement comes from the monotonic tiling prior, not audio matching.
Sub-bar placement needs a different emission signal (stem-aware chroma /
DTW or stretch-tolerant fingerprinting — `set_fingerprint_hits` exists but
is empty corpus-wide), not a better MERT head.

## Not wired yet

- PyTorch training loop beyond the small `MertAlignHead` (`--train-mert`)
- Learned weighted-sum over all MERT layers (still layer-6 probe)
- Full-corpus ref candidate pool (slot pool = GT-distinct ids per slot today)

## Commands

```bash
venvs/audio/bin/python -m workspaces.alignment_prototype.train --dry-run
venvs/audio/bin/python -m workspaces.alignment_prototype.train --eval
venvs/audio/bin/python -m workspaces.alignment_prototype.train --eval --train-mert
```
