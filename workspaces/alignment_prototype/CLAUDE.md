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
  slot's k spans assign to top-k candidates ordered by matched mix time

**BB12 held-out eval (2026-06-09):** identity 100% (30/30), ref_start MAE
0.79 s, set placement MAE ~171 s — placement is the open front (needs
sequence structure / boundary-proposer constraints, not similarity quality).
Candidates without MERT embeddings are logged loudly, never silently
zero-filled (that hid the slot-039 miss).

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
