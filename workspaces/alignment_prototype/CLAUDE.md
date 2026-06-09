# alignment_prototype — P5 span aligner (offline)

Incubates in `workspaces/` per [alignment_program_plan.md](../../docs/alignment_program_plan.md).
Promote to top-level `alignment/` when stable.

## Current scope

- Load exported `*_ground_truth.yaml` → `SpanTarget` rows
- Held-out split by base slot (`split.py`)
- Eval metrics + baselines (`eval.py`, `model.py`)
- `CopyGTBaseline` sanity model (loss should be 0 on eval)
- Huber placement + identity CE loss stubs

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
