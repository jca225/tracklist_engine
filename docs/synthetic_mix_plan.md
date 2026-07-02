# Synthetic-mix bootstrap plan (2026-06-25)

Low-opportunity-cost bet: if realistic synthetic pretrain transfers to real GT, we
shrink the labeling bottleneck dramatically. If not, we learn the distribution gap
and can stop wondering.

## Hypothesis

Train `MertAlignHead` on **realistic** stem mashups with perfect GT labels, then
finetune on a small real held-out set (BB12/BB11). Success = pretrain→finetune beats
finetune-only on identity + placement (ablation already wired in `pretrain.py`).

## What makes mixes "realistic" (not random stacks)

| Property | Random (stress-test) | Realistic (training) |
|----------|---------------------|----------------------|
| Structure | K random stems summed | 1 host bed + 1–2 acap overlays |
| Tempo | unmatched | overlay beatmatched to host BPM |
| Key | raw | pitch-shifted to compatible key |
| Overlap | arbitrary shifts | phrase-length span, edge fades |
| Depth | up to 6 | 1–2 (HuBERT strong zone) |

Random permutation remains useful as a **robustness probe** (`stem_match_probe`); it
is not the training distribution.

## Phased execution

### Phase 1 — Generator (this session)

**Goal:** produce ~10–500 labeled synthetic mixes + ear-check pass.

**Module:** `workspaces/alignment_prototype/synthetic_mix/`

| File | Role |
|------|------|
| `catalog.py` | Local stem inventory + recording_id + key/BPM from pi |
| `scenario.py` | Sample host bed + 1–2 payloads with compatibility filter |
| `render.py` | Beatmatch, pitch-shift, fade, composite audio |
| `labels.py` | Emit `GroundTruthSet` YAML (same schema as real GT) |
| `generate.py` | CLI: `--n`, `--curriculum`, `--out`, `--seed` |

**Output layout** (per mix):
```
data/synthetic_mixes/synth_NNNN/
  mix.flac                 # bed + overlays
  mix_vocals.flac          # vocal channel
  mix_instrumental.flac    # bed only
  ground_truth.yaml
  refs/<recording_id>_{vocals,instrumental}.flac
```

**Curriculum levels:**
- `easy` — 1 overlay, key_dist=0, bpm_fold ≤ 0.02
- `medium` — 1 overlay, key_dist ≤ 1, bpm_fold ≤ 0.05
- `hard` — 2 overlays, key_dist ≤ 2, bpm_fold ≤ 0.08

**Gate:** listen to 5 mixes — do they sound like plausible DJ mashups?

### Phase 2 — Pretrain on synthetic corpus

**Goal:** train head on synthetic, ablate vs finetune-only on real BB12 held-out.

```bash
# Generate corpus (start small)
venvs/audio/bin/python -m workspaces.alignment_prototype.synthetic_mix.generate \
  --n 100 --curriculum medium --out data/synthetic_mixes

# Pretrain (chroma = fast smoke; mert = real run — required for BB12 ablation)
venvs/audio/bin/python -m workspaces.alignment_prototype.pretrain \
  --synthetic-root data/synthetic_mixes --features mert --max-mixes 100 \
  --out workspaces/alignment_prototype/.cache/pretrain_synthetic_mert.pt

# Decisive ablation
venvs/audio/bin/python -m workspaces.alignment_prototype.pretrain --ablation \
  --pretrain-checkpoint workspaces/alignment_prototype/.cache/pretrain_synthetic_mert.pt
```

**Success criterion:** pretrain→finetune **identity_acc** and **MAE set_start** beat
finetune-only by a meaningful margin on BB12 held-out (≥5% identity or ≥2s placement).

### Phase 3 — Scale + MERT + cross-set

- Scale to 500–2000 mixes (549 pi stem sets + Discord corpus)
- MERT pretrain (GPU, slower but matches production features)
- Validate on BB11 held-out (cross-set transfer)
- Compare vs UnmixDB pretrain ablation (same harness)

## What this does NOT solve

1. **Corpus-scale open-set identity** — fingerprint + ANN index still needed at 20k scale
2. **Real GT elimination** — still need held-out real sets for finetune + proof
3. **Separation artifacts** — synthetic uses clean stems; real arm gap remains

## Raw material

- **Local cache:** 57 instrumental beds + 82 vocal payloads (137 taid dirs; beds and payloads are different tracks)
- 549 pi sets with `{vocals,instrumental}.flac` (scale target)
- ~2,474 Discord stem corpus (pi staging)
- Grids: `track_measures` / `set_measures` on pi
- Key/BPM: `track_audio_features` via pi (Essentia)

## Current status (2026-06-25)

**Phase 1 DONE** — generator, 100-mix corpus, Vast MERT pretrain, BB12 ablation.

**Result: flat** — pretrain→finetune did not beat finetune-only (identity +0.0%, MAE
set_start +0.056s). Cause: synthetic topology (1 bed + 1 acap / 90 s) ≠ BB12 (166 spans,
loops, ref_segments, handoffs). See **`docs/synthetic_mix_plan_v2_bb12.md`** for the
realistic generator plan.

Pulled artifacts:
- `workspaces/alignment_prototype/.cache/pretrain_synthetic_mert.pt`
- `data/synthetic_mixes/synth_pretrain.log`

## Kill criteria (when to stop investing)

- Phase 1 ear-check: mixes sound absurd → fix renderer before training
- Phase 2 ablation: pretrain→finetune ≤ finetune-only after 500 mixes → distribution gap too large; document and pivot
- Phase 2 ablation: positive but <2% lift → marginal; only continue if labeling cost savings justify it

## Related docs

- `docs/agent_handoff_stem_bootstrap_20260625.md` — probe numbers + context
- `docs/alignment_objective.md` — A/B/C standing
- `docs/alignment_program_plan.md` — tiered label schema
