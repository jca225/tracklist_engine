# eda/alignment/ — mix structure analysis

Exploratory probes on **mix-side MERT** — sections, events, and boundary detection.
This is the **data-analysis phase** described in
[docs/aligner_attention_design.md](../../docs/aligner_attention_design.md). It is
**not** the algorithmic aligner (`workspaces/alignment_prototype/`, future `alignment/`).

## Modules

| Module | Role |
|--------|------|
| `adaptive_markov.py` | Information-dynamics engine (Abdallah & Plumbley 2008 style) |
| `mert_vectors.py` | Decode per-measure MERT blobs; layer-6 probe vector |
| `tokenize.py` | VQ/k-means bar tokens |
| `boundaries.py` | GT boundary extraction, peak picking, scoring |
| `artifacts.py` | `.npz` cache format for per-bar mix MERT |
| `mix_structure_probe.py` | CLI entry point |

## Phase 0 — build a mix MERT artifact

The probe needs a bar-synchronous mix embedding file. BB12 (`set_id=1fsnxchk`) still
needs mix beat grid + MERT on canonical pi-storage (see
[bb12_information_dynamics_plan.md](../../docs/bb12_information_dynamics_plan.md)).

On Mac (mix audio already local or rsynced from pi-storage):

```bash
# 1) Beat grid — set_audio_id for BB12 is typically 5 on pi-storage
venvs/audio/bin/python -m analysis.set_analysis <set_audio_path>  # or beat_this via mac_analyze_sets

# 2) MERT per measure → save artifact
venvs/audio/bin/python -m eda.alignment.prepare_mix_artifact \
  --set-id 1fsnxchk \
  --audio /path/to/mix.m4a \
  --measure-times-json /path/to/measure_times.json \
  --out data/analysis/1fsnxchk_mix_mert.npz
```

Until that artifact exists, run the synthetic self-test:

```bash
venvs/audio/bin/python -m eda.alignment.mix_structure_probe --synthetic
```

## Phase 1+ — probe

```bash
venvs/audio/bin/python -m eda.alignment.mix_structure_probe \
  --artifact data/analysis/1fsnxchk_mix_mert.npz \
  --gt labeling/fixtures/bb12_ground_truth.yaml \
  --n-tokens 16 \
  --out data/analysis/1fsnxchk_structure_probe.json
```

Results also append to `data/analysis/aux.db` `analysis_results` when `--persist` is set.

Findings write-up: [findings.md](findings.md) (populated after first real run).
