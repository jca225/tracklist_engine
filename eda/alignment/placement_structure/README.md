# placement_structure — placement objectivity, tempo drift, cue-point utility

EDA answering three questions (full write-up: [FINDINGS.md](FINDINGS.md)):

1. **Is optimal placement objective?** `placement_grid.py` — GT entries /
   exits / loop-jumps phased against the mix bar grid (circular statistics),
   acappella-vs-bed phrase lattice, ref-side snap, phase transfer.
2. **How non-constant is tempo?** `tempo_drift.py` — fold-corrected
   instantaneous BPM curves for reference tracks and mixes; separates true
   tempo drift from beat-tracker half/double-time instability; measures what
   an anchored constant-BPM grid costs by end of track.
3. **Is cue-detr useful?** `cue_eval.py` — stored cues vs downbeat grid
   (coherence) and vs GT entry points (Monte-Carlo lift over random).

Run from repo root; results print to stdout and persist to `out/*.json`:

```bash
PYTHONPATH=. venvs/audio/bin/python eda/alignment/placement_structure/placement_grid.py
PYTHONPATH=. venvs/audio/bin/python eda/alignment/placement_structure/tempo_drift.py
PYTHONPATH=. venvs/audio/bin/python eda/alignment/placement_structure/cue_eval.py
```

Inputs: `labeling/fixtures/*_ground_truth.yaml`, `data/analysis/*_measure_times.json`,
local dev DB (`track_analysis`, `canonical_track_cue_points`) — BB11-era
snapshot, so ref-side coverage is BB11-weighted. Shared grid arithmetic in
[gridmath.py](gridmath.py).
