"""P5 aligner prototype — supervised span alignment (offline dev).

Inputs (future): mix features A, tokenized tracklist T, ref embeddings {E(x)}.
Output: span-level L matching ``GroundTruthTrack``.

This workspace is intentionally pi-independent: train/dev reads exported YAML +
local ``~/aligning/`` audio until MERT backfill (P4) lands on pi-storage.

Run:
    venvs/audio/bin/python -m workspaces.alignment_prototype.train \\
        --yaml labeling/fixtures/bb12_ground_truth.yaml --dry-run
"""
from __future__ import annotations
