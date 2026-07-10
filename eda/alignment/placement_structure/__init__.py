"""Placement & structure EDA: is optimal placement objective? How non-constant
is tempo? Are cue-detr cues empirically useful?

Three scripts, run from repo root (see README.md):
  placement_grid.py — GT entries/exits/loop-jumps vs the mix bar grid
  tempo_drift.py    — within-track instantaneous BPM vs the global-BPM assumption
  cue_eval.py       — cue-detr cues vs downbeats and GT entry points
"""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Checkout root by marker walk — no parents[N] depth coupling."""
    for p in Path(__file__).resolve().parents:
        if (p / "Makefile").exists():
            return p
    raise RuntimeError(f"repo root not found above {__file__}")
