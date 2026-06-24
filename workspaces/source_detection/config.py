"""Tunable constants and the run Config. Keep DSP magic numbers here, not
scattered across modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --- DSP (matches alignment_prototype/refine_ref_offsets.py so chroma is
#     directly comparable to the proven matched-filter) ---
SR = 22050
HOP = 512
FRAME_S = HOP / SR  # ~0.0232 s per chroma frame

# --- matcher ---
N_ROTATIONS = 12                     # circular chroma shifts == semitone search
TEMPLATE_S = 12.0                    # length of each source template window
TEMPLATE_STRIDE_S = 8.0              # hop between templates taken across a source
# stretch = mix_local_bpm / source_bpm. We search a tight multiplicative band
# around the BPM-derived estimate (NOT a blind seconds grid — that saturated at
# its edges in the prototype). When BPM is unknown we fall back to FALLBACK_STRETCHES.
STRETCH_BAND = (0.90, 1.11)          # clamp for the BPM-derived factor
STRETCH_STEPS = 5                    # samples across the band around the estimate
FALLBACK_STRETCHES = (0.92, 0.96, 1.0, 1.04, 1.08)
PEAK_MIN_SCORE = 0.55                # normalized matched-filter score floor for a raw hit
PEAK_MIN_DISTANCE_S = 4.0            # min spacing between peaks of one template

# --- postprocess ---
MERGE_GAP_S = 6.0                    # same-song hits this close (and same channel) merge
NMS_IOU = 0.5                        # overlapping conflicting detections suppressed above this
CONF_THRESHOLD = 0.60               # final confidence cut for reported detections

# --- prefilter ---
MERT_TOPK = 40                       # keep this many candidate sources per mix (None = all)
PREFILTER_SKIP_BELOW = 25            # skip MERT prefilter when n_sources <= this

# --- eval ---
EVAL_TOLERANCE_S = 5.0

REPO_ROOT = Path(__file__).resolve().parents[2]
ALIGNING_ROOT = Path.home() / "aligning"
CACHE_ROOT = Path(__file__).resolve().parent / ".cache"
OUT_ROOT = Path(__file__).resolve().parent / "out"


@dataclass
class Config:
    """Per-run knobs; defaults pull from the module constants above."""

    sr: int = SR
    hop: int = HOP
    n_rotations: int = N_ROTATIONS
    template_s: float = TEMPLATE_S
    template_stride_s: float = TEMPLATE_STRIDE_S
    peak_min_score: float = PEAK_MIN_SCORE
    conf_threshold: float = CONF_THRESHOLD
    mert_topk: int | None = MERT_TOPK
    eval_tolerance_s: float = EVAL_TOLERANCE_S
    workers: int = 6
    cache_root: Path = field(default_factory=lambda: CACHE_ROOT)
    out_root: Path = field(default_factory=lambda: OUT_ROOT)
