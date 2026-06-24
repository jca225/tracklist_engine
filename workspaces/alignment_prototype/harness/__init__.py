"""Driver-agnostic alignment harness.

One contract — AlignmentResult + the Probe interface — that an LLM agent, a
deterministic DSP pipeline, or the trained ML aligner all target against the same
ground truth and the same eval. This package holds the contract; probe adapters,
confidence calibration, and the joint eval build on top of it.
"""

from .contract import (
    AlignmentResult,
    CandidatePool,
    MixContext,
    Probe,
    RefContext,
    RefSegment,
)
from .merge import merge

__all__ = [
    "AlignmentResult",
    "CandidatePool",
    "MixContext",
    "Probe",
    "RefContext",
    "RefSegment",
    "merge",
]
