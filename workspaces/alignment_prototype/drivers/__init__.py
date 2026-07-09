"""End-to-end aligner drivers behind one interface, raced on one scorecard.

Three interchangeable drivers, each emitting the same PredictedTimeline JSON that
`score_timeline_vs_gt` judges:

  * ClassicalDriver  — shipped infer -> joint_ref_decode (the baseline)
  * AgenticDriver    — POMDP belief/ladder refines placement over the base
  * HybridMlDriver   — learned TrajectoryDecoder owns the ref-segment stage

See `base.EndToEndDriver` for the contract and `race` for the runner.
"""

from __future__ import annotations

from .agentic import AgenticDriver
from .base import EndToEndDriver, SetContext
from .classical import ClassicalDriver
from .ml import HybridMlDriver

DRIVERS = {
    ClassicalDriver.name: ClassicalDriver,
    AgenticDriver.name: AgenticDriver,
    HybridMlDriver.name: HybridMlDriver,
}

__all__ = [
    "AgenticDriver",
    "ClassicalDriver",
    "DRIVERS",
    "EndToEndDriver",
    "HybridMlDriver",
    "SetContext",
]
