"""Permission ladder — autonomy graded by belief quality.

The design's central adaptation from agent-system practice: instead of a
binary auto/escalate, a ladder whose rungs are set by the *measured* probe
precisions. A span earns autonomy; it is never granted by default. The
validation experiment reports the actual clean-rate per rung vs GT — that
measurement IS the calibration, so thresholds live in one tunable config.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from workspaces.alignment_prototype.agentic.belief import SpanBelief


class Mode(str, Enum):
    AUTO_COMMIT = "auto_commit"  # write pseudo-GT, log only
    REVIEW = "review"  # place + flag into the review queue
    SUGGEST = "suggest"  # propose candidates, commit nothing
    ESCALATE = "escalate"  # human labeling queue (active labeling)


@dataclass(frozen=True)
class Ladder:
    """Quality thresholds (belief.quality() ∈ [0,1]) → autonomy mode.

    Defaults: auto ≈ a 0.90-precision probe winning ≥5/6 of the vote mass;
    review ≈ a trustworthy probe winning a contested vote; suggest = any
    signal at all; below that, no probe said anything usable → escalate.
    """

    auto: float = 0.75
    review: float = 0.40
    suggest: float = 0.10
    combine: bool = False  # reward independent-probe agreement in quality()

    def mode(self, belief: SpanBelief) -> Mode:
        q = belief.quality(combine=self.combine)
        if q >= self.auto:
            return Mode.AUTO_COMMIT
        if q >= self.review:
            return Mode.REVIEW
        if q >= self.suggest:
            return Mode.SUGGEST
        return Mode.ESCALATE
