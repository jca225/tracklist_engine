from __future__ import annotations
from typing import Sequence
from workspaces.pws_aligner.votes import Vote


def label_density(spans: Sequence[Sequence[Vote]]) -> float:
    if not spans:
        return 0.0
    fired = sum(sum(1 for v in s if not v.abstained) for s in spans)
    return fired / len(spans)


def choose_aggregator(
    spans: Sequence[Sequence[Vote]], low: float = 1.0, high: float = 6.0
) -> str:
    d = label_density(spans)
    return "label_model" if low <= d <= high else "majority_vote"
