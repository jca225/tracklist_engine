from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from pws_aligner.core.votes import Vote


@dataclass(frozen=True)
class Hypothesis:
    recording_id: str | None
    offset_bin: int


def vote_to_hypothesis(vote: Vote, bin_s: float = 2.0) -> Hypothesis:
    if vote.abstained or vote.recording_id is None:
        return Hypothesis(None, 0)
    return Hypothesis(vote.recording_id, round(vote.offset_s / bin_s))


def build_hypothesis_space(
    votes: Sequence[Vote], bin_s: float = 2.0
) -> tuple[Hypothesis, ...]:
    null = Hypothesis(None, 0)
    others = {
        vote_to_hypothesis(v, bin_s)
        for v in votes
        if not v.abstained and v.recording_id is not None
    }
    ordered = sorted(others, key=lambda h: (h.recording_id or "", h.offset_bin))
    return (null, *ordered)
