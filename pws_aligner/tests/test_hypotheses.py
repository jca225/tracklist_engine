from __future__ import annotations
from pws_aligner.votes import Vote, AbstainReason
from pws_aligner.hypotheses import (
    Hypothesis,
    build_hypothesis_space,
    vote_to_hypothesis,
)

NULL = Hypothesis(None, 0)


def _v(probe, rid, off, ab=False):
    return Vote(
        probe,
        "s",
        rid,
        off,
        0.7,
        ab,
        AbstainReason.NONE if not ab else AbstainReason.NO_DATA,
        (),
    )


def test_offset_binning_and_null():
    assert vote_to_hypothesis(_v("fp", "r1", 121.0), bin_s=2.0) == Hypothesis(
        "r1", 60
    )  # round(121/2)=60
    assert vote_to_hypothesis(_v("h", None, 0.0, ab=True)) == NULL


def test_space_dedups_and_puts_null_first():
    votes = [_v("fp", "r1", 120.0), _v("chroma", "r1", 121.0), _v("h", "r2", 300.0)]
    space = build_hypothesis_space(votes, bin_s=2.0)
    assert space[0] == NULL
    assert set(space) == {NULL, Hypothesis("r1", 60), Hypothesis("r2", 150)}
    assert len(space) == 3  # r1@120 and r1@121 collapse to bin 60
