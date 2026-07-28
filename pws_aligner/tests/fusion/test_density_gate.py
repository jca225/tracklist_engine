from __future__ import annotations
from pws_aligner.core.votes import Vote, AbstainReason
from pws_aligner.fusion.density_gate import label_density, choose_aggregator


def _span(n_fire, n_abstain):
    fire = [
        Vote(f"p{i}", "s", "r1", 10.0, 0.7, False, AbstainReason.NONE, ())
        for i in range(n_fire)
    ]
    ab = [
        Vote(f"a{i}", "s", None, 0.0, 0.0, True, AbstainReason.NO_DATA, ())
        for i in range(n_abstain)
    ]
    return fire + ab


def test_density_counts_only_fired_votes():
    spans = [_span(3, 2), _span(1, 4)]
    assert label_density(spans) == 2.0  # (3 + 1) / 2


def test_gate_picks_mv_at_low_density_and_model_midband():
    assert (
        choose_aggregator([_span(1, 5) for _ in range(10)], low=1.5) == "majority_vote"
    )
    assert choose_aggregator([_span(3, 1) for _ in range(10)]) == "label_model"
