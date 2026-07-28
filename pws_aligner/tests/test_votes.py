from __future__ import annotations
from alignment.harness.contract import AlignmentResult
from pws_aligner.votes import Vote, AbstainReason, collect_votes


def test_collect_maps_result_and_abstain_reason():
    fired = AlignmentResult(
        recording_id="r1", offset_s=120.0, confidence=0.8, source="fp"
    )
    skipped = AlignmentResult.abstained(source="hubert")
    votes = collect_votes(
        "span0",
        [
            (fired, (0.8, 1.2), AbstainReason.NONE),
            (skipped, (0.0, 0.0), AbstainReason.NO_DATA),
        ],
    )
    assert votes == (
        Vote("fp", "span0", "r1", 120.0, 0.8, False, AbstainReason.NONE, (0.8, 1.2)),
        Vote(
            "hubert", "span0", None, None, 0.0, True, AbstainReason.NO_DATA, (0.0, 0.0)
        ),
    )
