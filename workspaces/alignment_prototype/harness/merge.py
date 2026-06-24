"""Phase 3.6a: combine probe results into one decision (deterministic driver core).

The harness exists so heterogeneous probes compose. This is where that pays off:
take the AlignmentResults several probes produced for one (mix span, candidate)
and fuse them into a single AlignmentResult. The rule rewards *independent
corroboration* — when a second probe agrees on the placement, confidence rises;
a lone confident probe is trusted less than two that agree. Abstaining probes are
ignored; if nothing clears the bar, the merge itself abstains rather than guess.

Confidence comparison assumes calibrated [0,1] (Phase 3.3); it already works with
the provisional monotone squashes because agreement is checked on offset, not score.
"""

from __future__ import annotations

from .contract import AlignmentResult


def merge(
    results: tuple[AlignmentResult, ...],
    *,
    offset_tol_s: float = 2.0,
    min_confidence: float = 0.0,
    agreement_bonus: float = 0.1,
) -> AlignmentResult:
    """Fuse probe results into one decision.

    Picks the highest-confidence non-abstaining result as the winner, then boosts
    its confidence by ``agreement_bonus`` per *other* probe that independently
    agrees (same recording_id, offset within ``offset_tol_s``), capped at 1.0.
    Returns AlignmentResult.abstained when no probe commits or the winner is below
    ``min_confidence``.
    """
    live = [r for r in results if not r.abstain]
    if not live:
        return AlignmentResult.abstained(source="merge")

    winner = max(live, key=lambda r: r.confidence)
    agree = [
        r
        for r in live
        if r is not winner
        and r.recording_id == winner.recording_id
        and abs(r.offset_s - winner.offset_s) <= offset_tol_s
    ]
    confidence = min(1.0, winner.confidence + agreement_bonus * len(agree))

    if confidence < min_confidence:
        return AlignmentResult.abstained(
            source="merge", recording_id=winner.recording_id
        )

    sources = "+".join(sorted({winner.source, *(r.source for r in agree)} - {""}))
    return AlignmentResult(
        recording_id=winner.recording_id,
        offset_s=winner.offset_s,
        ref_end_s=winner.ref_end_s,
        segments=winner.segments,
        tempo_ratio=winner.tempo_ratio,
        confidence=confidence,
        abstain=False,
        source=f"merge({sources})" if sources else "merge",
    )
