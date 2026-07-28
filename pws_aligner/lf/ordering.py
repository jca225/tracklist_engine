from __future__ import annotations

"""Soft tracklist-ordering labeling function (A5).

Tracklist order is today enforced as a HARD monotonic constraint
(``sequence_decode.monotonic_decode``). That hard path stays; this LF
re-expresses the same prior as a SOFT vote the label model can weigh —
so strong audio evidence can outvote it (reorders / medleys / B2B sets,
e.g. the Slide −746s poisoning case where the hard constraint dragged a
whole neighbourhood of spans off the diagonal).

Semantics: the DJ *probably* played tracks in listed order, so an
in-order placement gets a high-confidence ``in_order`` vote and a
backward step gets a LOW-confidence ``out_of_order`` vote whose
confidence decays with the size of the violation — but never reaches
zero and never abstains. A hard reject is exactly what this LF exists
to avoid.
"""

import math
from dataclasses import dataclass

from pws_aligner.core.votes import AbstainReason

_LF_NAME: str = "ordering_soft"

# Backward nudges below this are placement jitter, not reorders.
_JITTER_TOLERANCE_S: float = 2.0
# e-folding scale of the confidence decay for genuine backward steps.
# ~30 s ≈ a couple of phrases: a small swap is plausible, −746 s is not.
_DECAY_SCALE_S: float = 30.0
# In-order prior confidence (the label model learns the true accuracy).
_IN_ORDER_CONFIDENCE: float = 0.8
# Ceiling for out-of-order votes and floor for the decay — soft, never zero.
_OUT_OF_ORDER_CEIL: float = 0.4
_CONFIDENCE_FLOOR: float = 0.02


@dataclass(frozen=True)
class OrderingVote:
    """Vote from the ordering LF on a single span.

    ``label`` is ``"in_order"`` or ``"out_of_order"``; when ``abstained``
    is True it is ``"none"`` and MUST NOT be read as a decision.
    """

    lf: str
    span_id: str
    label: str
    confidence: float
    abstained: bool
    reason: AbstainReason


def _violation_s(
    own_start_s: float, prev_start_s: float | None, next_start_s: float | None
) -> float:
    """Largest ordering violation in seconds (0.0 when in order)."""
    backward = 0.0
    if prev_start_s is not None:
        backward = max(backward, prev_start_s - own_start_s)
    if next_start_s is not None:
        backward = max(backward, own_start_s - next_start_s)
    return backward


def ordering_soft_vote(
    span_id: str,
    own_start_s: float,
    prev_start_s: float | None,
    next_start_s: float | None,
) -> OrderingVote:
    """Soft monotonicity vote for one span given its neighbours' placements.

    Args:
        span_id: identifier for the span being voted on.
        own_start_s: this span's placed set_start (seconds into the mix).
        prev_start_s: the previous slot's placement, or None if unknown.
        next_start_s: the next slot's placement, or None if unknown.

    Returns:
        ``in_order`` at the prior confidence when the placement respects
        listed order (within jitter tolerance); ``out_of_order`` with a
        confidence that decays with the violation size but never hits
        zero; abstains (NO_DATA) when no neighbour placement exists.
    """
    if prev_start_s is None and next_start_s is None:
        return OrderingVote(
            lf=_LF_NAME,
            span_id=span_id,
            label="none",
            confidence=0.0,
            abstained=True,
            reason=AbstainReason.NO_DATA,
        )

    violation = _violation_s(own_start_s, prev_start_s, next_start_s)
    if violation <= _JITTER_TOLERANCE_S:
        return OrderingVote(
            lf=_LF_NAME,
            span_id=span_id,
            label="in_order",
            confidence=_IN_ORDER_CONFIDENCE,
            abstained=False,
            reason=AbstainReason.NONE,
        )

    decayed = _OUT_OF_ORDER_CEIL * math.exp(
        -(violation - _JITTER_TOLERANCE_S) / _DECAY_SCALE_S
    )
    return OrderingVote(
        lf=_LF_NAME,
        span_id=span_id,
        label="out_of_order",
        confidence=max(_CONFIDENCE_FLOOR, decayed),
        abstained=False,
        reason=AbstainReason.NONE,
    )
