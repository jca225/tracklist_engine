"""Pure decision logic (§9) — posterior → decision, no I/O.

An abstention (``UNRESOLVED``) is returned when the winner is not confident
enough (below ``minimum_posterior``) or the posterior is too spread
(``entropy`` above ``maximum_entropy``). This is the "abstain, never lie" law in
one place, testable without a database.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional

from .types import Decision, DecisionRule


def entropy(posterior: Mapping[str, float]) -> float:
    """Shannon entropy (nats) of a posterior; 0.0 for empty/degenerate."""
    total = sum(max(0.0, p) for p in posterior.values())
    if total <= 0.0:
        return 0.0
    h = 0.0
    for p in posterior.values():
        q = max(0.0, p) / total
        if q > 0.0:
            h -= q * math.log(q)
    return h


def decide(
    posterior: Mapping[str, float], rule: DecisionRule
) -> tuple[Decision, Optional[str]]:
    """(decision, chosen_candidate_id). Abstains (UNRESOLVED, None) when the
    winner is below the posterior floor or the entropy is above the ceiling."""
    if not posterior:
        return Decision.UNRESOLVED, None
    winner, probability = max(posterior.items(), key=lambda kv: kv[1])
    if probability < rule.minimum_posterior:
        return Decision.UNRESOLVED, None
    if entropy(posterior) > rule.maximum_entropy:
        return Decision.UNRESOLVED, None
    return Decision.ACCEPTED, winner
