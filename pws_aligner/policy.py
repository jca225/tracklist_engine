from __future__ import annotations

"""Per-span policy layer (B3) — the agent's policy over its sensor hierarchy.

Arbitrates what to do with each span after the label model has spoken:

  ACCEPT     label model confident — commit the fused placement.
  ACTUATE    the abstention is an INGREDIENT problem (NO_DATA /
             OUT_OF_DOMAIN) — re-acquire the reference (cheap; per-recording
             attempt budget lives in actuation.py, mirrored here).
  CALL_LLM   contested span with real signal — spend a mid-cost semantic
             verification (the LLM is one high-cost sensor, not the brain).
  ESCALATE   the human oracle — the most-budgeted sensor; also the right
             sensor when there is no usable signal for an LLM to verify.
  DEFER      every budget dry — propose nothing, commit nothing.

Threshold REUSE: the rungs come from agentic/policy.py's ``Ladder`` (one
tunable config across the agentic and PWS stacks): ``auto`` is the ACCEPT
rung, ``suggest`` is the some-signal floor below which an LLM call is wasted
spend. We reuse the Ladder's thresholds rather than its ``mode()`` because
the PWS posterior is a scalar confidence, not a SpanBelief.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from alignment.agentic.policy import Ladder

from .votes import AbstainReason

_MAX_REACQUIRE_ATTEMPTS = 2  # mirrors actuation._MAX_ATTEMPTS

_INGREDIENT_REASONS = frozenset({AbstainReason.NO_DATA, AbstainReason.OUT_OF_DOMAIN})


class Action(Enum):
    ACCEPT = "accept"
    ACTUATE = "actuate"
    CALL_LLM = "call_llm"
    ESCALATE = "escalate"
    DEFER = "defer"


@dataclass(frozen=True)
class Budgets:
    """Remaining spend for the two costed sensors.

    Actuation cost is bounded per recording (attempt budget), not per run,
    so it is not tracked here.
    """

    llm_calls: int
    human_labels: int


@dataclass(frozen=True)
class SpanDecisionInput:
    """Everything the policy needs to know about one span."""

    span_id: str
    posterior_confidence: float  # label-model posterior mass on the MAP recording
    abstain_reasons: tuple[AbstainReason, ...]
    reacquire_attempts: int = 0  # prior re-acquire count for this recording


def decide_span(
    inp: SpanDecisionInput,
    budgets: Budgets,
    *,
    ladder: Ladder | None = None,
) -> Action:
    """Map (posterior, abstention reasons, budgets) → one action for one span.

    Order encodes the cost hierarchy: free acceptance first, then the cheap
    corpus fix, then the mid-cost LLM, then the budgeted human, then defer.
    LOW_MARGIN never actuates — ambiguity is resolved by better sensing, not
    by re-downloading (detect-then-correct).
    """
    ladder = ladder or Ladder()
    q = inp.posterior_confidence

    if q >= ladder.auto:
        return Action.ACCEPT

    has_ingredient_problem = any(r in _INGREDIENT_REASONS for r in inp.abstain_reasons)
    if has_ingredient_problem and inp.reacquire_attempts < _MAX_REACQUIRE_ATTEMPTS:
        return Action.ACTUATE

    # Below the suggest rung there is no usable signal for the LLM to verify —
    # a semantic plausibility check of nothing is wasted spend; go human.
    if q >= ladder.suggest and budgets.llm_calls > 0:
        return Action.CALL_LLM

    if budgets.human_labels > 0:
        return Action.ESCALATE

    return Action.DEFER


def decide_spans(
    spans: Sequence[SpanDecisionInput],
    budgets: Budgets,
    *,
    ladder: Ladder | None = None,
) -> list[Action]:
    """Allocate actions across spans, decrementing budgets as they are spent.

    In-order allocation: earlier spans get first claim on the LLM/human
    budgets. Callers who want priority ordering (e.g. most-contested first)
    sort before calling.
    """
    llm_left = budgets.llm_calls
    human_left = budgets.human_labels
    actions: list[Action] = []
    for inp in spans:
        action = decide_span(
            inp,
            Budgets(llm_calls=llm_left, human_labels=human_left),
            ladder=ladder,
        )
        if action == Action.CALL_LLM:
            llm_left -= 1
        elif action == Action.ESCALATE:
            human_left -= 1
        actions.append(action)
    return actions
