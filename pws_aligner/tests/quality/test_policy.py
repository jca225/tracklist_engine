from __future__ import annotations

"""Tests for the per-span policy layer (B3).

The agent's policy over its sensor hierarchy: accept when the label model
is confident; actuate when the abstention is an ingredient problem (cheap);
call the LLM on contested spans (mid-cost); escalate to the human oracle
(most-budgeted); defer when every budget is dry. Thresholds are REUSED from
agentic/policy.py's Ladder — one tunable config, not a reinvention.
"""

from alignment.agentic.policy import Ladder
from pws_aligner.quality.policy import (
    Action,
    Budgets,
    SpanDecisionInput,
    decide_span,
    decide_spans,
)
from pws_aligner.core.votes import AbstainReason


def _inp(
    confidence: float,
    reasons: tuple[AbstainReason, ...] = (),
    attempts: int = 0,
) -> SpanDecisionInput:
    return SpanDecisionInput(
        span_id="s",
        posterior_confidence=confidence,
        abstain_reasons=reasons,
        reacquire_attempts=attempts,
    )


def test_confident_span_accepted():
    a = decide_span(_inp(0.95), Budgets(llm_calls=0, human_labels=0))
    assert a == Action.ACCEPT


def test_accept_threshold_comes_from_agentic_ladder():
    """Just above/below the Ladder's auto rung flips ACCEPT — genuine reuse."""
    ladder = Ladder()
    budgets = Budgets(llm_calls=0, human_labels=0)
    assert decide_span(_inp(ladder.auto + 0.01), budgets) == Action.ACCEPT
    assert decide_span(_inp(ladder.auto - 0.01), budgets) != Action.ACCEPT


def test_ingredient_abstention_actuates():
    a = decide_span(
        _inp(0.2, reasons=(AbstainReason.NO_DATA,)),
        Budgets(llm_calls=5, human_labels=5),
    )
    assert a == Action.ACTUATE


def test_actuation_budget_exhausted_falls_through():
    a = decide_span(
        _inp(0.5, reasons=(AbstainReason.NO_DATA,), attempts=2),
        Budgets(llm_calls=5, human_labels=5),
    )
    assert a != Action.ACTUATE


def test_low_margin_never_actuates():
    """Genuine ambiguity is not an ingredient problem — LLM, not redownload."""
    a = decide_span(
        _inp(0.5, reasons=(AbstainReason.LOW_MARGIN,)),
        Budgets(llm_calls=5, human_labels=5),
    )
    assert a == Action.CALL_LLM


def test_contested_span_without_llm_budget_escalates():
    a = decide_span(
        _inp(0.5, reasons=(AbstainReason.LOW_MARGIN,)),
        Budgets(llm_calls=0, human_labels=3),
    )
    assert a == Action.ESCALATE


def test_all_budgets_dry_defers():
    a = decide_span(
        _inp(0.05),
        Budgets(llm_calls=0, human_labels=0),
    )
    assert a == Action.DEFER


def test_hopeless_span_skips_llm_goes_to_human():
    """No usable signal (below the suggest rung) — an LLM verify of nothing
    is wasted spend; the human oracle is the right sensor."""
    a = decide_span(
        _inp(0.02),
        Budgets(llm_calls=5, human_labels=5),
    )
    assert a == Action.ESCALATE


def test_batch_decrements_budgets():
    """Two contested spans, one LLM call available: second falls to human."""
    spans = [
        _inp(0.5, reasons=(AbstainReason.LOW_MARGIN,)),
        _inp(0.5, reasons=(AbstainReason.LOW_MARGIN,)),
    ]
    actions = decide_spans(spans, Budgets(llm_calls=1, human_labels=1))
    assert actions == [Action.CALL_LLM, Action.ESCALATE]


def test_batch_human_budget_respected():
    spans = [_inp(0.02), _inp(0.02), _inp(0.02)]
    actions = decide_spans(spans, Budgets(llm_calls=0, human_labels=2))
    assert actions == [Action.ESCALATE, Action.ESCALATE, Action.DEFER]


def test_batch_accept_consumes_no_budget():
    spans = [_inp(0.95), _inp(0.95), _inp(0.02)]
    actions = decide_spans(spans, Budgets(llm_calls=0, human_labels=1))
    assert actions == [Action.ACCEPT, Action.ACCEPT, Action.ESCALATE]
