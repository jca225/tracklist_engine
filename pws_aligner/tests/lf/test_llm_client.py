from __future__ import annotations

"""Tests for the real Claude LlmClient (B4).

All tests run on a stub SDK client — no API key, no network, no cost in CI.
The real `anthropic` package is imported lazily inside ClaudeClient so the
default (no --llm) path never touches it.
"""

import sys
from dataclasses import dataclass, field

from pws_aligner.lf.llm_client import ClaudeClient, run_llm_lf
from pws_aligner.lf.llm import LlmVote


@dataclass
class _StubBlock:
    type: str
    text: str = ""


@dataclass
class _StubResponse:
    content: list
    stop_reason: str = "end_turn"


class _StubMessages:
    def __init__(self, response: _StubResponse) -> None:
        self._response = response
        self.calls: list[dict] = []

    def create(self, **kwargs) -> _StubResponse:
        self.calls.append(kwargs)
        return self._response


@dataclass
class _StubSdk:
    messages: _StubMessages = field(
        default_factory=lambda: _StubMessages(
            _StubResponse(
                [_StubBlock("text", '{"plausible": true, "confidence": 0.9}')]
            )
        )
    )


def test_complete_returns_text_block():
    sdk = _StubSdk()
    client = ClaudeClient(sdk_client=sdk)
    out = client.complete("prompt here")
    assert out == '{"plausible": true, "confidence": 0.9}'
    assert len(sdk.messages.calls) == 1
    assert sdk.messages.calls[0]["messages"][0]["content"] == "prompt here"


def test_default_model_is_opus():
    sdk = _StubSdk()
    ClaudeClient(sdk_client=sdk).complete("p")
    assert sdk.messages.calls[0]["model"] == "claude-opus-4-8"


def test_refusal_returns_empty_string():
    """A refusal (empty content) must yield '' → llm_lf abstains OUT_OF_DOMAIN."""
    sdk = _StubSdk()
    sdk.messages = _StubMessages(_StubResponse([], stop_reason="refusal"))
    out = ClaudeClient(sdk_client=sdk).complete("p")
    assert out == ""


def test_no_anthropic_import_without_client():
    """Importing llm_client must NOT import the anthropic SDK (lazy import)."""
    assert "pws_aligner.lf.llm_client" in sys.modules
    # if anthropic was pulled in by our module import, this fails
    import pws_aligner.lf.llm_client as m

    assert "anthropic" not in m.__dict__


def test_run_llm_lf_budgeted():
    """run_llm_lf votes on at most max_spans spans and returns LlmVotes."""
    sdk = _StubSdk()
    client = ClaudeClient(sdk_client=sdk)
    span_docs = [
        {"span_id": str(i), "recording_id": f"r{i}", "name": f"Track {i}"}
        for i in range(10)
    ]
    votes = run_llm_lf(span_docs, client, max_spans=3, dj="Two Friends")
    assert len(votes) == 3
    assert all(isinstance(v, LlmVote) for v in votes)
    assert len(sdk.messages.calls) == 3


def test_run_llm_lf_zero_budget_calls_nothing():
    sdk = _StubSdk()
    client = ClaudeClient(sdk_client=sdk)
    votes = run_llm_lf([{"span_id": "1", "recording_id": "r1"}], client, max_spans=0)
    assert votes == []
    assert sdk.messages.calls == []
