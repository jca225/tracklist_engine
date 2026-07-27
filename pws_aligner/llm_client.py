from __future__ import annotations

"""Real Claude-API LlmClient for the LLM labeling function (B4).

The LLM LF scaffold (llm_lf.py) takes any injected client satisfying the
``LlmClient`` protocol. This module provides the real one, wrapping the
Claude API — behind the ``--llm`` flag in run_phase1, default OFF.

Disciplines:
  - The ``anthropic`` SDK is imported LAZILY inside the constructor: the
    default path (no --llm) never imports it, and CI never needs the
    package or an API key. Tests inject a stub SDK client.
  - A refusal / empty response returns "" — llm_lf then abstains
    OUT_OF_DOMAIN (unparseable output is never trusted).
  - ``run_llm_lf`` is hard-budgeted (``max_spans``): the LLM is a
    high-cost sensor, run on a handful of spans, never blanket-run.
"""

from typing import Any, Mapping, Sequence

from .llm_lf import LlmVote, candidate_plausibility_vote

_DEFAULT_MODEL = "claude-opus-4-8"
_DEFAULT_MAX_TOKENS = 256  # a plausibility verdict is a one-line JSON object


class ClaudeClient:
    """LlmClient implementation backed by the Claude API.

    Args:
        model: Claude model id (default claude-opus-4-8).
        max_tokens: response cap — the verdict is tiny JSON.
        sdk_client: injected SDK-compatible client (tests use a stub);
            when None, ``anthropic.Anthropic()`` is constructed lazily,
            resolving the API key from the environment.
    """

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        sdk_client: Any | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        if sdk_client is None:
            import anthropic  # lazy: only the --llm path pays this import

            sdk_client = anthropic.Anthropic()
        self._sdk = sdk_client

    def complete(self, prompt: str) -> str:
        """One prompt → the first text block's text ('' on refusal/empty)."""
        response = self._sdk.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if getattr(response, "stop_reason", None) == "refusal":
            return ""
        for block in response.content:
            if getattr(block, "type", None) == "text":
                return str(block.text)
        return ""


def run_llm_lf(
    span_docs: Sequence[Mapping],
    client: ClaudeClient,
    *,
    max_spans: int,
    dj: str = "?",
    era: str = "?",
) -> list[LlmVote]:
    """Run the LLM plausibility LF on at most ``max_spans`` spans.

    Builds the plausibility context from each span doc's neighbours and
    candidate name; spans beyond the budget are simply not voted on (no
    abstain spam — absence of a vote is the correct representation of
    "sensor not run").
    """
    votes: list[LlmVote] = []
    names = [str(d.get("name", "")) for d in span_docs]
    for i, doc in enumerate(span_docs[: max(0, max_spans)]):
        neighbors = [
            n
            for n in (
                names[i - 1] if i > 0 else "",
                names[i + 1] if i + 1 < len(names) else "",
            )
            if n
        ]
        context = {
            "dj": dj,
            "era": era,
            "neighbors": neighbors,
            "candidate_name": doc.get("name", "?"),
            "candidate_recording_id": doc.get("recording_id"),
        }
        votes.append(candidate_plausibility_vote(str(doc["span_id"]), context, client))
    return votes
