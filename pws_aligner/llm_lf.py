from __future__ import annotations

"""LLM-as-labeling-function for the PWS aligner.

An LF is any function that emits a noisy, abstaining vote; an LLM qualifies. The
label model does not care where a vote comes from — it learns the LLM's accuracy
like any probe. What makes an LLM LF worth its cost is that it votes on what the
audio-DSP LFs *cannot see*: semantic plausibility from world knowledge (does it
make sense for THIS DJ, in THIS era, between THESE neighbors, to play THIS track?).
That is the Weaver weak-verifier role — a language/world-knowledge sensor
complementary to the audio channels.

Two disciplines are baked in:
- **Self-reported confidence is for gating only.** LLM confidence is poorly
  calibrated, so it never becomes the fusion weight — the label model learns the
  true per-LF accuracy. We keep the LLM's number only to gate abstention.
- **Unparseable output is never trusted.** A hallucinated / non-JSON completion
  abstains (OUT_OF_DOMAIN); we never guess a vote from prose.

The client is injected (``LlmClient`` protocol) so this is testable with a fake
client and carries no API dependency here. Cost note: an LLM LF is a high-cost
sensor — it sits between the free DSP LFs and the budgeted human oracle in the
agent's sensor hierarchy; run it where text/semantics carry signal (version
ambiguity, unidentified spans), abstaining cheaply elsewhere.
"""

import json
from dataclasses import dataclass
from typing import Mapping, Protocol

from .votes import AbstainReason

_LF_NAME = "llm_plausibility"


class LlmClient(Protocol):
    """Minimal completion client. Real impl wraps the Claude API (see claude-api)."""

    def complete(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class LlmVote:
    lf: str
    span_id: str
    recording_id: str | None  # endorsed candidate, or None (implausible / abstain)
    label: str  # "plausible" | "implausible" | "none"
    confidence: float  # LLM self-report, [0,1]; gating only — NOT the fusion weight
    abstained: bool
    reason: AbstainReason


def _build_prompt(context: Mapping) -> str:
    neighbors = ", ".join(context.get("neighbors", []))
    return (
        "You are a DJ-set expert acting as a plausibility verifier.\n"
        f"DJ: {context.get('dj', '?')}\n"
        f"Era: {context.get('era', '?')}\n"
        f"Neighbouring tracks in the set: {neighbors}\n"
        f"Candidate track for this span: {context.get('candidate_name', '?')}\n\n"
        "Is it plausible this DJ played this candidate here, given the neighbours, "
        "era, and the DJ's style? Answer ONLY with JSON: "
        '{"plausible": true|false, "confidence": 0.0-1.0}.'
    )


def _extract_json(raw: str) -> dict:
    """Parse the first JSON object in the completion. Raises if none/invalid."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in completion")
    return json.loads(raw[start : end + 1])


def candidate_plausibility_vote(
    span_id: str,
    context: Mapping,
    client: LlmClient,
    *,
    min_confidence: float = 0.6,
) -> LlmVote:
    """Vote on whether a candidate is a plausible thing to play in this span.

    plausible + confident → endorse the candidate (recording_id set);
    implausible → do not endorse (recording_id None, but a fired vote);
    low confidence → abstain LOW_MARGIN; unparseable → abstain OUT_OF_DOMAIN.
    """
    raw = client.complete(_build_prompt(context))
    try:
        parsed = _extract_json(raw)
        plausible = bool(parsed["plausible"])
        conf = float(parsed["confidence"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return LlmVote(
            _LF_NAME, span_id, None, "none", 0.0, True, AbstainReason.OUT_OF_DOMAIN
        )

    conf = min(1.0, max(0.0, conf))
    if conf < min_confidence:
        return LlmVote(
            _LF_NAME, span_id, None, "none", conf, True, AbstainReason.LOW_MARGIN
        )
    if plausible:
        rid = context.get("candidate_recording_id")
        return LlmVote(
            _LF_NAME, span_id, rid, "plausible", conf, False, AbstainReason.NONE
        )
    return LlmVote(
        _LF_NAME, span_id, None, "implausible", conf, False, AbstainReason.NONE
    )
