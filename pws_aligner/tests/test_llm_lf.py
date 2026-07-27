from __future__ import annotations

from pws_aligner.llm_lf import (
    LlmVote,
    candidate_plausibility_vote,
)
from pws_aligner.votes import AbstainReason


class _FakeClient:
    """A stand-in LLM client returning a canned completion (no API calls)."""

    def __init__(self, response: str):
        self._response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._response


_CTX = {
    "candidate_recording_id": "recA",
    "candidate_name": "Avicii - Levels",
    "dj": "Two Friends",
    "neighbors": ["Kygo - Firestone", "Zedd - Clarity"],
    "era": 2015,
}


def test_plausible_high_confidence_fires():
    client = _FakeClient('{"plausible": true, "confidence": 0.9}')
    v = candidate_plausibility_vote("span_1", _CTX, client)
    assert v.lf == "llm_plausibility"
    assert v.label == "plausible"
    assert v.recording_id == "recA"
    assert not v.abstained
    assert v.confidence == 0.9


def test_implausible_downvotes_candidate():
    client = _FakeClient('{"plausible": false, "confidence": 0.85}')
    v = candidate_plausibility_vote("span_1", _CTX, client)
    assert v.label == "implausible"
    assert v.recording_id is None  # implausible = do not endorse this candidate
    assert not v.abstained


def test_low_confidence_abstains_low_margin():
    client = _FakeClient('{"plausible": true, "confidence": 0.3}')
    v = candidate_plausibility_vote("span_1", _CTX, client, min_confidence=0.6)
    assert v.abstained
    assert v.reason == AbstainReason.LOW_MARGIN


def test_malformed_output_abstains_out_of_domain():
    # a hallucinated / unparseable completion must NOT be trusted — abstain, don't guess
    client = _FakeClient("Sure! Here is my answer: definitely plausible :)")
    v = candidate_plausibility_vote("span_1", _CTX, client)
    assert v.abstained
    assert v.reason == AbstainReason.OUT_OF_DOMAIN


def test_confidence_clamped_and_self_reported():
    # confidence is the LLM's SELF-REPORT (for gating only); label model learns true accuracy
    client = _FakeClient('{"plausible": true, "confidence": 1.5}')
    v = candidate_plausibility_vote("span_1", _CTX, client)
    assert 0.0 <= v.confidence <= 1.0


def test_prompt_includes_context():
    client = _FakeClient('{"plausible": true, "confidence": 0.8}')
    candidate_plausibility_vote("span_1", _CTX, client)
    prompt = client.prompts[0]
    assert "Avicii - Levels" in prompt
    assert "Two Friends" in prompt
    assert "Firestone" in prompt  # neighbors given as context
