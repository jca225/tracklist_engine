"""Decode bridge: posterior dict -> timeline span placement.

Converts the PWS label model's per-span posterior (dict[Hypothesis, float])
into the placement dict that run_phase1 writes into the timeline JSON consumed
by score_timeline_vs_gt.
"""

from __future__ import annotations

from workspaces.pws_aligner.hypotheses import Hypothesis

# TYPE_CHECKING guard avoids a circular import at runtime; FusedSpan is only
# needed for the type annotation in fused_to_placement.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from workspaces.pws_aligner.continuous_model import FusedSpan


def posterior_to_placement(
    span_id: str,
    posterior: dict[Hypothesis, float],
    bin_s: float = 2.0,
) -> dict:
    """MAP hypothesis -> placement dict for one timeline span.

    Parameters
    ----------
    span_id:
        Identifies which span this placement belongs to (passed through to the
        output dict).
    posterior:
        ``{Hypothesis: probability}`` as returned by ``LabelModel.predict_proba``.
    bin_s:
        Seconds per offset bin (must match what ``vote_to_hypothesis`` used).

    Returns
    -------
    dict with keys:
        ``span_id``, ``recording_id``, ``offset_s``, ``confidence``, ``abstain``.
    ``abstain`` is True when the MAP hypothesis is the NULL one (recording_id is
    None).  ``offset_s`` is the bin centre in seconds (``offset_bin * bin_s``).
    """
    top = max(posterior, key=posterior.__getitem__)
    is_null = top.recording_id is None
    return {
        "span_id": span_id,
        "recording_id": None if is_null else top.recording_id,
        "offset_s": 0.0 if is_null else top.offset_bin * bin_s,
        "confidence": float(posterior[top]),
        "abstain": is_null,
    }


def fused_to_placement(span_id: str, fused: "FusedSpan") -> dict:
    """Continuous-model analog of posterior_to_placement: same output schema,
    but offset_s is the un-binned fused value (the whole point of the
    continuous model is not to quantize).

    Output keys are identical to posterior_to_placement:
        ``span_id``, ``recording_id``, ``offset_s``, ``confidence``, ``abstain``.
    """
    abstain = fused.recording_id is None
    return {
        "span_id": span_id,
        "recording_id": fused.recording_id,
        "offset_s": 0.0 if abstain else round(fused.offset_s, 3),
        "confidence": round(fused.confidence, 4),
        "abstain": abstain,
    }
