"""Decode bridge: posterior dict -> timeline span placement.

Converts the PWS label model's per-span posterior (dict[Hypothesis, float])
into the placement dict that run_phase1 writes into the timeline JSON consumed
by score_timeline_vs_gt.
"""

from __future__ import annotations

from workspaces.pws_aligner.hypotheses import Hypothesis


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
