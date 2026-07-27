from __future__ import annotations

"""Claimed-axes identity labeling function (A4).

The scraped ``claimed_version`` / ``claimed_stem`` fields are an unused noisy
identity prior: if the tracklist says "Remix" for this slot and we have a remix
recording_id, the claim endorses that identity. Pure metadata — no audio.

The label model learns the true accuracy of this prior from the corpus; we emit
a fixed prior confidence that the model can weight appropriately.

Emits a ``ClaimedIdentityVote`` (a frozen dataclass) rather than reusing
``OperationVote``, because this vote is identity-flavoured (carries
``recording_id``) rather than operation-flavoured. The contract mirrors
``OperationVote`` in spirit: ``abstained``, ``reason``, ``confidence``,
``lf``, ``span_id``.
"""

from dataclasses import dataclass

from .votes import AbstainReason

# Fixed prior confidence — the label model learns the true accuracy; this is
# just our prior belief that a scraped claim is roughly right.
_PRIOR_CONFIDENCE: float = 0.6

_LF_NAME: str = "claimed_axes_identity"
_ENDORSED_LABEL: str = "identity_endorsed"


@dataclass(frozen=True)
class ClaimedIdentityVote:
    """Vote from the claimed-axes identity LF on a single span.

    Fields mirror ``OperationVote`` for downstream compatibility, plus
    ``recording_id`` so the label model can key on the specific recording.

    When ``abstained`` is True, ``label`` is ``"none"`` and MUST NOT be
    read as a decision.
    """

    lf: str
    span_id: str
    recording_id: str | None
    label: str
    confidence: float
    abstained: bool
    reason: AbstainReason


def claimed_identity_vote(
    span_id: str,
    recording_id: str | None,
    claimed_version: str | None,
    claimed_stem: str | None,
) -> ClaimedIdentityVote:
    """Emit a soft identity vote from the scraped claimed_version / claimed_stem.

    Endorses the recording with a fixed prior confidence when at least one claim
    axis is present and non-empty. Abstains (NO_DATA) when both claims are absent
    or empty — there is no metadata signal to endorse.

    Args:
        span_id: identifier for the span being voted on.
        recording_id: the recording this span is associated with.
        claimed_version: scraped claimed_version (e.g. "remix", "original") or None.
        claimed_stem: scraped claimed_stem (e.g. "acappella", "instrumental") or None.

    Returns:
        A ClaimedIdentityVote endorsing the recording or abstaining.
    """
    has_version = bool(claimed_version and claimed_version.strip())
    has_stem = bool(claimed_stem and claimed_stem.strip())

    if not has_version and not has_stem:
        return ClaimedIdentityVote(
            lf=_LF_NAME,
            span_id=span_id,
            recording_id=recording_id,
            label="none",
            confidence=0.0,
            abstained=True,
            reason=AbstainReason.NO_DATA,
        )

    return ClaimedIdentityVote(
        lf=_LF_NAME,
        span_id=span_id,
        recording_id=recording_id,
        label=_ENDORSED_LABEL,
        confidence=_PRIOR_CONFIDENCE,
        abstained=False,
        reason=AbstainReason.NONE,
    )
