from __future__ import annotations

"""Cue-time placement labeling function for the PWS aligner.

1001tracklists scrapes a ``cue_seconds`` timestamp for ~100% of corpus slots.
This timestamp marks where in the mix a track plays (SET_START) and is the
cheapest, highest-coverage signal we have.

Noise model — measured on BB11 vs audio-corrected GT (leakage caveat: BB11 GT
was cue-seeded, so these are an empirical prior the downstream label model
refines rather than a clean held-out estimate):

    signed residual (cue − true_set_start): median +0.8 s
    |residual| median: 1.1 s
    within 5 s: 67 %
    within 20 s: 81 %
    heavy tail: some slots 80–92 s off

This module turns the raw cue into a PlacementVote: bias-corrected set_start_s
+ a fixed inlier-rate confidence.  Per-span σ is NOT baked in here — a
downstream label model learns that from the residual distribution.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Module-level constants (measured on BB11; treat as an empirical prior)
# ---------------------------------------------------------------------------

CUE_BIAS_S: float = 0.8
"""Median signed residual (cue − true_set_start) on BB11.

Cue marks the mix-in moment, which is slightly late relative to true entry.
We subtract this to produce a bias-corrected set_start estimate.
"""

CUE_INLIER_SIGMA_S: float = 2.0
"""Approximate 1-sigma spread of the inlier population (BB11 |residual| ~1.1 s median).

Not used at emit time -- included so the downstream label model can initialise
the inlier Gaussian rather than rediscovering it from scratch.
"""

CUE_INLIER_RATE: float = 0.8
"""Fraction of cue timestamps that are within ~20 s of true set_start (BB11).

This is the emit-time confidence: the prior probability that a given cue vote
is in the inlier regime.  The label model learns a per-probe sigma that refines
this into an inverse-variance weight.
"""

CUE_OUTLIER_HALF_WIDTH_S: float = 300.0
"""Half-width of the flat outlier component (mixture model prior).

Covers the ~19 % of slots that are more than 20 s off.  300 s (5 min) is a
conservatively wide uniform slab that spans typical DJ-set slot spacings.
"""


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlacementVote:
    """A single placement vote emitted by a labeling function.

    Join key
    --------
    ``recording_id`` is THE join key when merging votes against the timeline
    or GT.  Do NOT join on slot_label / slot index:  BB12 timeline slots top
    out at 42 while GT slots reach 155 on multi-segment sets -- a slot-index
    join silently corrupts placement scoring on those sets.

    ``slot_label`` is retained for traceability only (human-readable log).
    """

    lf: str
    """Name of the labeling function that emitted this vote, e.g. ``"cue_time"``."""

    set_id: str
    """Identifier for the DJ set (e.g. ``"BB11"``, ``"2nvzlh2k"``)."""

    recording_id: str
    """THE join key -- canonical recording identifier.  Never the slot index."""

    slot_label: str
    """Human-readable slot label for traceability.  NOT the join key."""

    set_start_s: float | None
    """Bias-corrected placement estimate in seconds, or None if abstained."""

    confidence: float
    """Inlier-rate prior in [0, 1].  0.0 when abstained."""

    abstained: bool
    """True when the LF cannot produce a useful estimate."""

    reason: str
    """Short reason string: ``""`` when fired, ``"no_cue"`` when abstained."""


# ---------------------------------------------------------------------------
# LF implementation
# ---------------------------------------------------------------------------


def cue_placement_vote(
    set_id: str,
    recording_id: str,
    slot_label: str,
    cue_seconds: float | None,
) -> PlacementVote:
    """Emit a single placement vote from a scraped cue timestamp.

    Parameters
    ----------
    set_id:
        DJ set identifier.
    recording_id:
        Canonical recording identifier -- the join key.  See ``PlacementVote``
        docstring for why slot index must not be used instead.
    slot_label:
        Human-readable label for traceability.
    cue_seconds:
        Scraped cue timestamp in seconds, or None.  The value 0.0 is a
        sentinel meaning "no cue / plays at mix start" and is treated as
        missing (abstain) rather than a real placement at t=0.

    Returns
    -------
    PlacementVote
        Abstained (reason="no_cue") when cue_seconds is None or 0.0.
        Otherwise set_start_s = cue_seconds - CUE_BIAS_S, confidence =
        CUE_INLIER_RATE.
    """
    if cue_seconds is None or cue_seconds == 0.0:
        return PlacementVote(
            lf="cue_time",
            set_id=set_id,
            recording_id=recording_id,
            slot_label=slot_label,
            set_start_s=None,
            confidence=0.0,
            abstained=True,
            reason="no_cue",
        )

    return PlacementVote(
        lf="cue_time",
        set_id=set_id,
        recording_id=recording_id,
        slot_label=slot_label,
        set_start_s=cue_seconds - CUE_BIAS_S,
        confidence=CUE_INLIER_RATE,
        abstained=False,
        reason="",
    )


def cue_placement_votes(rows: Sequence[Mapping]) -> list[PlacementVote]:
    """Batch-emit placement votes from a sequence of row mappings.

    Each row must provide: ``set_id``, ``recording_id``, ``slot_label``,
    ``cue_seconds``.

    Rows with falsy ``recording_id`` are silently skipped -- they cannot be
    joined against the timeline or GT so emitting a vote for them would
    pollute the label-model input.  Order of the remaining rows is preserved.

    Parameters
    ----------
    rows:
        Sequence of Mapping-like objects (dicts, sqlite3.Row, etc.).

    Returns
    -------
    list[PlacementVote]
        One vote per row with a truthy recording_id, in input order.
    """
    result: list[PlacementVote] = []
    for row in rows:
        recording_id = row["recording_id"]
        # recording_id is the join key: timeline and GT use different slot-index
        # conventions on multiseg sets (BB12: timeline slots max 42 vs GT max 155)
        # -- a slot-index join silently corrupts.  Skip rows we can't key.
        if not recording_id:
            continue
        result.append(
            cue_placement_vote(
                set_id=row["set_id"],
                recording_id=recording_id,
                slot_label=row["slot_label"],
                cue_seconds=row["cue_seconds"],
            )
        )
    return result
