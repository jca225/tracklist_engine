"""Collision-aware retrieval of raw landmark time correspondences."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from workspaces.alignment_prototype.landmark_fp import FHOP, SR, LandmarkFingerprint

from .schema import LandmarkMatch


def retrieve_matches(
    mix_hashes: Mapping[object, Sequence[int]],
    reference: LandmarkFingerprint,
    *,
    recording_id: str,
    ref_stem: str,
    mix_channel: str,
    pair_cap: int = 64,
) -> tuple[LandmarkMatch, ...]:
    """Return weighted raw matches without selecting an offset or time window.

    A shared hash produces the Cartesian product of its mix and reference
    timestamps. Common hashes are both expensive and non-localizing, so keys
    above ``pair_cap`` abstain. Surviving pairs receive inverse-frequency
    weight based on that product. This is retrieval cleanup, not a placement
    threshold: every retained time pair reaches the path decoder.
    """
    if pair_cap < 1:
        raise ValueError("pair_cap must be positive")
    scale = FHOP / SR
    out: list[LandmarkMatch] = []
    for key, mix_times in mix_hashes.items():
        ref_times = reference.hashes.get(key)
        if not ref_times:
            continue
        frequency = len(mix_times) * len(ref_times)
        if frequency > pair_cap:
            continue
        weight = 1.0 / math.log2(2.0 + frequency)
        for mix_frame in mix_times:
            for ref_frame in ref_times:
                if mix_frame < 0 or ref_frame < 0:
                    continue
                out.append(
                    LandmarkMatch(
                        recording_id=recording_id,
                        ref_stem=ref_stem,
                        mix_channel=mix_channel,
                        mix_time_s=float(mix_frame) * scale,
                        ref_time_s=float(ref_frame) * scale,
                        weight=weight,
                        hash_frequency=frequency,
                    )
                )
    return tuple(out)
