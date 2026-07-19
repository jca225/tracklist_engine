"""Whole-mix local alignment from raw fingerprint correspondence points."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from workspaces.alignment_prototype.looptrace.config import SEG_V1, SegmentConfig
from workspaces.alignment_prototype.looptrace.segments import (
    cover_dp_runs,
    path_inlier_evidence,
)

from .schema import ConstituentSegment, LandmarkMatch


def _validate_homogeneous(matches: Sequence[LandmarkMatch]) -> tuple[str, str, str]:
    keys = {(m.recording_id, m.ref_stem, m.mix_channel) for m in matches}
    if len(keys) != 1:
        raise ValueError("matches must belong to one recording and channel route")
    return next(iter(keys))


def decode_constituent(
    matches: Sequence[LandmarkMatch],
    *,
    mix_duration_s: float,
    allowed_slopes: Sequence[float] = (1.0,),
    config: SegmentConfig = SEG_V1,
    min_run_evidence_fraction: float = 0.05,
) -> tuple[ConstituentSegment, ...]:
    """Decode disjoint appearances of one constituent across an entire mix.

    The input is the raw sparse match cloud. Hough candidates and the
    NULL-aware cover DP decide both the active diagonal and its explicit
    boundaries. Multiple non-NULL runs preserve gaps, jumps, and re-entries.

    ``min_run_evidence_fraction`` drops weak secondary collision runs relative
    to the strongest kept path. Default 0.05 is the synthetic-fixture floor;
    raising it (e.g. 0.25) rejects more false extras but regressed real-set
    recall on both BB11 and BB12 — do not retune on those sets alone.
    """
    if not matches:
        return ()
    if not np.isfinite(mix_duration_s) or mix_duration_s <= 0:
        raise ValueError("mix_duration_s must be finite and positive")
    if not allowed_slopes or any(
        not np.isfinite(slope) or slope <= 0 for slope in allowed_slopes
    ):
        raise ValueError("allowed_slopes must contain positive finite values")
    if not 0.0 <= min_run_evidence_fraction <= 1.0:
        raise ValueError("min_run_evidence_fraction must be in [0, 1]")

    recording_id, ref_stem, mix_channel = _validate_homogeneous(matches)
    points = np.asarray(
        [(match.mix_time_s, match.ref_time_s) for match in matches],
        dtype=np.float32,
    )
    weights = np.asarray([match.weight for match in matches], dtype=np.float64)

    best_runs = ()
    best_slope = 0.0
    best_score = float("-inf")
    for slope in allowed_slopes:
        runs, _support = cover_dp_runs(
            points,
            float(slope),
            mix_duration_s,
            config,
            weights=weights,
        )
        legacy = [(run.mix_start_s, run.song_start_s, run.song_end_s) for run in runs]
        inliers = path_inlier_evidence(
            points,
            legacy,
            float(slope),
            mix_duration_s,
            config,
            weights=weights,
        )
        score = inliers - config.seg_cost_s * len(runs)
        if score > best_score:
            best_runs = tuple(runs)
            best_slope = float(slope)
            best_score = score

    if not best_runs or best_score <= 0:
        return ()

    total_weight = float(weights.sum())
    strongest_run = max(run.evidence for run in best_runs)
    out: list[ConstituentSegment] = []
    for run in best_runs:
        # A Hough bank is deliberately recall-heavy. Very weak states can tile
        # collision noise for a few seconds even though NULL wins most of the
        # gap. Require each emitted run to carry a fixed share of the strongest
        # path evidence; the threshold is synthetic-fixture configuration, not
        # calibrated confidence or a real-set slot rule.
        if run.evidence < min_run_evidence_fraction * strongest_run:
            continue
        # Evidence share is deliberately bounded and local. It is useful for
        # comparing runs from this retrieval, not calibrated acceptance.
        confidence = min(1.0, run.evidence / max(total_weight, 1e-9))
        if run.song_start_s < 0 or run.song_end_s <= run.song_start_s:
            continue
        out.append(
            ConstituentSegment(
                recording_id=recording_id,
                slot_label=None,
                ref_stem=ref_stem,
                mix_channel=mix_channel,
                mix_start_s=run.mix_start_s,
                mix_end_s=run.mix_end_s,
                ref_start_s=run.song_start_s,
                ref_end_s=run.song_end_s,
                slope=best_slope,
                evidence=run.evidence,
                confidence=confidence,
            )
        )
    return tuple(out)
