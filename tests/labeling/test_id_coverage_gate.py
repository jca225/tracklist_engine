from __future__ import annotations

from labeling.export_als_to_gt import id_coverage
from labeling.ground_truth.schema import GroundTruthTrack


def _t(id_source: str, rid: str | None) -> GroundTruthTrack:
    return GroundTruthTrack(
        label="x",
        track_id=rid,
        claimed_stem="regular",
        set_start_s=0.0,
        set_end_s=1.0,
        ref_start_s=0.0,
        id_source=id_source,
    )


def test_coverage_counts_content_only() -> None:
    tracks = [_t("content", "r1"), _t("abstain", None), _t("content", "r2")]
    resolved, total, frac = id_coverage(tracks)
    assert (resolved, total) == (2, 3)
    assert abs(frac - 2 / 3) < 1e-9
