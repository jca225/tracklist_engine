"""GT schema round-trip for volume-aware export fields."""
from __future__ import annotations

from labeling.ground_truth.schema import GroundTruthSet, GroundTruthTrack, dump, load
from core.result import Ok


def test_audible_fields_round_trip(tmp_path):
    gt_track = GroundTruthTrack(
        label="Artist - Title",
        track_id="abc123",
        claimed_stem="acappella",
        set_start_s=100.0,
        set_end_s=120.0,
        ref_start_s=5.0,
        ref_end_s=25.0,
        slot_label="130",
        audible_frac=0.65,
        audible_start_s=105.0,
        audible_end_s=118.0,
        skip_training=False,
    )
    text = dump(GroundTruthSet(set_id="testset", tracks=(gt_track,)))
    path = tmp_path / "gt.yaml"
    path.write_text(text)
    match load(path):
        case Ok(gt):
            t = gt.tracks[0]
            assert t.audible_frac == 0.65
            assert t.audible_start_s == 105.0
            assert t.audible_end_s == 118.0
            assert t.skip_training is False
        case _:
            raise AssertionError("expected Ok")
