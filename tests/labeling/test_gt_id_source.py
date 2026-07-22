from __future__ import annotations

from pathlib import Path

from labeling.ground_truth.schema import (
    GroundTruthSet,
    GroundTruthTrack,
    dump,
    load,
)


def _track(**kw) -> GroundTruthTrack:
    base = dict(
        label="X",
        track_id="rec1",
        claimed_stem="regular",
        set_start_s=1.0,
        set_end_s=2.0,
        ref_start_s=0.0,
    )
    base.update(kw)
    return GroundTruthTrack(**base)


def test_id_source_defaults_empty() -> None:
    assert _track().id_source == ""


def test_id_source_round_trips(tmp_path: Path) -> None:
    gt = GroundTruthSet(
        set_id="s",
        tracks=(
            _track(id_source="content"),
            _track(track_id=None, id_source="abstain"),
        ),
    )
    p = tmp_path / "gt.yaml"
    p.write_text(dump(gt))
    back = load(p)
    assert back.is_ok()
    assert [t.id_source for t in back.value.tracks] == ["content", "abstain"]
