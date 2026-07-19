from pathlib import Path

from workspaces.alignment_prototype.identity_bridge import (
    canonical_recording_id,
    canonicalize_gt_rows,
    load_identity_map,
)


def test_identity_bridge_normalizes_before_join(tmp_path: Path) -> None:
    p = tmp_path / "labeling" / "fixtures" / "id_maps"
    p.mkdir(parents=True)
    (p / "set1.json").write_text('{"legacy7": "recording7"}')
    identity_map = load_identity_map("set1", repo=tmp_path)
    assert canonical_recording_id("legacy7", identity_map) == "recording7"
    assert canonical_recording_id("already", identity_map) == "already"
    rows = canonicalize_gt_rows(
        "set1",
        [{"track_id": "legacy7", "slot_label": "007"}, {"track_id": None}],
        repo=tmp_path,
    )
    assert rows[0]["track_id"] == "recording7"
    assert rows[1]["track_id"] is None
