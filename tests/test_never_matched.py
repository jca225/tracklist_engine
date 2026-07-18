from pathlib import Path
import json
from workspaces.alignment_prototype.never_matched import (
    never_matched_recordings,
    write_never_matched,
)

_GT = [
    {"track_id": "matched1", "slot_label": "010", "claimed_stem": "regular"},
    {"track_id": "gone1", "slot_label": "097", "claimed_stem": "acappella"},
    {"track_id": "gone2", "slot_label": "121", "claimed_stem": "instrumental"},
    {
        "track_id": None,
        "slot_label": "200",
        "claimed_stem": "regular",
    },  # mix-only, skip
]
_SPANS = [{"recording_id": "matched1", "slot_label": "010"}]


def test_never_matched_covers_all_stems_and_skips_trackless():
    out = never_matched_recordings(_GT, _SPANS)
    ids = {e["recording_id"] for e in out}
    assert ids == {"gone1", "gone2"}  # instrumental included, not acappella-only
    assert all("slot_label" in e and "claimed_stem" in e for e in out)


def test_write_never_matched_json_shape(tmp_path):
    p = tmp_path / "nm.json"
    doc = write_never_matched("1fsnxchk", _GT, _SPANS, p)
    assert doc["set_id"] == "1fsnxchk"
    on_disk = json.loads(p.read_text())
    assert on_disk == doc
    assert len(on_disk["never_matched_recordings"]) == 2
