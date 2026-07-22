"""Content binding replaces the slot_id_map guess: a clip whose slot 'looks like'
a different recording must bind to its own audio content or abstain — never the
slot's id."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from labeling.als.models import ParsedClip, WarpMarkers
from labeling.export_als_to_gt import _clip_row, _load_content_catalog
from labeling.als import build_manifest_index


def _mapper():
    class M:  # arr==set seconds
        def arr_to_set_sec(self, a):
            return a

    return M()


def _clip(path):
    return ParsedClip(
        group_name="g",
        track_name="t",
        path=path,
        arr_start=0.0,
        arr_end=5.0,
        loop_start=0.0,
        loop_end=5.0,
        pitch_coarse=0,
        pitch_fine=0,
        warp=WarpMarkers(points=((0.0, 0.0), (5.0, 5.0))),
    )


def test_binds_to_own_content_not_slot(tmp_path: Path) -> None:
    # File lives under slot '028' but its bytes belong to recording 'beatles'.
    d = tmp_path / "stems" / "028__X"
    d.mkdir(parents=True)
    f = d / "vocals.flac"
    f.write_bytes(b"BEATLES-VOCALS" * 100)
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps({"set_id": "s", "tracks": []}))
    (tmp_path / "content_catalog.json").write_text(
        json.dumps(
            {
                "set_id": "s",
                "entries": [
                    {
                        "content_sha256": sha,
                        "payload_sha256": None,
                        "recording_id": "beatles",
                        "track_audio_id": "ta",
                        "stem": "acappella",
                    }
                ],
            }
        )
    )
    manifest = build_manifest_index(tmp_path / "manifest.json")
    catalog = _load_content_catalog(tmp_path)
    row = _clip_row(_clip(str(f)), _mapper(), manifest, catalog)
    assert row.recording_id == "beatles"
    assert row.id_source == "content"


def test_abstains_when_no_content_match(tmp_path: Path) -> None:
    d = tmp_path / "stems" / "031__Y"
    d.mkdir(parents=True)
    f = d / "vocals.flac"
    f.write_bytes(b"CCR-VOCALS" * 100)
    (tmp_path / "manifest.json").write_text(json.dumps({"set_id": "s", "tracks": []}))
    (tmp_path / "content_catalog.json").write_text(
        json.dumps({"set_id": "s", "entries": []})
    )
    manifest = build_manifest_index(tmp_path / "manifest.json")
    catalog = _load_content_catalog(tmp_path)
    row = _clip_row(_clip(str(f)), _mapper(), manifest, catalog)
    assert row.recording_id is None
    assert row.id_source == "abstain"
