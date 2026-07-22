"""C1b: renumbering a clip's slot must NOT change its content-bound identity.

This is the property slot_id_map violated by construction — identity followed the
slot number, so a renumber silently rebound the row to a different song. Content
binding makes identity a function of the audio bytes alone.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from labeling.als.models import ParsedClip, WarpMarkers
from labeling.export_als_to_gt import _content_bind, _load_content_catalog


def _clip(path: str) -> ParsedClip:
    return ParsedClip(
        group_name="",
        track_name="",
        path=path,
        arr_start=0.0,
        arr_end=1.0,
        loop_start=0.0,
        loop_end=1.0,
        pitch_coarse=0,
        pitch_fine=0,
        warp=WarpMarkers(points=((0.0, 0.0), (1.0, 1.0))),
    )


def test_renumber_preserves_content_identity(tmp_path: Path) -> None:
    payload = b"SAME-AUDIO-BYTES" * 200
    a = tmp_path / "028__Beatles" / "vocals.flac"
    b = tmp_path / "144__Beatles" / "vocals.flac"
    for f in (a, b):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
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
    cat = _load_content_catalog(tmp_path)
    # Identical bytes under two different slot numbers -> identical identity.
    assert _content_bind(_clip(str(a)), cat) == _content_bind(_clip(str(b)), cat)
    assert _content_bind(_clip(str(a)), cat) == (
        "beatles",
        "acappella",
        "regular",
        "content",
    )
