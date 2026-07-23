"""A2: the content bind resolves a complete axis point (stem + variant).

`_content_bind` used to return `recording_id` alone, throwing away `stem`
and `variant` even though the bind resolved a specific `track_audio` row.
The GT row's `claimed_stem`/`claimed_variant` should come from the SOUND
bind when one exists, not the weaker path/manifest guess — a content-bound
clip must not silently keep a path-derived stem that disagrees with the
catalog entry it actually matched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from labeling.als.models import ParsedClip, WarpMarkers
from labeling.als import build_manifest_index
from labeling.extract._shared import _clip_row, _content_bind, _load_content_catalog


def _mapper():
    class M:  # arr==set seconds
        def arr_to_set_sec(self, a):
            return a

    return M()


def _clip(path: str) -> ParsedClip:
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


def _catalog(set_dir: Path, entries: list[dict]) -> None:
    (set_dir / "content_catalog.json").write_text(
        json.dumps({"set_id": "s", "entries": entries})
    )


def test_content_bind_returns_stem_and_variant(tmp_path: Path) -> None:
    f = tmp_path / "cand.m4a"
    f.write_bytes(b"ACAP-EXTENDED-BYTES" * 100)
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    _catalog(
        tmp_path,
        [
            {
                "content_sha256": sha,
                "payload_sha256": None,
                "recording_id": "recX",
                "track_audio_id": "ta1",
                "stem": "acappella",
                "variant": "extended",
            }
        ],
    )
    cat = _load_content_catalog(tmp_path)
    assert _content_bind(_clip(str(f)), cat) == (
        "recX",
        "acappella",
        "extended",
        "content",
    )


def test_content_bind_abstain_returns_all_none(tmp_path: Path) -> None:
    _catalog(tmp_path, [])
    cat = _load_content_catalog(tmp_path)
    assert _content_bind(_clip(str(tmp_path / "gone.m4a")), cat) == (
        None,
        None,
        None,
        "abstain",
    )


def test_content_bound_clip_row_claimed_stem_comes_from_bind(tmp_path: Path) -> None:
    # File lives under a path that path/manifest classification would read as
    # 'regular' (plain tracks/ dir, not stems/vocals/), but its BYTES are
    # catalogued as the acappella. The content bind must win.
    d = tmp_path / "tracks"
    d.mkdir(parents=True)
    f = d / "030__Some Track.m4a"
    f.write_bytes(b"BYTES-ARE-THE-ACAPPELLA" * 100)
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps({"set_id": "s", "tracks": []}))
    _catalog(
        tmp_path,
        [
            {
                "content_sha256": sha,
                "payload_sha256": None,
                "recording_id": "recY",
                "track_audio_id": "ta2",
                "stem": "acappella",
                "variant": "extended",
            }
        ],
    )
    manifest = build_manifest_index(tmp_path / "manifest.json")
    catalog = _load_content_catalog(tmp_path)

    row = _clip_row(_clip(str(f)), _mapper(), manifest, catalog)

    assert row.id_source == "content"
    assert row.claimed_stem == "acappella"  # from the BIND, not the path guess
    assert row.claimed_variant == "extended"


def test_abstaining_clip_row_falls_back_to_path_manifest_stem(tmp_path: Path) -> None:
    # No content_catalog.json entry matches -> abstain -> claimed_stem must
    # fall back to the path/manifest classification, claimed_variant='regular'.
    d = tmp_path / "stems" / "031__Y"
    d.mkdir(parents=True)
    f = d / "vocals.flac"
    f.write_bytes(b"UNCATALOGUED-VOCALS" * 100)
    (tmp_path / "manifest.json").write_text(json.dumps({"set_id": "s", "tracks": []}))
    _catalog(tmp_path, [])
    manifest = build_manifest_index(tmp_path / "manifest.json")
    catalog = _load_content_catalog(tmp_path)

    row = _clip_row(_clip(str(f)), _mapper(), manifest, catalog)

    assert row.id_source == "abstain"
    assert row.claimed_stem == "acappella"  # path/manifest guess (stems/vocals/)
    assert row.claimed_variant == "regular"
