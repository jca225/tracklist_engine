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


def _catalog(set_dir: Path, entries: list[dict]) -> None:
    (set_dir / "content_catalog.json").write_text(
        json.dumps({"set_id": "s", "entries": entries})
    )


def test_binds_by_full_file_sha256(tmp_path: Path) -> None:
    f = tmp_path / "cand.m4a"
    f.write_bytes(b"CANDIDATE-BYTES" * 100)
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
            }
        ],
    )
    cat = _load_content_catalog(tmp_path)
    assert _content_bind(_clip(str(f)), cat) == (
        "recX",
        "acappella",
        "regular",
        "content",
    )


def test_abstains_when_bytes_not_in_catalog(tmp_path: Path) -> None:
    f = tmp_path / "unknown.m4a"
    f.write_bytes(b"NOT-CATALOGUED" * 50)
    _catalog(
        tmp_path,
        [
            {
                "content_sha256": "deadbeef",
                "payload_sha256": None,
                "recording_id": "recX",
                "track_audio_id": "ta1",
                "stem": "regular",
            }
        ],
    )
    cat = _load_content_catalog(tmp_path)
    assert _content_bind(_clip(str(f)), cat) == (None, None, None, "abstain")


def test_missing_file_abstains(tmp_path: Path) -> None:
    _catalog(tmp_path, [])
    cat = _load_content_catalog(tmp_path)
    assert _content_bind(_clip(str(tmp_path / "gone.m4a")), cat) == (
        None,
        None,
        None,
        "abstain",
    )


def test_collision_same_hash_different_recording_ids_abstains(tmp_path: Path) -> None:
    """Two catalog rows share content_sha256 but disagree on recording_id.

    This is the exact poison this branch kills: same audio bytes ingested
    under two different recording_ids (duplicate rip / mashup constituent /
    wrong-version). A clip whose bytes hash to that shared value must NOT
    bind confidently to either — it must abstain.
    """
    f = tmp_path / "collide.m4a"
    f.write_bytes(b"COLLIDING-BYTES" * 100)
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    _catalog(
        tmp_path,
        [
            {
                "content_sha256": sha,
                "payload_sha256": None,
                "recording_id": "recA",
                "track_audio_id": "ta1",
                "stem": "regular",
            },
            {
                "content_sha256": sha,
                "payload_sha256": None,
                "recording_id": "recB",
                "track_audio_id": "ta2",
                "stem": "regular",
            },
        ],
    )
    cat = _load_content_catalog(tmp_path)
    assert _content_bind(_clip(str(f)), cat) == (None, None, None, "abstain")


def test_same_recording_id_two_rows_still_binds(tmp_path: Path) -> None:
    """A hash mapping to a single recording_id across two rows still binds.

    e.g. a regular master and its own variant row that happen to share
    nothing ambiguous — only a DISTINCT recording_id conflict is ambiguous.
    """
    f = tmp_path / "shared.m4a"
    f.write_bytes(b"SHARED-BYTES" * 100)
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    _catalog(
        tmp_path,
        [
            {
                "content_sha256": sha,
                "payload_sha256": None,
                "recording_id": "recSame",
                "track_audio_id": "ta1",
                "stem": "regular",
            },
            {
                "content_sha256": sha,
                "payload_sha256": None,
                "recording_id": "recSame",
                "track_audio_id": "ta2",
                "stem": "regular",
            },
        ],
    )
    cat = _load_content_catalog(tmp_path)
    assert _content_bind(_clip(str(f)), cat) == (
        "recSame",
        "regular",
        "regular",
        "content",
    )


def test_binds_tagged_master_by_mdat(tmp_path: Path) -> None:
    import struct
    from labeling.identity.content_hash import mdat_sha256

    def box(t, b):
        return struct.pack(">I", len(b) + 8) + t + b

    payload = b"MASTER-AUDIO" * 200
    f = tmp_path / "154__A - B [100bpm 5B].m4a"
    f.write_bytes(
        box(b"ftyp", b"isom")
        + box(b"moov", box(b"udta", b"tag"))
        + box(b"mdat", payload)
    )
    _catalog(
        tmp_path,
        [
            {
                "content_sha256": "not-the-file",
                "payload_sha256": mdat_sha256(f),
                "recording_id": "recM",
                "track_audio_id": "ta9",
                "stem": "regular",
            }
        ],
    )
    cat = _load_content_catalog(tmp_path)
    assert _content_bind(_clip(str(f)), cat) == (
        "recM",
        "regular",
        "regular",
        "content",
    )
