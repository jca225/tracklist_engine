"""content_hash: full-file sha256 == track_audio.sha256; mdat hash is tag-invariant."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

from labeling.content_hash import file_sha256, mdat_sha256


def _write_min_mp4(path: Path, mdat_payload: bytes, tag_blob: bytes) -> None:
    # ftyp, then a 'moov'->'udta' carrying tag_blob, then 'mdat' with payload.
    def box(typ: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body) + 8) + typ + body

    ftyp = box(b"ftyp", b"isom" + b"\x00\x00\x02\x00" + b"isomiso2")
    udta = box(b"udta", box(b"\xa9nam", tag_blob))
    moov = box(b"moov", udta)
    mdat = box(b"mdat", mdat_payload)
    path.write_bytes(ftyp + moov + mdat)


def test_file_sha256_matches_hashlib(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello world" * 1000)
    assert file_sha256(p) == hashlib.sha256(p.read_bytes()).hexdigest()


def test_mdat_hash_is_tag_invariant(tmp_path: Path) -> None:
    payload = b"AUDIO-PAYLOAD-BYTES" * 500
    a = tmp_path / "a.m4a"
    b = tmp_path / "b.m4a"
    _write_min_mp4(a, payload, tag_blob=b"tags-before")
    _write_min_mp4(b, payload, tag_blob=b"COMPLETELY-DIFFERENT-LONGER-TAGS")
    # Different container metadata -> different full-file hash ...
    assert file_sha256(a) != file_sha256(b)
    # ... but identical audio payload -> identical mdat hash.
    assert mdat_sha256(a) == mdat_sha256(b)
    assert mdat_sha256(a) == hashlib.sha256(payload).hexdigest()


def test_mdat_hash_none_for_non_mp4(tmp_path: Path) -> None:
    p = tmp_path / "not.mp4"
    p.write_bytes(b"fLaC" + b"\x00" * 100)
    assert mdat_sha256(p) is None
