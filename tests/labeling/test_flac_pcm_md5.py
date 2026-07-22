"""B3 — flac_pcm_md5: the FLAC decoded-PCM MD5 payload key (spec v2.3 / P11).

Tag-invariant and lossless-re-encode-invariant: the stem population's free
sound channel. Verified non-null before use (some encoders leave it zero).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from labeling.content_hash import flac_pcm_md5

_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "als"
    / "audio"
    / "stems"
    / "001w1__Bastille - Good Grief (Don Diablo Remix)"
    / "instrumental.flac"
)


def _minimal_flac(md5: bytes) -> bytes:
    """A minimal valid-enough FLAC: magic + a STREAMINFO block (type 0, last)
    whose trailing 16 bytes are `md5`. Enough for flac_pcm_md5 to parse."""
    assert len(md5) == 16
    streaminfo = bytes(18) + md5  # 18 bytes of header fields we don't read + MD5
    assert len(streaminfo) == 34
    block_header = bytes([0x80]) + (34).to_bytes(3, "big")  # last-block, type 0, len 34
    return b"fLaC" + block_header + streaminfo


def test_reads_known_md5(tmp_path: Path):
    md5 = bytes(range(16))  # 000102…0f
    p = tmp_path / "s.flac"
    p.write_bytes(_minimal_flac(md5))
    assert flac_pcm_md5(p) == md5.hex()


def test_all_zero_md5_is_none(tmp_path: Path):
    p = tmp_path / "z.flac"
    p.write_bytes(_minimal_flac(b"\x00" * 16))
    assert flac_pcm_md5(p) is None


def test_non_flac_is_none(tmp_path: Path):
    p = tmp_path / "not.flac"
    p.write_bytes(b"ID3\x04nope this is not a flac file at all")
    assert flac_pcm_md5(p) is None


def test_truncated_is_none(tmp_path: Path):
    p = tmp_path / "trunc.flac"
    p.write_bytes(b"fLaC" + bytes([0x80]) + (34).to_bytes(3, "big") + b"\x00" * 4)
    assert flac_pcm_md5(p) is None


@pytest.mark.skipif(not _FIXTURE.exists(), reason="fixture FLAC absent")
def test_real_fixture_flac_parses(tmp_path: Path):
    """On a real FLAC the result is either a 32-hex string or None (unset MD5) —
    never a crash, and stable across a copy (tag-invariance smoke)."""
    v = flac_pcm_md5(_FIXTURE)
    assert v is None or (isinstance(v, str) and len(v) == 32 and int(v, 16) >= 0)
    # invariance: copying the bytes must not change the decoded-PCM MD5
    copy = tmp_path / "copy.flac"
    copy.write_bytes(_FIXTURE.read_bytes())
    assert flac_pcm_md5(copy) == v
