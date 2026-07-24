"""Content-addressed clip→identity resolver (Operation Crush §9 — slice 2).

The resolver has NO fallback ladder: it binds a clip to identity by content
(OriginalFileSize+Crc → head_hash) and, on a miss, abstains loudly (Err) — it
never guesses by filename/slot. This is the mechanism that kills the poison class.
"""

from __future__ import annotations

from labeling.als.models import ParsedClip, WarpMarkers
from labeling.identity.content_resolver import (
    CatalogEntry,
    ContentCatalog,
    resolve_clip_identity,
)

_CAT = ContentCatalog.from_entries(
    [
        CatalogEntry(
            track_audio_id="ta1",
            recording_id="rec1",
            stem="regular",
            file_size=100,
            crc=200,
            head_hash="abc",
        )
    ]
)

_CAT_ACAP_EXT = ContentCatalog.from_entries(
    [
        CatalogEntry(
            track_audio_id="ta2",
            recording_id="rec2",
            stem="acappella",
            variant="extended",
            file_size=300,
            crc=400,
            head_hash="def",
        )
    ]
)


def _clip(file_size=None, crc=None, path="x.flac") -> ParsedClip:
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
        file_size=file_size,
        crc=crc,
    )


def test_resolves_by_size_crc():
    r = resolve_clip_identity(_clip(file_size=100, crc=200), _CAT)
    assert r.is_ok()
    assert r.value.track_audio_id == "ta1"
    assert r.value.recording_id == "rec1"
    assert r.value.stem == "regular"
    assert r.value.matched_by == "size_crc"


def test_abstains_loudly_on_no_content_match():
    # A filename/slot that "looks like" ta1 must NOT resolve when the content
    # (size,crc) doesn't match — no guess, an Err diagnostic instead.
    r = resolve_clip_identity(
        _clip(file_size=999, crc=999, path="001__ta1-lookalike.flac"), _CAT
    )
    assert not r.is_ok()
    assert r.error.file_size == 999


def test_head_hash_fallback_when_size_crc_absent():
    # size/crc missing on the clip, but the resolved file hashes to a known head.
    r = resolve_clip_identity(
        _clip(file_size=None, crc=None, path="x.flac"),
        _CAT,
        head_hash_of=lambda _p: "abc",
    )
    assert r.is_ok()
    assert r.value.track_audio_id == "ta1"
    assert r.value.matched_by == "head_hash"


def test_head_hash_miss_still_abstains():
    r = resolve_clip_identity(
        _clip(file_size=None, crc=None, path="x.flac"),
        _CAT,
        head_hash_of=lambda _p: "no-such-hash",
    )
    assert not r.is_ok()


def test_resolved_identity_carries_stem_and_variant_together():
    # A2: the bind must resolve a COMPLETE axis point (recording_id, stem,
    # variant) — not stem alone with variant thrown away out-of-band.
    r = resolve_clip_identity(_clip(file_size=300, crc=400), _CAT_ACAP_EXT)
    assert r.is_ok()
    assert r.value.stem == "acappella"
    assert r.value.variant == "extended"
