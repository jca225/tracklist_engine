"""Axis-scoped ambiguity hard-abstain (Operation Crush GT-binding completeness, A1).

The content-catalog resolver must never last-writer-wins to an arbitrary
identity when two catalogued rows share a content hash but disagree on the
identity AXIS tuple (recording_id, stem[, variant]). Getting the *work* right
but the *stem* wrong is a wrong label — exactly the poison this branch kills.

Covers both sites:
  (a) `ContentCatalog.from_entries` — direct unit test that a conflicting
      `head_hash` (and `(file_size, crc)`) key is dropped from the map
      entirely, not clobbered last-writer-wins.
  (b) `_load_content_catalog` — same scenario via a real `content_catalog.json`
      on disk, same recording_id but different stem.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from labeling.content_resolver import CatalogEntry, ContentCatalog
from labeling.export_als_to_gt import _load_content_catalog


def _catalog_json(set_dir: Path, entries: list[dict]) -> None:
    (set_dir / "content_catalog.json").write_text(
        json.dumps({"set_id": "s", "entries": entries})
    )


# --- Site 2: labeling/content_resolver.py ContentCatalog.from_entries ---


def test_from_entries_drops_head_hash_on_same_recording_different_stem():
    entries = [
        CatalogEntry(
            track_audio_id="ta1",
            recording_id="rec1",
            stem="regular",
            head_hash="hh",
        ),
        CatalogEntry(
            track_audio_id="ta2",
            recording_id="rec1",
            stem="acappella",
            head_hash="hh",
        ),
    ]
    cat = ContentCatalog.from_entries(entries)
    assert "hh" not in cat.by_head_hash


def test_from_entries_drops_size_crc_on_same_recording_different_stem():
    entries = [
        CatalogEntry(
            track_audio_id="ta1",
            recording_id="rec1",
            stem="regular",
            file_size=100,
            crc=200,
        ),
        CatalogEntry(
            track_audio_id="ta2",
            recording_id="rec1",
            stem="acappella",
            file_size=100,
            crc=200,
        ),
    ]
    cat = ContentCatalog.from_entries(entries)
    assert (100, 200) not in cat.by_size_crc


def test_from_entries_keeps_size_crc_when_axis_agrees():
    # Symmetry with test_from_entries_keeps_key_when_axis_tuple_agrees, but
    # for the by_size_crc path: a benign duplicate (identical (file_size, crc)
    # AND identical (recording_id, stem, variant)) must still resolve.
    entries = [
        CatalogEntry(
            track_audio_id="ta1",
            recording_id="rec1",
            stem="regular",
            file_size=100,
            crc=200,
        ),
        CatalogEntry(
            track_audio_id="ta2",
            recording_id="rec1",
            stem="regular",
            file_size=100,
            crc=200,
        ),
    ]
    cat = ContentCatalog.from_entries(entries)
    assert (100, 200) in cat.by_size_crc
    assert cat.by_size_crc[(100, 200)].recording_id == "rec1"
    assert cat.by_size_crc[(100, 200)].stem == "regular"


def test_from_entries_keeps_key_when_axis_tuple_agrees():
    # Benign duplicate: same (recording_id, stem) on both rows — not ambiguous.
    entries = [
        CatalogEntry(
            track_audio_id="ta1",
            recording_id="rec1",
            stem="regular",
            head_hash="hh",
        ),
        CatalogEntry(
            track_audio_id="ta2",
            recording_id="rec1",
            stem="regular",
            head_hash="hh",
        ),
    ]
    cat = ContentCatalog.from_entries(entries)
    assert "hh" in cat.by_head_hash
    assert cat.by_head_hash["hh"].recording_id == "rec1"
    assert cat.by_head_hash["hh"].stem == "regular"


# --- Site 1: labeling/export_als_to_gt._load_content_catalog ---


def test_load_content_catalog_abstains_on_same_recording_different_stem(
    tmp_path: Path,
) -> None:
    """Two rows, same content_sha256, same recording_id, DIFFERENT stem.

    This is the exact leak: bare recording_id-keyed ambiguity detection sees
    only 1 distinct recording_id and lets the hash through, then
    `from_entries` silently picks one stem. The fix keys ambiguity on the
    axis tuple, so this hash must not resolve at all.
    """
    f = tmp_path / "same_bytes.m4a"
    f.write_bytes(b"SAME-BYTES-DIFFERENT-STEM" * 100)
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    _catalog_json(
        tmp_path,
        [
            {
                "content_sha256": sha,
                "payload_sha256": None,
                "recording_id": "rec1",
                "track_audio_id": "ta1",
                "stem": "regular",
            },
            {
                "content_sha256": sha,
                "payload_sha256": None,
                "recording_id": "rec1",
                "track_audio_id": "ta2",
                "stem": "acappella",
            },
        ],
    )
    cat = _load_content_catalog(tmp_path)
    assert cat is not None
    assert sha not in cat.by_head_hash


def test_load_content_catalog_still_resolves_true_duplicate(
    tmp_path: Path,
) -> None:
    # Benign case: identical axis tuple on both rows — must still resolve.
    f = tmp_path / "dup_bytes.m4a"
    f.write_bytes(b"TRUE-DUPLICATE-BYTES" * 100)
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    _catalog_json(
        tmp_path,
        [
            {
                "content_sha256": sha,
                "payload_sha256": None,
                "recording_id": "rec1",
                "track_audio_id": "ta1",
                "stem": "regular",
            },
            {
                "content_sha256": sha,
                "payload_sha256": None,
                "recording_id": "rec1",
                "track_audio_id": "ta2",
                "stem": "regular",
            },
        ],
    )
    cat = _load_content_catalog(tmp_path)
    assert cat is not None
    assert sha in cat.by_head_hash
    assert cat.by_head_hash[sha].recording_id == "rec1"
    assert cat.by_head_hash[sha].stem == "regular"
