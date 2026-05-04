"""Build a human-readable symlink tree at /mnt/storage/library/ pointing
into /mnt/storage/objects/ (audio) and /mnt/storage/stems/ (demucs).

Per-track folder layout:

    library/{Artist}/{Title}{ - VersionTag if remix/rework}/
        original.{ext}        -> track_audio.path
        vocals.{ext}          -> track_stems.path  WHERE stem_name='vocals'
        instrumental.{ext}    -> track_stems.path  WHERE stem_name='instrumental'

Tracks that have audio but no track_metadata entry land under
library/_unmatched/{track_id}.{ext}.

Idempotent — clears `library/` and rebuilds. Skips entries whose
underlying file doesn't exist on this filesystem (e.g. paths that
still point at Mac during the rsync transition).

Usage (on pi-storage):

    venvs/web_crawler/bin/python -m library.builder \\
        --db /mnt/storage/data/db/music_database.db \\
        --library /mnt/storage/library
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("library.builder")


import unicodedata as _unicodedata

# Filesystem name sanitization
# - NFC-normalize so combining marks like U+0308 (combining diaeresis) get
#   merged into the precomposed form (Beyoncé instead of Beyonce´). Linux
#   filesystems accept either, but if the systemd unit doesn't set
#   PYTHONUTF8=1 / LC_ALL=*.UTF-8, Python falls back to latin-1 for
#   filesystem encoding and combining marks then crash mkdir.
# - drop control chars
# - replace path separators
# - replace shell-hostile chars
# - collapse whitespace
# - trim trailing dots/spaces (Windows-friendly, harmless on POSIX)
_BAD_CHARS = re.compile(r'[\x00-\x1f/\\:*?"<>|]')


def sanitize_component(s: str, max_len: int = 200) -> str:
    s = _unicodedata.normalize("NFC", s)
    s = _BAD_CHARS.sub("-", s)
    s = re.sub(r"\s+", " ", s).strip(" .")
    if len(s) > max_len:
        s = s[:max_len].rstrip(" .")
    return s or "_"


def parse_artists(artists_json: str | None) -> list[str]:
    if not artists_json:
        return []
    try:
        v = json.loads(artists_json)
        return [str(a) for a in v if a]
    except Exception:
        return []


def join_artists(artists: list[str]) -> str:
    """Match 1001tracklists' '&' style for multi-artist tracks."""
    if not artists:
        return ""
    return " & ".join(a.strip() for a in artists if a.strip())


def track_dir_name(title: str | None, version_tag: str | None) -> str:
    base = title or "_untitled"
    if version_tag and version_tag.lower() not in ("none", ""):
        base = f"{base} - {version_tag}"
    return sanitize_component(base)


def make_symlink(src: str | Path, dst: Path) -> bool:
    """Create dst -> src. Returns True on success.

    src must exist on this filesystem. dst is overwritten.
    """
    src = Path(src)
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        try:
            dst.unlink()
        except IsADirectoryError:
            shutil.rmtree(dst)
    try:
        dst.symlink_to(src)
        return True
    except OSError as e:
        log.warning("symlink failed %s -> %s: %s", dst, src, e)
        return False


def build(db_path: Path, library_root: Path) -> dict[str, int]:
    log.info("opening %s", db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    log.info("rebuilding %s from scratch", library_root)
    if library_root.exists():
        shutil.rmtree(library_root)
    library_root.mkdir(parents=True, exist_ok=True)
    (library_root / "_unmatched").mkdir(exist_ok=True)

    # Pull every audio row + (left-join) its metadata.
    # Tracks WITHOUT metadata still get an _unmatched/ entry.
    audio_rows = conn.execute("""
        SELECT
            ta.track_audio_id,
            ta.track_id,
            ta.path           AS audio_path,
            ta.codec          AS audio_codec,
            ta.variant_tag    AS variant_tag,
            tm.title          AS title,
            tm.artists_json   AS artists_json,
            tm.version_tag    AS version_tag
        FROM track_audio ta
        LEFT JOIN track_metadata tm USING(track_id)
    """).fetchall()
    log.info("track_audio rows: %d (with metadata: %d)",
             len(audio_rows),
             sum(1 for r in audio_rows if r["title"]))

    # Stems index: track_audio_id -> {stem_name: path}
    stems_idx: dict[int, dict[str, str]] = defaultdict(dict)
    for r in conn.execute("""
        SELECT track_audio_id, stem_name, path
        FROM track_stems
        WHERE stem_name IN ('vocals', 'instrumental')
    """):
        stems_idx[r["track_audio_id"]][r["stem_name"]] = r["path"]

    # Conflict tracking: collisions on (Artist, Title) folder name across
    # different track_ids. We disambiguate by appending the track_id.
    folder_owners: dict[tuple[str, str], str] = {}
    counts = {
        "matched": 0,
        "unmatched": 0,
        "skipped_missing_path": 0,
        "stems_linked": 0,
        "conflicts": 0,
    }

    for row in audio_rows:
        audio_path = row["audio_path"]
        if not audio_path or not Path(audio_path).exists():
            counts["skipped_missing_path"] += 1
            continue

        ext = Path(audio_path).suffix.lstrip(".") or "bin"

        if not row["title"]:
            # Unmatched: simple flat name keyed on track_id
            dst = library_root / "_unmatched" / f"{row['track_id']}.{ext}"
            if make_symlink(audio_path, dst):
                counts["unmatched"] += 1
            continue

        artist = sanitize_component(join_artists(parse_artists(row["artists_json"])) or "_unknown_artist")
        title_dir = track_dir_name(row["title"], row["version_tag"])

        # Disambiguate on (artist, title) collision across distinct track_ids
        key = (artist, title_dir)
        owner = folder_owners.get(key)
        if owner is not None and owner != row["track_id"]:
            title_dir = f"{title_dir} (track_id={row['track_id']})"
            counts["conflicts"] += 1
        else:
            folder_owners[key] = row["track_id"]

        track_dir = library_root / artist / title_dir

        # original
        if make_symlink(audio_path, track_dir / f"original.{ext}"):
            counts["matched"] += 1

        # stems (forward-compatible: empty index today, populated post-demucs)
        for stem_name, stem_path in stems_idx.get(row["track_audio_id"], {}).items():
            if not Path(stem_path).exists():
                continue
            stem_ext = Path(stem_path).suffix.lstrip(".") or "wav"
            if make_symlink(stem_path, track_dir / f"{stem_name}.{stem_ext}"):
                counts["stems_linked"] += 1

    conn.close()
    log.info("DONE — %s", counts)
    return counts


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--db", required=True, type=Path)
    p.add_argument("--library", required=True, type=Path,
                   help="Root of the library symlink tree, e.g. /mnt/storage/library")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.db.exists():
        print(f"DB not found: {args.db}", file=sys.stderr)
        return 2

    build(args.db, args.library)
    return 0


if __name__ == "__main__":
    sys.exit(main())
