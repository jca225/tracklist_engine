"""Rename the audio files in an aligning folder to carry an inline
``[NNNbpm KK]`` tempo+key tag — the "annotator rename convention" from
labeling/CLAUDE.md, so Ableton's clip browser shows tempo/key at a glance.

This is the offline counterpart to ``tag_aligning_folder.py``: that tool writes
the iTunes metadata atoms (``tmpo`` + ``initialkey``); this one reflects those
*already-present* atoms into the filename. No pi-storage query needed.

For each ``tracks/*.m4a`` (also .mp3/.wav):
  - already tagged (``[123bpm 8A]`` or ``[no-features]``) -> skipped
  - has tmpo + initialkey                                 -> ``... [<bpm>bpm <KK>]``
  - missing either                                        -> ``... [no-features]``

With ``--stems`` the matching ``stems/<same-name>/`` subdir gets the same tag
(stem subdirs hold no metadata of their own — the tag is copied from the track).

Usage:
    ./venvs/audio/bin/python labeling/inline_tag_aligning_folder.py \\
        ~/aligning/1rfb0yl9__Disco\\ Lines... --stems --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from mutagen.mp4 import MP4

AUDIO_EXTS = {".m4a", ".mp3", ".wav"}
# Matches an existing user tag so we never double-tag: [84bpm 6B] or [no-features]
_TAG_RE = re.compile(r"\s*\[(?:\d+bpm\s+\S+|no-features)\]$")


def read_bpm_key(path: Path) -> tuple[int | None, str | None]:
    """Read (bpm, camelot_key) from an .m4a's iTunes atoms. (None, None) for
    non-m4a or untagged files."""
    if path.suffix.lower() != ".m4a":
        return None, None
    audio = MP4(path)
    tmpo = audio.get("tmpo")
    bpm = int(tmpo[0]) if tmpo else None
    raw = audio.get("----:com.apple.iTunes:initialkey")
    key = raw[0].decode("utf-8") if raw else None
    return bpm, key


def tag_for(bpm: int | None, key: str | None) -> str:
    return f"[{bpm}bpm {key}]" if (bpm is not None and key) else "[no-features]"


def tagged_name(stem: str, bpm: int | None, key: str | None) -> str:
    return f"{stem} {tag_for(bpm, key)}"


def rename(src: Path, dst: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    src.rename(dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", help="Path to an aligning folder (contains tracks/)")
    ap.add_argument(
        "--stems",
        action="store_true",
        help="Also rename matching stems/<name>/ subdirs",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Show renames without performing them"
    )
    args = ap.parse_args()

    folder = Path(args.folder).expanduser()
    tracks_dir = folder / "tracks"
    if not tracks_dir.is_dir():
        print(f"ERROR: no tracks/ dir in {folder}", file=sys.stderr)
        return 1
    stems_dir = folder / "stems"

    renamed = skipped = nofeat = 0
    # name (without extension) of every renamed track -> its inline tag,
    # so we can apply the identical tag to the matching stems subdir.
    tag_by_stemname: dict[str, str] = {}

    for src in sorted(tracks_dir.iterdir()):
        if src.suffix.lower() not in AUDIO_EXTS or not src.is_file():
            continue
        if _TAG_RE.search(src.stem):
            skipped += 1
            continue
        bpm, key = read_bpm_key(src)
        tag = tag_for(bpm, key)
        if tag == "[no-features]":
            nofeat += 1
        new_stem = f"{src.stem} {tag}"
        dst = src.with_name(new_stem + src.suffix)
        tag_by_stemname[src.stem] = tag
        prefix = "[dry]" if args.dry_run else "[ok] "
        print(f"  {prefix} {src.name}  ->  {dst.name}")
        rename(src, dst, dry_run=args.dry_run)
        renamed += 1

    stem_renamed = 0
    if args.stems and stems_dir.is_dir():
        for sub in sorted(stems_dir.iterdir()):
            if not sub.is_dir() or _TAG_RE.search(sub.name):
                continue
            tag = tag_by_stemname.get(sub.name)
            if tag is None:
                continue  # no matching track file (or already-tagged track)
            dst = sub.with_name(f"{sub.name} {tag}")
            if dst.exists():
                continue  # a tagged copy already there (re-pull artifact)
            prefix = "[dry]" if args.dry_run else "[ok] "
            print(f"  {prefix} stems/{sub.name}  ->  {dst.name}")
            rename(sub, dst, dry_run=args.dry_run)
            stem_renamed += 1

    verb = "would rename" if args.dry_run else "renamed"
    print(
        f"\nTracks {verb}: {renamed} (no-features: {nofeat}), "
        f"already-tagged skipped: {skipped}, stem dirs {verb}: {stem_renamed}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
