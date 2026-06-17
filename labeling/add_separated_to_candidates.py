"""Expose each song's DL-separated stems alongside the online candidates.

fetch_candidate_stems.py drops online acappella/instrumental candidates into
``stems/<song>/candidates/<layer>/``. The Demucs/RoFormer-separated stems
(``vocals.flac`` / ``instrumental.flac``) are the *baseline* an annotator A/Bs
those candidates against — but they live one level up, in the stem folder. This
links them INTO ``candidates/<layer>/`` as ``separated__<layer>.flac`` so every
audition option for a layer sits in one place.

Symlink (not copy): the FLAC stems are large and already on disk; Ableton on
macOS follows symlinks. Re-runnable (skips links already present).

Usage:
    ./venvs/audio/bin/python labeling/add_separated_to_candidates.py \\
        ~/aligning/2nvzlh2k__...BB11  [--copy] [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

LAYERS = ("vocals", "instrumental")


def link_folder(stem_folder: Path, *, copy: bool, dry_run: bool) -> int:
    n = 0
    for layer in LAYERS:
        src = stem_folder / f"{layer}.flac"
        if not src.is_file():
            continue
        dest_dir = stem_folder / "candidates" / layer
        dest = dest_dir / f"separated__{layer}.flac"
        if dest.exists() or dest.is_symlink():
            continue
        if dry_run:
            print(f"  [dry] {stem_folder.name}/candidates/{layer}/{dest.name}")
            n += 1
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        if copy:
            shutil.copy2(src, dest)
        else:
            # relative symlink so the folder stays portable if moved wholesale
            os.symlink(os.path.relpath(src, dest_dir), dest)
        print(f"  [ok]  {stem_folder.name}/candidates/{layer}/{dest.name}")
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", help="Aligning folder (contains stems/)")
    ap.add_argument("--copy", action="store_true", help="Copy instead of symlink")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stems = Path(args.folder).expanduser() / "stems"
    if not stems.is_dir():
        print(f"ERROR: no stems/ in {args.folder}", file=sys.stderr)
        return 1

    linked = folders = 0
    for d in sorted(stems.iterdir()):
        if not d.is_dir():
            continue
        added = link_folder(d, copy=args.copy, dry_run=args.dry_run)
        if added:
            folders += 1
            linked += added
    verb = "would link" if args.dry_run else ("copied" if args.copy else "linked")
    print(f"\n{verb} {linked} separated stems across {folders} song folders")
    return 0


if __name__ == "__main__":
    sys.exit(main())
