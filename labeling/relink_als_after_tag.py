"""Relink an Ableton .als after inline_tag_aligning_folder.py renamed the audio.

Renaming ``tracks/Foo.m4a`` -> ``tracks/Foo [126bpm 8B].m4a`` (and the matching
``stems/Foo/`` -> ``stems/Foo [126bpm 8B]/``) leaves the live session pointing at
the old names, so every clip shows **offline**. This rewrites the session's file
references (``Path`` / ``RelativePath`` / ``Name``) from the old name to the new
tagged name so Live finds the audio again.

It edits the .als as text (gunzip -> string-replace -> gzip), so it does NOT
touch device/automation state — none of the deep-copy crash hazards apply. Still:
**open the session in Live afterwards to confirm**, and a timestamped backup of
the original .als is written next to it.

The tag is inserted just before the extension (files) or at the end of the dir
name (stems), so each rename is the substring edit ``OLD<.ext|/>`` ->
``OLD [tag]<.ext|/>``. We derive OLD by stripping the tag off the on-disk name.

Usage:
    ./venvs/audio/bin/python labeling/relink_als_after_tag.py \\
        ~/aligning/2nvzlh2k__...BB11  [--dry-run]
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from labeling.als_io import strip_user_tags

AUDIO_EXTS = {".m4a", ".mp3", ".wav"}


def xml_escape(s: str) -> str:
    """Match Ableton's attribute-value escaping (& < > only; quotes are double,
    apostrophes stay literal)."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_renames(folder: Path) -> list[tuple[str, str]]:
    """Return (old_substring, new_substring) edits for every tagged track file
    and stem dir. Files edit ``OLD.ext`` -> ``NEW.ext``; dirs edit ``OLD/`` ->
    ``NEW/``."""
    edits: list[tuple[str, str]] = []

    tracks = folder / "tracks"
    if tracks.is_dir():
        for f in tracks.iterdir():
            if f.suffix.lower() not in AUDIO_EXTS or not f.is_file():
                continue
            old_stem = strip_user_tags(f.stem)
            if old_stem == f.stem:
                continue  # untagged — nothing to relink
            edits.append((f"{old_stem}{f.suffix}", f"{f.stem}{f.suffix}"))

    stems = folder / "stems"
    if stems.is_dir():
        for d in stems.iterdir():
            if not d.is_dir():
                continue
            old_name = strip_user_tags(d.name)
            if old_name == d.name:
                continue
            edits.append((f"{old_name}/", f"{d.name}/"))

    return edits


def relink(als_path: Path, edits: list[tuple[str, str]], *, dry_run: bool) -> int:
    xml = gzip.decompress(als_path.read_bytes()).decode("utf-8")
    total = 0
    for old, new in edits:
        for o, n in ((old, new), (xml_escape(old), xml_escape(new))):
            if o == n:
                continue
            hits = xml.count(o)
            if hits:
                xml = xml.replace(o, n)
                total += hits
    if total and not dry_run:
        backup = als_path.with_suffix(als_path.suffix + ".prerelink.bak")
        if not backup.exists():
            backup.write_bytes(als_path.read_bytes())
        als_path.write_bytes(gzip.compress(xml.encode("utf-8")))
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", help="Aligning folder containing the .als")
    ap.add_argument("--als", help="Specific .als (default: every *.als in folder)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    folder = Path(args.folder).expanduser()
    if not folder.is_dir():
        print(f"ERROR: not a dir: {folder}", file=sys.stderr)
        return 1

    edits = build_renames(folder)
    if not edits:
        print("No tagged files found — nothing to relink.")
        return 0
    print(f"Derived {len(edits)} rename edits from on-disk tags.")

    als_files = (
        [Path(args.als).expanduser()] if args.als else sorted(folder.glob("*.als"))
    )
    if not als_files:
        print(f"No .als in {folder}", file=sys.stderr)
        return 1

    for als in als_files:
        if not als.is_file():
            print(f"  [skip] missing: {als.name}")
            continue
        refs = relink(als, edits, dry_run=args.dry_run)
        verb = "would update" if args.dry_run else "updated"
        print(
            f"  [{'dry' if args.dry_run else 'ok'}] {als.name}: {verb} {refs} file references"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
