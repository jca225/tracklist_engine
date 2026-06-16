"""Fill the ``[?]`` placeholder in Ableton clip names with the real
``[NNNbpm KK]`` tag, read from each clip's own referenced audio file.

The seeded sessions name every clip ``<title> [?]`` as a placeholder for
tempo+key (BB12 was filled by hand to ``<title> [126bpm 8B]``). Each
``<AudioClip>`` references its audio via ``SampleRef/.../Path``; that file now
carries the inline tag (see inline_tag_aligning_folder.py), so we copy the
file's tag into the clip name, replacing ``[?]``.

Edits only the clip ``<Name>`` attribute text (a targeted string substitution on
the decompressed XML — no lxml re-serialization, no structural change), so none
of the deep-copy crash hazards apply. A timestamped backup is written and you
should still open the session in Live to confirm.

Usage:
    ./venvs/audio/bin/python labeling/fill_als_clip_tags.py \\
        ~/aligning/2nvzlh2k__...BB11  [--dry-run]
"""

from __future__ import annotations

import argparse
import gzip
import html
import re
import sys
from pathlib import Path

from lxml import etree

# trailing inline tag on a filename stem: [126bpm 8B] or [no-features]
_FILE_TAG = re.compile(r"(\[(?:\d+bpm [^\]]+|no-features)\])\s*$")


def xml_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def file_tag_for(path_value: str) -> str | None:
    """Extract ``[126bpm 8B]`` from a clip's referenced file path, or None.

    For ``tracks/Foo [126bpm 8B].m4a`` the tag is on the filename; for a stem
    play ``stems/Foo [126bpm 8B]/vocals.flac`` it's on the parent directory."""
    p = Path(html.unescape(path_value))
    name = p.name
    stem = name.rsplit(".", 1)[0] if "." in name else name
    for candidate in (stem, p.parent.name):
        m = _FILE_TAG.search(candidate)
        if m:
            return m.group(1)
    return None


def build_name_edits(als_path: Path) -> tuple[dict[str, str], list[str]]:
    """Return ({old_name -> new_name}, conflicts). Maps each clip whose Name has
    ``[?]`` to the same Name with ``[?]`` replaced by its file's tag."""
    root = etree.fromstring(gzip.decompress(als_path.read_bytes()))
    edits: dict[str, str] = {}
    conflicts: list[str] = []
    for clip in root.iter("AudioClip"):
        name_el = clip.find("Name")
        if name_el is None:
            continue
        val = name_el.get("Value") or ""
        if "[?]" not in val:
            continue
        path_els = clip.findall(".//SampleRef//Path")
        if not path_els:
            continue
        tag = file_tag_for(path_els[0].get("Value") or "")
        if tag is None:
            continue
        new = val.replace("[?]", tag)
        if val in edits and edits[val] != new:
            conflicts.append(val)
            continue
        edits[val] = new
    return edits, conflicts


def apply_edits(als_path: Path, edits: dict[str, str], *, dry_run: bool) -> int:
    xml = gzip.decompress(als_path.read_bytes()).decode("utf-8")
    total = 0
    for old, new in edits.items():
        needle = f'Value="{xml_escape(old)}"'
        repl = f'Value="{xml_escape(new)}"'
        hits = xml.count(needle)
        if hits:
            xml = xml.replace(needle, repl)
            total += hits
    if total and not dry_run:
        backup = als_path.with_suffix(als_path.suffix + ".prefill.bak")
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
        edits, conflicts = build_name_edits(als)
        if not edits:
            print(f"  [--] {als.name}: no '[?]' clip names to fill")
            continue
        applied = apply_edits(als, edits, dry_run=args.dry_run)
        verb = "would fill" if args.dry_run else "filled"
        print(
            f"  [{'dry' if args.dry_run else 'ok'}] {als.name}: "
            f"{verb} {applied} clip names ({len(edits)} distinct)"
        )
        for old in list(edits)[:4]:
            print(f"        {old}  ->  {edits[old]}")
        if conflicts:
            print(
                f"        !! {len(conflicts)} ambiguous names skipped: {conflicts[:3]}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
