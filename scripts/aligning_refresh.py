#!/usr/bin/env python3
"""Chain aligning folder tag / relink / fill steps after pull."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = REPO / "venvs" / "audio" / "bin" / "python"


def _run(script: str, *args: str) -> int:
    cmd = [str(PY), str(REPO / script), *args]
    print("+", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("set_dir", type=Path, help="~/aligning/<set_id>__... folder")
    ap.add_argument(
        "--als", type=Path, default=None, help="Path to .als (default: first in folder)"
    )
    ap.add_argument("--no-stems", action="store_true", help="Skip stem subdir tagging")
    args = ap.parse_args()
    set_dir = args.set_dir.expanduser().resolve()
    if not set_dir.is_dir():
        print(f"not a directory: {set_dir}", file=sys.stderr)
        return 1
    als = args.als
    if als is None:
        cands = list(set_dir.glob("*.als"))
        if not cands:
            print("no .als in set_dir", file=sys.stderr)
            return 1
        als = cands[0]

    tag_args = [str(set_dir)]
    if not args.no_stems:
        tag_args.append("--stems")
    rc = _run("labeling/inline_tag_aligning_folder.py", *tag_args)
    if rc:
        return rc
    rc = _run("labeling/relink_als_after_tag.py", str(set_dir), "--als", str(als))
    if rc:
        return rc
    return _run("labeling/fill_als_clip_tags.py", str(set_dir), "--als", str(als))


if __name__ == "__main__":
    sys.exit(main())
