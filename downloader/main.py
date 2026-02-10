from __future__ import annotations

import sys
from pathlib import Path

# Allows `python downloader/main.py` execution from repo root.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from downloader.cli import main as cli_main, parse_args
from downloader.pipeline import download_from_music_db


__all__ = ["download_from_music_db", "parse_args", "main"]


def main() -> int:
    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
