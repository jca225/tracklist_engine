#!/usr/bin/env python3
"""Batch-ingest winner files from stems/*/candidates/ into pi-storage."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = REPO / "venvs" / "audio" / "bin" / "python"


def _winners(set_dir: Path) -> list[tuple[Path, str, str]]:
    """Yield (file, track_id, role) from candidate sidecars or naming."""
    out: list[tuple[Path, str, str]] = []
    for cand_dir in set_dir.glob("stems/*/candidates"):
        winner = cand_dir / "WINNER.txt"
        if winner.is_file():
            lines = winner.read_text().strip().splitlines()
            fname = lines[0].strip() if lines else ""
            tid = lines[1].strip() if len(lines) > 1 else ""
            role = lines[2].strip() if len(lines) > 2 else "acappella"
            fpath = cand_dir / fname
            if fpath.is_file() and tid:
                out.append((fpath, tid, role))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set-dir", type=Path, required=True)
    ap.add_argument("--set-id", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reason", default="source:candidate_winner|quality:human_pick")
    args = ap.parse_args()
    set_dir = args.set_dir.expanduser().resolve()
    n = 0
    for fpath, tid, role in _winners(set_dir):
        cmd = [
            str(PY),
            str(REPO / "scripts" / "ingest_stem_url.py"),
            "--file",
            str(fpath),
            "--track-id",
            tid,
            "--role",
            role,
            "--set-id",
            args.set_id,
            "--reason",
            args.reason,
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        print("+", " ".join(cmd))
        if not args.dry_run:
            subprocess.call(cmd)
        n += 1
    print(f"Processed {n} winner(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
