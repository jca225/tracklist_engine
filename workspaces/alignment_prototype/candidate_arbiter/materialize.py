"""Materialize a baseline placement-candidate bank from a timeline JSON."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from .io import write_baseline_bank

_PACKAGE = Path(__file__).resolve().parent
_ALIGNMENT = _PACKAGE.parent
_REPO = _ALIGNMENT.parent.parent
_OUT = _ALIGNMENT / "out" / "candidates"


def _producer_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        cwd=_REPO,
    ).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--producer-sha", default=None)
    args = parser.parse_args(argv)

    output = args.output
    if output is None:
        import json

        set_id = json.loads(args.timeline.read_text())["set_id"]
        output = _OUT / f"{set_id}.jsonl"
    write_baseline_bank(
        args.timeline,
        output,
        producer_sha=args.producer_sha or _producer_sha(),
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
