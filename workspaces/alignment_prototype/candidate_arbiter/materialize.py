"""Materialize a baseline placement-candidate bank from a timeline JSON."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from .adapters import candidates_from_timeline
from .io import sha256_file, write_baseline_bank, write_candidate_bank
from .schema import CandidateBankMetadata

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
    parser.add_argument(
        "--include-proposals",
        action="store_true",
        help="include serialized probe proposals in shadow mode",
    )
    args = parser.parse_args(argv)

    output = args.output
    if output is None:
        import json

        set_id = json.loads(args.timeline.read_text())["set_id"]
        output = _OUT / f"{set_id}.jsonl"
    producer_sha = args.producer_sha or _producer_sha()
    if args.include_proposals:
        import json

        timeline = json.loads(args.timeline.read_text())
        write_candidate_bank(
            output,
            CandidateBankMetadata(
                producer_sha=producer_sha,
                source_timeline_sha256=sha256_file(args.timeline),
            ),
            candidates_from_timeline(timeline),
        )
    else:
        write_baseline_bank(
            args.timeline,
            output,
            producer_sha=producer_sha,
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
