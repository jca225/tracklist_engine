"""Train the learned identity verifier on canonical DB data.

This script reads the canonical Tracklist Engine DB and produces a trained
ensemble model (Random Forest on MERT + Essentia + fingerprint + text features)
for same-song and same-version verification.

Run on a machine with direct access to the canonical DB (pi-storage or a synced
copy):

  venvs/audio/bin/python scripts/train_identity_verifier.py \
    --db /mnt/storage/data/db/music_database.db \
    --output models/identity_verifier.pkl

The output is a pickle of (song_verifier, version_verifier) that can be loaded
by the acquisition executor.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Any

from analysis.identity_data import _load_recordings
from analysis.identity_learned import train_verifiers

_LOG = logging.getLogger("train_identity_verifier")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train the learned identity verifier")
    p.add_argument("--db", type=Path, required=True, help="Path to music_database.db")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("models/identity_verifier.pkl"),
        help="Output pickle path",
    )
    p.add_argument(
        "--min-recordings",
        type=int,
        default=10,
        help="Minimum recordings required to train",
    )
    p.add_argument("--log-level", default="INFO", help="Logging level")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.db.exists():
        _LOG.error("DB not found: %s", args.db)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)

    _LOG.info("Loading recordings from %s", args.db)
    recordings = _load_recordings(args.db)
    _LOG.info("Loaded %d recordings", len(recordings))

    if len(recordings) < args.min_recordings:
        _LOG.error(
            "Too few recordings to train (%d < %d)",
            len(recordings),
            args.min_recordings,
        )
        return 1

    song_verifier, version_verifier, metrics = train_verifiers(
        list(recordings.values())
    )

    with open(args.output, "wb") as f:
        pickle.dump(
            {"song": song_verifier, "version": version_verifier, "metrics": metrics}, f
        )
    _LOG.info("Saved model to %s", args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
