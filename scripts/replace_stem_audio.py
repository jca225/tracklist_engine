#!/usr/bin/env python3
"""Replace an acappella or instrumental track_audio row (quality / identity fix).

Thin wrapper around replace_track_audio.py for the stem-axis labeling workflow:
inherits stem from the retired row, defaults correction ledger axis to stem,
and runs the chromaprint identity check after a successful replace.

Typical flow (on pi-storage after auditioning YouTube candidates):

    venvs/audio/bin/python scripts/replace_stem_audio.py \\
        --track-audio-id 4011 \\
        --url 'https://www.youtube.com/watch?v=...' \\
        --set-id 2vpur281 --position 022 \\
        --reason 'quality:good|identity:OK|yt studio acapella vs ytm muddy'

See docs/stem_discovery_playbook.md.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts import acquire_variant as av
from scripts import replace_track_audio as rta


def _track_id_from_taid(db_path: Path, taid: int) -> str | None:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT track_id FROM track_audio WHERE track_audio_id = ?",
            (taid,),
        ).fetchone()
    return row[0] if row else None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--track-audio-id", type=int, required=True,
                   help="track_audio_id of the stem row to replace")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--url", help="YouTube / YT Music URL")
    g.add_argument("--file", type=Path, help="Local audio file")
    p.add_argument("--player-id", default=None,
                   help="player_id for --file (defaults to filename stem)")
    p.add_argument("--stem", choices=rta._STEM_CHOICES, default=None,
                   help="Override stem (default: inherit from retired row)")
    p.add_argument("--set-id", default=None)
    p.add_argument("--position", default=None)
    p.add_argument("--reason", required=True,
                   help="Structured note, e.g. quality:good|identity:OK|...")
    p.add_argument("--no-identity-check", action="store_true")
    p.add_argument("--no-log", action="store_true")
    p.add_argument("--no-promote-reference", action="store_true")
    p.add_argument("--db", type=Path,
                   default=Path(os.environ.get("TRACKLIST_DB",
                                               "/mnt/storage/data/db/music_database.db")))
    p.add_argument("--audio-root", type=Path,
                   default=Path(os.environ.get("TRACKLIST_AUDIO_ROOT", "/mnt/storage")))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = _parse_args(argv)
    old = rta.snapshot_row(a.db, a.track_audio_id)
    if old is None:
        print(f"track_audio_id {a.track_audio_id} not found", file=sys.stderr)
        return 1

    track_id = _track_id_from_taid(a.db, a.track_audio_id)
    if track_id is None:
        print(f"track_audio_id {a.track_audio_id} not found", file=sys.stderr)
        return 1

    stem = rta._resolve_stem_for_replace(a.stem, old)
    if stem == "regular":
        print(
            "warning: retiring a regular stem row — "
            "use replace_track_audio for version fixes",
            file=sys.stderr,
        )

    rta_args = [
        "--track-audio-id", str(a.track_audio_id),
        "--axis", "stem",
        "--reason", a.reason,
        "--db", str(a.db),
        "--audio-root", str(a.audio_root),
    ]
    if a.set_id:
        rta_args.extend(["--set-id", a.set_id])
    if a.position:
        rta_args.extend(["--position", a.position])
    if a.stem:
        rta_args.extend(["--stem", a.stem])
    if a.no_log:
        rta_args.append("--no-log")
    if a.no_promote_reference:
        rta_args.append("--no-promote-reference")
    if a.url:
        rta_args.extend(["--url", a.url])
    else:
        rta_args.extend(["--file", str(a.file)])
        if a.player_id:
            rta_args.extend(["--player-id", a.player_id])

    rc = rta.main(rta_args)
    if rc == 0 and not a.no_identity_check:
        av._identity_check(a.db, track_id, stem)
    return rc


if __name__ == "__main__":
    sys.exit(main())
