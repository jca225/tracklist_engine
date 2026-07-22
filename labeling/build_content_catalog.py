"""Build a set's content_catalog.json from the canonical DB (runs on pi).

Emits {content_sha256, payload_sha256, recording_id, track_audio_id, stem} per
audio artifact a GT clip can reference: every track_audio row for the set's
recordings (content_sha256 from the DB; payload_sha256 = mdat hash for m4a), plus
demucs vocals/instrumental stems (hashed here — track_stems has no stored hash).

stdlib only; run under pi's bare python3:
    python3 -m labeling.build_content_catalog <set_id>   # prints JSON to stdout
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from labeling.content_hash import file_sha256 as _file_sha256
from labeling.content_hash import mdat_sha256 as _mdat_sha256

_DB = "/mnt/storage/data/db/music_database.db"
_M4A_EXT = (".m4a", ".mp4", ".m4b")
_STEM_TO_AXIS = {"vocals": "acappella", "instrumental": "instrumental"}


def build_catalog(conn, set_id, *, file_sha256=_file_sha256, mdat_sha256=_mdat_sha256):
    recs = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT recording_id FROM set_track_slots "
            "WHERE set_id=? AND recording_id IS NOT NULL",
            (set_id,),
        )
    ]
    entries: list[dict] = []
    if not recs:
        return {"set_id": set_id, "entries": entries}
    qmarks = ",".join("?" * len(recs))

    for taid, rid, stem, sha, path, variant in conn.execute(
        f"SELECT track_audio_id, recording_id, stem, sha256, path, variant "
        f"FROM track_audio WHERE recording_id IN ({qmarks})",
        recs,
    ):
        payload = None
        p = str(path or "")
        if p.lower().endswith(_M4A_EXT):
            try:
                payload = mdat_sha256(p)
            except OSError:
                payload = None
        entries.append(
            {
                "content_sha256": sha,
                "payload_sha256": payload,
                "recording_id": rid,
                "track_audio_id": str(taid),
                "stem": stem or "regular",
                "variant": variant or "regular",
            }
        )

    for taid, rid, stem_name, spath in conn.execute(
        f"SELECT ts.track_audio_id, ta.recording_id, ts.stem_name, ts.path "
        f"FROM track_stems ts JOIN track_audio ta ON ta.track_audio_id=ts.track_audio_id "
        f"WHERE ta.recording_id IN ({qmarks}) AND ts.stem_name IN ('vocals','instrumental')",
        recs,
    ):
        try:
            csha = file_sha256(str(spath))
        except OSError:
            continue
        entries.append(
            {
                "content_sha256": csha,
                "payload_sha256": None,
                "recording_id": rid,
                "track_audio_id": str(taid),
                "stem": _STEM_TO_AXIS.get(stem_name, stem_name),
                # Safe default: a separated stem really inherits the parent
                # track_audio's variant, but wiring that through is Task A3
                # (the stem-derivation loop refinement). Emitting a default
                # here just keeps the key present/non-crashing.
                "variant": "regular",
            }
        )
    return {"set_id": set_id, "entries": entries}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: build_content_catalog <set_id> [db_path]", file=sys.stderr)
        return 2
    set_id = argv[0]
    db = argv[1] if len(argv) > 1 else _DB
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        print(json.dumps(build_catalog(conn, set_id)))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
