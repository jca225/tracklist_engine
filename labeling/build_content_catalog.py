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
    # Widened to the pull's own resolution (labeling/pull_set_for_alignment.py
    # `wanted` CTE, set_track_slots arm): COALESCE(recording_id, track_id) so a
    # legacy/Rvmor-gap slot (NULL recording_id, track_id-only identity) is not
    # silently dropped from the catalog's scope (P10). Does NOT add the pull's
    # dj_set_track_media_links UNION arm — out of scope for this builder.
    recs = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT COALESCE(recording_id, track_id) FROM set_track_slots "
            "WHERE set_id=? AND COALESCE(recording_id, track_id) IS NOT NULL",
            (set_id,),
        )
    ]
    entries: list[dict] = []
    if not recs:
        return {"set_id": set_id, "entries": entries}
    qmarks = ",".join("?" * len(recs))

    # Dual-key match (mirrors the pull's `ta.track_id IN wanted OR
    # ta.recording_id IN wanted`): a legacy track_audio row may carry the
    # identity only on track_id (recording_id NULL). Emit the COALESCEd id so
    # the entry binds to a non-null identity consistent with set_track_slots.
    for taid, rid, stem, sha, path, variant in conn.execute(
        f"SELECT track_audio_id, COALESCE(recording_id, track_id) AS rid, "
        f"stem, sha256, path, variant "
        f"FROM track_audio WHERE track_id IN ({qmarks}) OR recording_id IN ({qmarks})",
        recs + recs,
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
                "kind": "master",
            }
        )

    # A separated (demucs/roformer) stem is only a valid acappella/instrumental
    # catalog entry when its parent track_audio row is the regular master
    # (ta.stem='regular'). If the parent is itself an acappella/instrumental
    # master, its separated residual is not the recording's real acappella/
    # instrumental — cataloguing it would be a wrong-stem-axis entry (P14).
    for taid, rid, stem_name, spath, variant in conn.execute(
        f"SELECT ts.track_audio_id, COALESCE(ta.recording_id, ta.track_id) AS rid, "
        f"ts.stem_name, ts.path, ta.variant "
        f"FROM track_stems ts JOIN track_audio ta ON ta.track_audio_id=ts.track_audio_id "
        f"WHERE (ta.track_id IN ({qmarks}) OR ta.recording_id IN ({qmarks})) "
        f"AND ta.stem='regular' AND ts.stem_name IN ('vocals','instrumental')",
        recs + recs,
    ):
        # Strict lookup (not a raw passthrough, P15): component stems
        # (drums/bass/other) have no point in {regular,acappella,instrumental}
        # and must be excluded, not emitted under their raw name. This is
        # belt-and-suspenders alongside the WHERE clause above.
        axis = _STEM_TO_AXIS.get(stem_name)
        if axis is None:
            continue
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
                "stem": axis,
                # A separation preserves the parent master's length, so the
                # separated stem inherits the parent track_audio's variant
                # (mirrors the master loop's `variant or "regular"`).
                "variant": variant or "regular",
                "kind": "separated",
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
