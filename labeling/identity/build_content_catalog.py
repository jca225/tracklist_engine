"""Build a set's content_catalog.json from the canonical DB (runs on pi).

Emits {content_sha256, payload_sha256, recording_id, track_audio_id, stem} per
audio artifact a GT clip can reference: every track_audio row for the set's
recordings (content_sha256 from the DB; payload_sha256 = tag-invariant mdat hash
for m4a / decoded-PCM MD5 for FLAC), plus demucs vocals/instrumental stems
(hashed here — track_stems has no stored hash; FLAC stems also get a PCM-MD5
payload key so a re-encoded/re-containered stem still binds — spec v2.3/P11).

C2: also emits certificate-gated *historical* generations from
``content_history`` (valid=1 only): payload-equality or derivation
(parent_content_sha256 matches the current regular master's content hash).
Historical rows stamp ``id_source='historical-content'`` + ``cert``.

stdlib only; run under pi's bare python3:
    python3 -m labeling.identity.build_content_catalog <set_id>   # prints JSON to stdout
"""

from __future__ import annotations

import json
import sqlite3
import sys

from labeling.identity.content_hash import file_sha256 as _file_sha256
from labeling.identity.content_hash import flac_pcm_md5 as _flac_pcm_md5
from labeling.identity.content_hash import mdat_sha256 as _mdat_sha256

_DB = "/mnt/storage/data/db/music_database.db"
_M4A_EXT = (".m4a", ".mp4", ".m4b")
_STEM_TO_AXIS = {"vocals": "acappella", "instrumental": "instrumental"}


def _payload_for(path, mdat_sha256, flac_pcm_md5):
    """Tag-invariant payload key by container: mdat for mp4, decoded-PCM MD5 for
    FLAC, else None. The FLAC key gives the stem population (payload was always
    None before) a free sound channel that survives re-encode/re-container
    (spec v2.3/P11). OSError → None (matches the mdat call site's guard)."""
    p = str(path or "")
    lp = p.lower()
    try:
        if lp.endswith(_M4A_EXT):
            return mdat_sha256(p)
        if lp.endswith(".flac"):
            return flac_pcm_md5(p)
    except OSError:
        return None
    return None


def _emit_historical(conn, recs: list[str], live_entries: list[dict]) -> list[dict]:
    """C2 — certificate-gated historical generations from content_history.

    Emit only with a sound certificate (v4.1):
    * payload — historical payload_sha256 equals a live sound entry's payload
    * derivation — kind='separated' and parent_content_sha256 equals the
      current regular master's content_sha256 for that recording
    Tombstoned (valid=0) and uncertified generations are skipped.
    """
    if not recs:
        return []
    live_payloads = {
        e["payload_sha256"] for e in live_entries if e.get("payload_sha256")
    }
    # Current regular master content hash per recording (derivation parent).
    parent_master: dict[str, str] = {}
    for e in live_entries:
        if (
            e.get("kind") == "master"
            and e.get("stem") == "regular"
            and e.get("content_sha256")
            and e.get("recording_id")
        ):
            parent_master[str(e["recording_id"])] = str(e["content_sha256"])

    live_content = {
        e.get("content_sha256") for e in live_entries if e.get("content_sha256")
    }

    qmarks = ",".join("?" * len(recs))
    try:
        rows = conn.execute(
            f"""
            SELECT recording_id, stem, variant, kind, track_audio_id,
                   content_sha256, payload_sha256,
                   parent_content_sha256, parent_payload_sha256
            FROM content_history
            WHERE recording_id IN ({qmarks}) AND valid = 1
            """,
            recs,
        ).fetchall()
    except sqlite3.OperationalError:
        # Table / parent_* columns not yet migrated — no historical emit.
        return []

    out: list[dict] = []
    for (
        rid,
        stem,
        variant,
        kind,
        taid,
        csha,
        psha,
        parent_c,
        _parent_p,
    ) in rows:
        if not csha:
            continue
        if csha in live_content:
            continue  # already bound by live catalog entry
        cert: str | None = None
        if psha and psha in live_payloads:
            cert = "payload"
        elif (
            kind == "separated" and parent_c and parent_master.get(str(rid)) == parent_c
        ):
            cert = "derivation"
        if cert is None:
            continue
        out.append(
            {
                "content_sha256": csha,
                "payload_sha256": psha,
                "recording_id": rid,
                "track_audio_id": str(taid) if taid is not None else "",
                "stem": stem or "regular",
                "variant": variant or "regular",
                "kind": kind or "master",
                "id_source": "historical-content",
                "cert": cert,
            }
        )
    return out


def build_catalog(
    conn,
    set_id,
    *,
    file_sha256=_file_sha256,
    mdat_sha256=_mdat_sha256,
    flac_pcm_md5=_flac_pcm_md5,
):
    # Widened to the pull's own resolution (labeling/acquire/pull_set_for_alignment.py
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
        payload = _payload_for(path, mdat_sha256, flac_pcm_md5)
        entries.append(
            {
                "content_sha256": sha,
                "payload_sha256": payload,
                "recording_id": rid,
                "track_audio_id": str(taid),
                "stem": stem or "regular",
                "variant": variant or "regular",
                "kind": "master",
                "id_source": "content",
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
                "payload_sha256": _payload_for(spath, mdat_sha256, flac_pcm_md5),
                "recording_id": rid,
                "track_audio_id": str(taid),
                "stem": axis,
                # A separation preserves the parent master's length, so the
                # separated stem inherits the parent track_audio's variant
                # (mirrors the master loop's `variant or "regular"`).
                "variant": variant or "regular",
                "kind": "separated",
                "id_source": "content",
            }
        )

    entries.extend(_emit_historical(conn, recs, entries))
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
