"""Import legacy dj-listener-pipeline JSONL into taste warehouse."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from personalization.config import MixTarget, TasteSettings
from personalization.enrich import make_user_id
from personalization.persistence import (
    connect,
    insert_comments,
    insert_likes,
    insert_playlists,
    upsert_listener,
)
from personalization.records import ListenerRow, ScLikeRow, ScMixCommentRow, ScPlaylistRow

logger = logging.getLogger(__name__)


def _listener_from_rec(rec: dict, mix: MixTarget) -> ListenerRow | None:
    uid = rec.get("sc_user_id") or rec.get("user_id")
    if uid is None:
        return None
    handle = rec.get("permalink") or str(uid)
    evidence = {"imported": True}
    if rec.get("relation"):
        evidence["relation"] = rec["relation"]
    return ListenerRow(
        user_id=make_user_id("soundcloud", handle),
        platform="soundcloud",
        handle=handle,
        mix_id=mix.mix_id,
        sc_user_id=int(uid),
        first_seen_at=datetime.now(timezone.utc).isoformat(),
        source_evidence_json=json.dumps(evidence),
        username=rec.get("username"),
        followers_count=rec.get("followers_count"),
        followings_count=rec.get("followings_count"),
        verified=bool(rec.get("verified")),
        city=rec.get("city"),
        country_code=rec.get("country_code"),
    )


def import_archive_dir(settings: TasteSettings, mix: MixTarget, archive_dir: Path) -> dict[str, int]:
    """Import listeners, likes, comments, playlists from Archive `data/raw/bb11/` layout."""
    conn = connect(settings.db_path)
    stats = {"listeners": 0, "likes": 0, "comments": 0, "playlists": 0}

    sc_dir = archive_dir / "soundcloud"
    for p in sorted(sc_dir.glob("*.jsonl")):
        if p.name.startswith("_"):
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            row = _listener_from_rec(json.loads(line), mix)
            if row is None:
                continue
            upsert_listener(conn, row)
            stats["listeners"] += 1

    conn.commit()
    handle_by_sc: dict[int, str] = {}
    for row in conn.execute(
        "SELECT sc_user_id, handle FROM listeners WHERE mix_id = ? AND sc_user_id IS NOT NULL",
        (mix.mix_id,),
    ):
        handle_by_sc[int(row[0])] = str(row[1])

    likes_dir = archive_dir / "soundcloud_likes"
    batch: list[ScLikeRow] = []
    for p in sorted(likes_dir.glob("*.jsonl")):
        if p.name.startswith("_"):
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            sc_uid = int(rec["sc_user_id"])
            handle = handle_by_sc.get(sc_uid, str(sc_uid))
            batch.append(
                ScLikeRow(
                    user_id=make_user_id("soundcloud", handle),
                    mix_id=mix.mix_id,
                    sc_user_id=sc_uid,
                    liked_at=str(rec["liked_at"]),
                    track_id=int(rec["track_id"]),
                    track_title=str(rec.get("track_title") or ""),
                    track_permalink=rec.get("track_permalink"),
                    track_artist_username=rec.get("track_artist_username"),
                    track_genre=rec.get("track_genre"),
                    raw_json=json.dumps(rec, default=str),
                )
            )
            if len(batch) >= 5000:
                stats["likes"] += insert_likes(conn, tuple(batch))
                batch = []
    if batch:
        stats["likes"] += insert_likes(conn, tuple(batch))

    comments_dir = archive_dir / "soundcloud_comments"
    c_batch: list[ScMixCommentRow] = []
    for p in sorted(comments_dir.glob("*.jsonl")):
        if p.name.startswith("_"):
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            sc_uid = int(rec.get("user_id") or rec.get("sc_user_id"))
            handle = rec.get("permalink") or handle_by_sc.get(sc_uid, str(sc_uid))
            cid = rec.get("comment_id")
            if cid is None:
                continue
            pos = rec.get("track_position_ms")
            c_batch.append(
                ScMixCommentRow(
                    user_id=make_user_id("soundcloud", handle),
                    mix_id=mix.mix_id,
                    sc_user_id=sc_uid,
                    sc_track_id=int(rec["track_id"]),
                    comment_id=int(cid),
                    commented_at=str(rec.get("created_at") or rec.get("commented_at")),
                    mix_position_ms=int(pos) if pos is not None else None,
                    body=str(rec.get("body") or ""),
                    raw_json=json.dumps(rec, default=str),
                )
            )
            if len(c_batch) >= 5000:
                stats["comments"] += insert_comments(conn, tuple(c_batch))
                c_batch = []
    if c_batch:
        stats["comments"] += insert_comments(conn, tuple(c_batch))

    playlists_dir = archive_dir / "soundcloud_playlists"
    pl_batch: list[ScPlaylistRow] = []
    for p in sorted(playlists_dir.glob("*.jsonl")):
        if p.name.startswith("_"):
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            sc_uid = int(rec["sc_user_id"])
            pid = rec.get("playlist_id")
            if pid is None:
                continue
            handle = handle_by_sc.get(sc_uid, str(sc_uid))
            track_ids = rec.get("track_ids") or []
            pl_batch.append(
                ScPlaylistRow(
                    user_id=make_user_id("soundcloud", handle),
                    mix_id=mix.mix_id,
                    sc_user_id=sc_uid,
                    playlist_id=int(pid),
                    title=rec.get("title"),
                    track_count=rec.get("track_count"),
                    track_ids_json=json.dumps(track_ids),
                    created_at=rec.get("created_at"),
                    last_modified=rec.get("last_modified"),
                    raw_json=json.dumps(rec, default=str),
                )
            )
            if len(pl_batch) >= 2000:
                stats["playlists"] += insert_playlists(conn, tuple(pl_batch))
                pl_batch = []
    if pl_batch:
        stats["playlists"] += insert_playlists(conn, tuple(pl_batch))

    conn.close()
    logger.info("imported mix=%s stats=%s", mix.mix_id, stats)
    return stats
