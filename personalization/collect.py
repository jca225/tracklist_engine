"""Collect SoundCloud listeners from a mix upload (one resumable pass)."""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from personalization.config import MixTarget, TasteSettings
from personalization.persistence import (
    connect,
    insert_comments,
    load_checkpoint,
    log_run,
    save_checkpoint,
    upsert_listener,
)
from personalization.records import ListenerRow, ScMixCommentRow
from personalization.soundcloud_client import (
    SC_API,
    RateLimiter,
    extract_client_id,
    next_url,
    resolve_track,
    rl_get,
    sc_client,
)

logger = logging.getLogger(__name__)

PAGE_LIMIT = 200
MAX_PAGES_PER_TICK = 5  # keep systemd ticks short


def make_user_id(platform: str, handle: str) -> str:
    return hashlib.sha256(f"{platform}:{handle.lower()}".encode()).hexdigest()[:16]


def _raw_dir(settings: TasteSettings, mix_id: str) -> Path:
    d = settings.data_dir / "raw" / mix_id / "soundcloud"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def collect_tick(settings: TasteSettings, mix: MixTarget) -> int:
    """One resumable collect tick. Returns new listener rows upserted."""
    if not mix.soundcloud_url:
        logger.warning("mix %s has no soundcloud_url", mix.mix_id)
        return 0

    started = datetime.now(timezone.utc).isoformat()
    conn = connect(settings.db_path)
    ck = load_checkpoint(conn, mix.mix_id, "collect")
    out_path = _raw_dir(settings, mix.mix_id) / f"{datetime.now(timezone.utc):%Y%m%d}.jsonl"
    rl = RateLimiter(settings.soundcloud_rpm)
    n_upserted = 0

    with sc_client() as client:
        client_id = ck.get("client_id") or extract_client_id(client, rl)
        track_id = ck.get("track_id")
        if track_id is None:
            track = resolve_track(client, rl, client_id, mix.soundcloud_url)
            track_id = track["id"]
            ck["track_id"] = track_id
            ck["client_id"] = client_id

        for stream in ("likers", "reposters"):
            done_key = f"{stream}_done"
            cursor_key = f"{stream}_cursor"
            if ck.get(done_key):
                continue
            url: str | None = ck.get(cursor_key) or (
                f"{SC_API}/tracks/{track_id}/{stream}?client_id={client_id}&limit={PAGE_LIMIT}"
            )
            pages = 0
            batch: list[dict[str, Any]] = []
            while url and pages < MAX_PAGES_PER_TICK:
                resp = rl_get(client, rl, url)
                data = resp.json()
                for u in data.get("collection") or []:
                    uid = u.get("id")
                    if uid is None:
                        continue
                    handle = u.get("permalink") or str(uid)
                    rec = {
                        "sc_user_id": uid,
                        "permalink": u.get("permalink"),
                        "username": u.get("username"),
                        "relation": stream.rstrip("s"),
                        "mix_id": mix.mix_id,
                        "track_id": track_id,
                    }
                    batch.append(rec)
                    user_id = make_user_id("soundcloud", handle)
                    upsert_listener(
                        conn,
                        ListenerRow(
                            user_id=user_id,
                            platform="soundcloud",
                            handle=handle,
                            mix_id=mix.mix_id,
                            sc_user_id=int(uid),
                            first_seen_at=datetime.now(timezone.utc).isoformat(),
                            source_evidence_json=json.dumps({"relations": [stream.rstrip("s")]}),
                        ),
                    )
                    n_upserted += 1
                nxt = data.get("next_href")
                pages += 1
                if not nxt:
                    ck[done_key] = True
                    ck[cursor_key] = None
                    url = None
                else:
                    url = next_url(nxt, client_id)
                    ck[cursor_key] = url
            if batch:
                _append_jsonl(out_path, batch)
            conn.commit()

        # comments — body + created_at + playhead position in mix (track_position_ms)
        if not ck.get("comments_done"):
            url = ck.get("comments_cursor") or (
                f"{SC_API}/tracks/{track_id}/comments?client_id={client_id}&threaded=0&limit={PAGE_LIMIT}"
            )
            pages = 0
            jsonl_batch: list[dict[str, Any]] = []
            comment_rows: list[ScMixCommentRow] = []
            while url and pages < MAX_PAGES_PER_TICK:
                resp = rl_get(client, rl, url)
                data = resp.json()
                for c in data.get("collection") or []:
                    user = c.get("user") or {}
                    uid = user.get("id")
                    cid = c.get("id")
                    if uid is None or cid is None:
                        continue
                    handle = user.get("permalink") or str(uid)
                    user_id = make_user_id("soundcloud", handle)
                    created_at = c.get("created_at") or datetime.now(timezone.utc).isoformat()
                    pos_ms = c.get("timestamp")
                    body = (c.get("body") or "").strip()
                    raw = {
                        "track_id": track_id,
                        "comment_id": cid,
                        "sc_user_id": uid,
                        "permalink": user.get("permalink"),
                        "username": user.get("username"),
                        "body": body,
                        "created_at": created_at,
                        "track_position_ms": pos_ms,
                        "mix_id": mix.mix_id,
                    }
                    jsonl_batch.append(raw)
                    comment_rows.append(
                        ScMixCommentRow(
                            user_id=user_id,
                            mix_id=mix.mix_id,
                            sc_user_id=int(uid),
                            sc_track_id=int(track_id),
                            comment_id=int(cid),
                            commented_at=str(created_at),
                            mix_position_ms=int(pos_ms) if pos_ms is not None else None,
                            body=body,
                            raw_json=json.dumps(raw, default=str),
                        )
                    )
                    upsert_listener(
                        conn,
                        ListenerRow(
                            user_id=user_id,
                            platform="soundcloud",
                            handle=handle,
                            mix_id=mix.mix_id,
                            sc_user_id=int(uid),
                            first_seen_at=datetime.now(timezone.utc).isoformat(),
                            source_evidence_json=json.dumps({"relations": ["commenter"]}),
                        ),
                    )
                    n_upserted += 1
                nxt = data.get("next_href")
                pages += 1
                if not nxt:
                    ck["comments_done"] = True
                    ck["comments_cursor"] = None
                    url = None
                else:
                    url = next_url(nxt, client_id)
                    ck["comments_cursor"] = url
            if jsonl_batch:
                _append_jsonl(_raw_dir(settings, mix.mix_id) / "soundcloud_comments.jsonl", jsonl_batch)
            if comment_rows:
                insert_comments(conn, tuple(comment_rows))
            conn.commit()

    save_checkpoint(conn, mix.mix_id, "collect", ck)
    log_run(conn, phase="collect", mix_id=mix.mix_id, started_at=started, output_rows=n_upserted, params=ck)
    conn.close()
    logger.info("collect tick mix=%s upserted=%d", mix.mix_id, n_upserted)
    return n_upserted
