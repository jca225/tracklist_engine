"""Enrich cohort users with SoundCloud track_likes."""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from workspaces.taste_prior.config import MixTarget, TasteSettings
from workspaces.taste_prior.persistence import (
    connect,
    insert_likes,
    listener_sc_ids,
    load_checkpoint,
    log_run,
    save_checkpoint,
)
from workspaces.taste_prior.records import ScLikeRow
from workspaces.taste_prior.soundcloud_client import (
    SC_API,
    SKIP_STATUS_CODES,
    RateLimiter,
    extract_client_id,
    next_url,
    rl_get,
    sc_client,
)

logger = logging.getLogger(__name__)

LIKES_PAGE_LIMIT = 200
DEFAULT_MAX_LIKES_PER_USER = 500


def make_user_id(platform: str, handle: str) -> str:
    return hashlib.sha256(f"{platform}:{handle.lower()}".encode()).hexdigest()[:16]


def _likes_jsonl(settings: TasteSettings, mix_id: str) -> Path:
    d = settings.data_dir / "raw" / mix_id / "soundcloud_likes"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{datetime.now(timezone.utc):%Y%m%d}.jsonl"


def enrich_batch(settings: TasteSettings, mix: MixTarget, *, batch_size: int = 20) -> int:
    """Enrich up to batch_size users; returns likes inserted."""
    started = datetime.now(timezone.utc).isoformat()
    conn = connect(settings.db_path)
    ck = load_checkpoint(conn, mix.mix_id, "enrich_likes")
    completed: set[int] = set(ck.get("completed_sc_user_ids") or [])
    in_progress: dict[str, Any] = ck.get("in_progress") or {}

    all_ids = listener_sc_ids(conn, mix.mix_id)
    pending = [uid for uid in all_ids if uid not in completed and str(uid) not in in_progress]
    if not pending and in_progress:
        pending = [int(k) for k in in_progress.keys()]

    targets = pending[:batch_size]
    if not targets and not in_progress:
        logger.info("enrich complete for mix=%s", mix.mix_id)
        conn.close()
        return 0

    rl = RateLimiter(settings.soundcloud_rpm)
    inserted = 0
    jsonl_path = _likes_jsonl(settings, mix.mix_id)

    with sc_client() as client:
        client_id = ck.get("client_id") or extract_client_id(client, rl)
        ck["client_id"] = client_id

        work = targets or [int(k) for k in list(in_progress.keys())[:batch_size]]
        for sc_uid in work:
            key = str(sc_uid)
            handle_row = conn.execute(
                "SELECT handle FROM listeners WHERE sc_user_id = ? AND mix_id = ?",
                (sc_uid, mix.mix_id),
            ).fetchone()
            handle = str(handle_row["handle"]) if handle_row else str(sc_uid)
            cursor = in_progress.get(key)
            url: str | None = cursor or (
                f"{SC_API}/users/{sc_uid}/track_likes?client_id={client_id}&limit={LIKES_PAGE_LIMIT}"
            )
            rows: list[ScLikeRow] = []
            jsonl_batch: list[dict[str, Any]] = []
            user_done = False
            skip_user = False
            pages = 0

            while url and len(rows) < DEFAULT_MAX_LIKES_PER_USER and pages < 10:
                try:
                    resp = rl_get(client, rl, url)
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in SKIP_STATUS_CODES:
                        logger.warning(
                            "enrich skip user sc_uid=%s status=%s",
                            sc_uid,
                            e.response.status_code,
                        )
                        skip_user = True
                        break
                    raise
                except httpx.TransportError as e:
                    # Retries exhausted — resume from in_progress cursor on next tick.
                    logger.warning("enrich defer user sc_uid=%s transport=%s pages=%d", sc_uid, e, pages)
                    break
                data = resp.json()
                for item in data.get("collection") or []:
                    track = item.get("track") or item
                    tid = track.get("id")
                    if tid is None:
                        continue
                    liked_at = item.get("created_at") or datetime.now(timezone.utc).isoformat()
                    raw = {
                        "sc_user_id": sc_uid,
                        "liked_at": liked_at,
                        "track_id": tid,
                        "track_title": track.get("title"),
                        "track_permalink": track.get("permalink"),
                        "track_artist_username": (track.get("user") or {}).get("username"),
                        "track_genre": track.get("genre"),
                        "mix_id": mix.mix_id,
                    }
                    jsonl_batch.append(raw)
                    rows.append(
                        ScLikeRow(
                            user_id=make_user_id("soundcloud", handle),
                            mix_id=mix.mix_id,
                            sc_user_id=sc_uid,
                            liked_at=str(liked_at),
                            track_id=int(tid),
                            track_title=str(track.get("title") or ""),
                            track_permalink=track.get("permalink"),
                            track_artist_username=(track.get("user") or {}).get("username"),
                            track_genre=track.get("genre"),
                            raw_json=json.dumps(raw, default=str),
                        )
                    )
                nxt = data.get("next_href")
                pages += 1
                if not nxt or len(rows) >= DEFAULT_MAX_LIKES_PER_USER:
                    user_done = True
                    url = None
                else:
                    url = next_url(nxt, client_id)
                    in_progress[key] = url

            if jsonl_batch:
                with jsonl_path.open("a") as f:
                    for r in jsonl_batch:
                        f.write(json.dumps(r, default=str) + "\n")

            if rows:
                inserted += insert_likes(conn, tuple(rows))

            if user_done or skip_user:
                completed.add(sc_uid)
                in_progress.pop(key, None)
            elif pages == 0 and not rows:
                in_progress.pop(key, None)

            ck["completed_sc_user_ids"] = sorted(completed)
            ck["in_progress"] = in_progress
            save_checkpoint(conn, mix.mix_id, "enrich_likes", ck)
    log_run(
        conn,
        phase="enrich_likes",
        mix_id=mix.mix_id,
        started_at=started,
        output_rows=inserted,
        params={"batch": batch_size, "completed": len(completed)},
    )
    conn.close()
    logger.info("enrich tick mix=%s inserted=%d completed=%d", mix.mix_id, inserted, len(completed))
    return inserted
