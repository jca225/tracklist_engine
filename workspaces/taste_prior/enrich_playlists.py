"""Enrich cohort users with SoundCloud playlists (live scrape)."""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from workspaces.taste_prior.config import MixTarget, TasteSettings
from workspaces.taste_prior.enrich import make_user_id
from workspaces.taste_prior.persistence import (
    connect,
    insert_playlists,
    listener_sc_ids,
    load_checkpoint,
    log_run,
    save_checkpoint,
)
from workspaces.taste_prior.records import ScPlaylistRow
from workspaces.taste_prior.soundcloud_client import (
    SC_API,
    USER_AGENT,
    RateLimiter,
    extract_client_id,
    next_url,
    rl_get,
)

logger = logging.getLogger(__name__)

PLAYLISTS_PAGE_LIMIT = 20
DEFAULT_MAX_PLAYLISTS_PER_USER = 50


def _playlists_jsonl(settings: TasteSettings, mix_id: str) -> Path:
    d = settings.data_dir / "raw" / mix_id / "soundcloud_playlists"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{datetime.now(timezone.utc):%Y%m%d}.jsonl"


def enrich_playlists_batch(
    settings: TasteSettings, mix: MixTarget, *, batch_size: int = 20
) -> int:
    started = datetime.now(timezone.utc).isoformat()
    conn = connect(settings.db_path)
    ck = load_checkpoint(conn, mix.mix_id, "enrich_playlists")
    completed: set[int] = set(ck.get("completed_sc_user_ids") or [])

    pending = [uid for uid in listener_sc_ids(conn, mix.mix_id) if uid not in completed][:batch_size]
    if not pending:
        conn.close()
        return 0

    rl = RateLimiter(settings.soundcloud_rpm)
    inserted = 0
    jsonl_path = _playlists_jsonl(settings, mix.mix_id)

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30, follow_redirects=True) as client:
        client_id = ck.get("client_id") or extract_client_id(client, rl)
        ck["client_id"] = client_id

        for sc_uid in pending:
            handle_row = conn.execute(
                "SELECT handle FROM listeners WHERE sc_user_id = ? AND mix_id = ?",
                (sc_uid, mix.mix_id),
            ).fetchone()
            handle = str(handle_row["handle"]) if handle_row else str(sc_uid)
            uid = make_user_id("soundcloud", handle)
            url: str | None = (
                f"{SC_API}/users/{sc_uid}/playlists?client_id={client_id}&limit={PLAYLISTS_PAGE_LIMIT}"
            )
            rows: list[ScPlaylistRow] = []
            jsonl_batch: list[dict[str, Any]] = []
            pages = 0
            while url and len(rows) < DEFAULT_MAX_PLAYLISTS_PER_USER and pages < 10:
                try:
                    resp = rl_get(client, rl, url)
                except httpx.HTTPStatusError as e:
                    if e.response.status_code in (401, 403, 404, 429, 500, 502, 503):
                        logger.warning(
                            "enrich-playlists skip user sc_uid=%s status=%s",
                            sc_uid,
                            e.response.status_code,
                        )
                        break
                    raise
                data = resp.json()
                for pl in data.get("collection") or []:
                    pid = pl.get("id")
                    if pid is None:
                        continue
                    track_ids = [t.get("id") for t in (pl.get("tracks") or []) if t.get("id")]
                    raw = {
                        "sc_user_id": sc_uid,
                        "playlist_id": pid,
                        "title": pl.get("title"),
                        "track_count": pl.get("track_count") or len(track_ids),
                        "track_ids": track_ids,
                        "mix_id": mix.mix_id,
                    }
                    jsonl_batch.append(raw)
                    rows.append(
                        ScPlaylistRow(
                            user_id=uid,
                            mix_id=mix.mix_id,
                            sc_user_id=sc_uid,
                            playlist_id=int(pid),
                            title=pl.get("title"),
                            track_count=pl.get("track_count"),
                            track_ids_json=json.dumps(track_ids),
                            created_at=pl.get("created_at"),
                            last_modified=pl.get("last_modified"),
                            raw_json=json.dumps(pl, default=str),
                        )
                    )
                nxt = data.get("next_href")
                pages += 1
                url = next_url(nxt, client_id) if nxt else None

            if jsonl_batch:
                with jsonl_path.open("a") as f:
                    for r in jsonl_batch:
                        f.write(json.dumps(r, default=str) + "\n")
            if rows:
                inserted += insert_playlists(conn, tuple(rows))
            completed.add(sc_uid)

    ck["completed_sc_user_ids"] = sorted(completed)
    save_checkpoint(conn, mix.mix_id, "enrich_playlists", ck)
    log_run(
        conn,
        phase="enrich_playlists",
        mix_id=mix.mix_id,
        started_at=started,
        output_rows=inserted,
        params={"batch": batch_size, "completed": len(completed)},
    )
    conn.close()
    return inserted
