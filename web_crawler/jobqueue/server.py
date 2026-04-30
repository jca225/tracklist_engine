"""FastAPI service that exposes the subset of MusicDatabase used by retry.py.

Runs on pi-storage. pi-worker calls it via JobQueueClient (jobqueue/client.py).
The DB itself is the queue — these endpoints just wrap MusicDatabase methods.

Bind:   $JOBQUEUE_BIND        (default 0.0.0.0:8765, reachable over Tailscale)
Auth:   $JOBQUEUE_TOKEN       Bearer token, required on every endpoint except /healthz
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Iterable

# Allow running this as a module (`python -m web_crawler.jobqueue.server`)
# while still importing the rest of web_crawler/ as siblings.
_WEB_CRAWLER_DIR = Path(__file__).resolve().parent.parent
if str(_WEB_CRAWLER_DIR) not in sys.path:
    sys.path.insert(0, str(_WEB_CRAWLER_DIR))

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel

from config import load_config
from data_models import DJSetTrackMediaLink
from database import MusicDatabase

log = logging.getLogger("jobqueue.server")


# --- auth ----------------------------------------------------------------

def _expected_token() -> str:
    tok = os.environ.get("JOBQUEUE_TOKEN")
    if not tok:
        raise RuntimeError("JOBQUEUE_TOKEN env var must be set on the server")
    return tok


def require_token(authorization: str | None = Header(default=None)) -> None:
    expected = _expected_token()
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    if authorization.removeprefix("Bearer ").strip() != expected:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "bad token")


# --- DB ------------------------------------------------------------------

_db: MusicDatabase | None = None


def get_db() -> MusicDatabase:
    global _db
    if _db is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        cfg_path = project_root / "config.yaml"
        cfg = load_config(cfg_path, project_root)
        _db = MusicDatabase(str(cfg.paths.db_path), str(cfg.paths.schema_path))
    return _db


# --- pydantic schemas ----------------------------------------------------

class FailureScopeQuery(BaseModel):
    max_retries: int
    title_like: str | None = None


class FailureRow(BaseModel):
    failure_id: int
    set_id: str
    set_url: str | None
    track_title: str | None
    track_id: str | None
    tlp_id: str | None
    params_json: str | None
    error: str | None
    retries: int


class TrackMediaLinkPayload(BaseModel):
    set_id: str
    tlp_id: str | None
    track_id: str | None
    platform: str
    player_id: str | None
    id_object: str | None = None
    id_item: str | None = None
    id_source: str | None = None
    view_source: str | None = None
    view_item: str | None = None


# --- app -----------------------------------------------------------------

app = FastAPI(title="tracklist_engine job queue", version="0.1.0")


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


def _fetch_ajax_failures(db: MusicDatabase, max_retries: int, title_like: str | None) -> list[dict]:
    """Same SQL as retry.fetch_ajax_failures, inlined to avoid pulling
    playwright/scraper imports into the server process."""
    if title_like is None:
        sql = """
        SELECT f.failure_id, f.set_id, s.set_url, f.track_title, f.track_id,
               f.tlp_id, f.params_json, f.error, f.retries
        FROM scrape_failures f
        JOIN dj_sets s USING(set_id)
        WHERE f.stage = 'ajax'
          AND f.retries < ?
        ORDER BY f.set_id, f.failure_id
        """
        params: tuple = (max_retries,)
    else:
        sql = """
        SELECT f.failure_id, f.set_id, s.set_url, f.track_title, f.track_id,
               f.tlp_id, f.params_json, f.error, f.retries
        FROM scrape_failures f
        JOIN dj_sets s USING(set_id)
        WHERE s.title LIKE ?
          AND f.stage = 'ajax'
          AND f.retries < ?
        ORDER BY f.set_id, f.failure_id
        """
        params = (title_like, max_retries)
    db.cursor.execute(sql, params)
    cols = [d[0] for d in db.cursor.description]
    return [dict(zip(cols, row)) for row in db.cursor.fetchall()]


@app.post("/ajax/scope", dependencies=[Depends(require_token)])
def ajax_scope(
    query: FailureScopeQuery,
    db: MusicDatabase = Depends(get_db),
) -> list[FailureRow]:
    rows = _fetch_ajax_failures(db, query.max_retries, query.title_like)
    return [FailureRow(**r) for r in rows]


@app.post("/ajax/failures/{failure_id}/retry", dependencies=[Depends(require_token)])
def ajax_increment_retry(
    failure_id: int,
    db: MusicDatabase = Depends(get_db),
) -> dict:
    db.increment_failure_retries(failure_id)
    return {"ok": True}


@app.delete("/ajax/failures/{failure_id}", dependencies=[Depends(require_token)])
def ajax_delete_failure(
    failure_id: int,
    db: MusicDatabase = Depends(get_db),
) -> dict:
    db.delete_failure(failure_id)
    return {"ok": True}


@app.post("/track-media-links", dependencies=[Depends(require_token)])
def insert_track_media_links(
    links: list[TrackMediaLinkPayload],
    db: MusicDatabase = Depends(get_db),
) -> dict:
    domain: Iterable[DJSetTrackMediaLink] = [
        DJSetTrackMediaLink(**lk.model_dump()) for lk in links
    ]
    db.insert_track_media_links(domain)
    return {"ok": True, "count": len(links)}


# --- entrypoint for `python -m web_crawler.jobqueue.server` --------------

def main() -> None:
    import uvicorn

    bind = os.environ.get("JOBQUEUE_BIND", "0.0.0.0:8765")
    host, port_s = bind.rsplit(":", 1)
    port = int(port_s)

    # Force-instantiate db at startup so we fail fast if config is bad
    get_db()
    # Also force-validate token presence at startup
    _expected_token()

    log.info("starting jobqueue server on %s:%d", host, port)
    uvicorn.run("web_crawler.jobqueue.server:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
