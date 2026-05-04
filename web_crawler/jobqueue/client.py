"""HTTP client that mirrors the subset of MusicDatabase used by retry.py.

When retry.py is run with --rpc-url, it instantiates JobQueueClient instead
of MusicDatabase. The methods used by run_retries are duck-type compatible.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

# Allow `from web_crawler.jobqueue.client import JobQueueClient` to resolve
# the bare `data_models` import below — matches the sys.path convention
# already used in retry.py and jobqueue.server.
_WEB_CRAWLER_DIR = Path(__file__).resolve().parent.parent
if str(_WEB_CRAWLER_DIR) not in sys.path:
    sys.path.insert(0, str(_WEB_CRAWLER_DIR))

import httpx

from data_models import DJSetTrackMediaLink

log = logging.getLogger("jobqueue.client")


class JobQueueClient:
    """Drop-in replacement for the subset of MusicDatabase that retry.py uses.

    Construct with the URL of the jobqueue server (e.g. http://pi-storage:8765).
    Token is read from $JOBQUEUE_TOKEN. Methods raise httpx.HTTPStatusError on
    non-2xx — matching the existing retry.py error model where DB failures
    propagate as exceptions.
    """

    def __init__(self, base_url: str, token: str | None = None, timeout_s: float = 30.0) -> None:
        if token is None:
            token = os.environ.get("JOBQUEUE_TOKEN")
        if not token:
            raise RuntimeError(
                "JobQueueClient: token not provided and JOBQUEUE_TOKEN env var unset"
            )
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout_s,
        )

    def fetch_ajax_failures(
        self, max_retries: int, title_like: str | None
    ) -> list[dict]:
        r = self._client.post(
            "/ajax/scope",
            json={"max_retries": max_retries, "title_like": title_like},
        )
        r.raise_for_status()
        return r.json()

    def _bookkeep_with_retry(self, send: Callable[[], httpx.Response]) -> None:
        # Bookkeeping endpoints (mark-retry, delete) can hit a transient
        # SQLite lock on the server. Retry on 5xx with exponential backoff
        # so a momentary contention doesn't kill a multi-thousand-row drain.
        # Fetch and insert paths are intentionally not wrapped — a fetch
        # failure is fatal, and an insert must surface so we don't delete
        # the failure row believing the data landed.
        attempts = 4
        for attempt in range(attempts):
            r = send()
            if r.status_code < 500 or attempt == attempts - 1:
                r.raise_for_status()
                return
            log.warning(
                "jobqueue %s -> %d (attempt %d/%d), retrying",
                r.request.url, r.status_code, attempt + 1, attempts,
            )
            time.sleep(0.5 * (2 ** attempt))

    def increment_failure_retries(self, failure_id: int) -> None:
        self._bookkeep_with_retry(
            lambda: self._client.post(f"/ajax/failures/{failure_id}/retry")
        )

    def delete_failure(self, failure_id: int) -> None:
        self._bookkeep_with_retry(
            lambda: self._client.delete(f"/ajax/failures/{failure_id}")
        )

    def insert_track_media_links(self, links: Iterable[DJSetTrackMediaLink]) -> None:
        payload = [
            {
                "set_id": lk.set_id,
                "tlp_id": lk.tlp_id,
                "track_id": lk.track_id,
                "platform": lk.platform,
                "player_id": lk.player_id,
                "id_object": lk.id_object,
                "id_item": lk.id_item,
                "id_source": lk.id_source,
                "view_source": lk.view_source,
                "view_item": lk.view_item,
            }
            for lk in links
        ]
        if not payload:
            return
        r = self._client.post("/track-media-links", json=payload)
        r.raise_for_status()

    def close(self) -> None:
        self._client.close()
