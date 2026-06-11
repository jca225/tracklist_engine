"""SoundCloud api-v2 client helpers (from dj-listener-pipeline)."""
from __future__ import annotations

import logging
import re
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SC_HOME = "https://soundcloud.com/"
SC_API = "https://api-v2.soundcloud.com"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
SCRIPT_RE = re.compile(r'https://[a-z0-9\-\.]+\.sndcdn\.com/assets/[^"\']+\.js')
CLIENT_ID_RE = re.compile(r'client_id\s*[:=]\s*"([A-Za-z0-9]{20,40})"')

# SC's edge resets idle keep-alive sockets; one-shot connections are more reliable.
SC_CLIENT_LIMITS = httpx.Limits(max_keepalive_connections=0, max_connections=1)

# HTTP statuses that mean "skip this resource", not "abort the tick".
SKIP_STATUS_CODES = frozenset({401, 403, 404, 429, 500, 502, 503})


class RateLimiter:
    def __init__(self, rpm: int) -> None:
        self._interval = 60.0 / max(rpm, 1)
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last = time.monotonic()


def sc_client(**kwargs: Any) -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=30,
        follow_redirects=True,
        limits=SC_CLIENT_LIMITS,
        **kwargs,
    )


def rl_get(
    client: httpx.Client,
    rl: RateLimiter,
    url: str,
    *,
    max_retries: int = 3,
    **kwargs: Any,
) -> httpx.Response:
    last_err: httpx.TransportError | None = None
    for attempt in range(max_retries):
        try:
            rl.wait()
            resp = client.get(url, **kwargs)
            resp.raise_for_status()
            return resp
        except httpx.TransportError as e:
            last_err = e
            if attempt + 1 >= max_retries:
                raise
            delay = 0.5 * (2**attempt)
            logger.debug("rl_get transport retry %d/%d delay=%.1fs url=%s", attempt + 1, max_retries, delay, url)
            time.sleep(delay)
    assert last_err is not None
    raise last_err


def extract_client_id(client: httpx.Client, rl: RateLimiter) -> str:
    html = rl_get(client, rl, SC_HOME).text
    for script_url in reversed(SCRIPT_RE.findall(html)):
        try:
            js = rl_get(client, rl, script_url).text
        except httpx.HTTPError:
            continue
        m = CLIENT_ID_RE.search(js)
        if m:
            return m.group(1)
    raise RuntimeError("SoundCloud client_id not found in homepage scripts")


def resolve_track(client: httpx.Client, rl: RateLimiter, client_id: str, url: str) -> dict[str, Any]:
    resp = rl_get(
        client,
        rl,
        f"{SC_API}/resolve",
        params={"url": url, "client_id": client_id},
    )
    data = resp.json()
    if data.get("kind") != "track":
        raise ValueError(f"expected track, got kind={data.get('kind')}")
    return data


def next_url(nxt: str, client_id: str) -> str:
    sep = "&" if "?" in nxt else "?"
    return f"{nxt}{sep}client_id={client_id}"
