"""SoundCloud api-v2 client helpers (from dj-listener-pipeline)."""
from __future__ import annotations

import re
import time
from typing import Any

import httpx

SC_HOME = "https://soundcloud.com/"
SC_API = "https://api-v2.soundcloud.com"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
SCRIPT_RE = re.compile(r'https://[a-z0-9\-\.]+\.sndcdn\.com/assets/[^"\']+\.js')
CLIENT_ID_RE = re.compile(r'client_id\s*[:=]\s*"([A-Za-z0-9]{20,40})"')


class RateLimiter:
    def __init__(self, rpm: int) -> None:
        self._interval = 60.0 / max(rpm, 1)
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last = time.monotonic()


def rl_get(client: httpx.Client, rl: RateLimiter, url: str, **kwargs: Any) -> httpx.Response:
    rl.wait()
    resp = client.get(url, **kwargs)
    resp.raise_for_status()
    return resp


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
