"""Back-compat shim — canonical client now lives in soundcloud.client."""

from __future__ import annotations

from typing import Any

from soundcloud.client import (  # noqa: F401
    CLIENT_ID_RE,
    SC_API,
    SC_HOME,
    SCRIPT_RE,
    SKIP_STATUS_CODES,
    USER_AGENT,
    RateLimiter,
    extract_client_id,
    next_url,
    rl_get,
    resolve,
    sc_client,
)


def resolve_track(
    client: Any, rl: RateLimiter, client_id: str, url: str
) -> dict[str, Any]:
    """Resolve a URL and assert it is a track (legacy personalization behavior)."""
    data = resolve(client, rl, client_id, url)
    if data.get("kind") != "track":
        raise ValueError(f"expected track, got kind={data.get('kind')}")
    return data
