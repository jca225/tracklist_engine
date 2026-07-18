"""SoundCloud api-v2 endpoint primitives — the reusable data-lake fetch surface.

Every function is read-only and does NO parsing/DB work; it yields raw dicts.
Collections paginate via linked_partitioning + next_href.
"""

from __future__ import annotations

from typing import Any, Iterator

import httpx

from soundcloud.client import SC_API, SKIP_STATUS_CODES, next_url, rl_get

_PAGE = 200


def _get_json(client: Any, rl: Any, url: str, params: dict | None = None) -> dict:
    return rl_get(client, rl, url, params=params).json()


def _paged(
    client: Any, rl: Any, cid: str, path: str, params: dict | None = None
) -> Iterator[dict]:
    base = {"client_id": cid, "limit": _PAGE, "linked_partitioning": 1}
    if params:
        base.update(params)
    url = f"{SC_API}{path}"
    first = True
    while url:
        try:
            data = _get_json(client, rl, url, params=base if first else None)
        except httpx.HTTPStatusError as e:
            if e.response is not None and e.response.status_code in SKIP_STATUS_CODES:
                return
            raise
        first = False
        for item in data.get("collection", []):
            yield item
        nxt = data.get("next_href")
        url = next_url(nxt, cid) if nxt else ""


def user(client: Any, rl: Any, cid: str, uid: int) -> dict:
    return _get_json(client, rl, f"{SC_API}/users/{uid}", params={"client_id": cid})


def track(client: Any, rl: Any, cid: str, tid: int) -> dict:
    return _get_json(client, rl, f"{SC_API}/tracks/{tid}", params={"client_id": cid})


def playlist(client: Any, rl: Any, cid: str, pid: int) -> dict:
    return _get_json(client, rl, f"{SC_API}/playlists/{pid}", params={"client_id": cid})


def user_likes(client: Any, rl: Any, cid: str, uid: int) -> Iterator[dict]:
    return _paged(client, rl, cid, f"/users/{uid}/likes")


def user_reposts(client: Any, rl: Any, cid: str, uid: int) -> Iterator[dict]:
    return _paged(client, rl, cid, f"/stream/users/{uid}/reposts")


def user_playlists(client: Any, rl: Any, cid: str, uid: int) -> Iterator[dict]:
    return _paged(client, rl, cid, f"/users/{uid}/playlists")


def user_tracks(client: Any, rl: Any, cid: str, uid: int) -> Iterator[dict]:
    return _paged(client, rl, cid, f"/users/{uid}/tracks")


def user_followings(client: Any, rl: Any, cid: str, uid: int) -> Iterator[dict]:
    return _paged(client, rl, cid, f"/users/{uid}/followings")


def user_followers(client: Any, rl: Any, cid: str, uid: int) -> Iterator[dict]:
    return _paged(client, rl, cid, f"/users/{uid}/followers")
