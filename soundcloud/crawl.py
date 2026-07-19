"""Depth-1 (parameterizable) frontier crawler over the fetch primitives."""

from __future__ import annotations

from collections import deque
from typing import Any

from soundcloud import fetch as _fetch
from soundcloud import rawlake, store
from soundcloud.config import SoundCloudSettings
from soundcloud.records import CrawlPolicy


def _repost_item(raw: dict) -> tuple[str, dict] | None:
    if "track" in raw and raw["track"]:
        return "track", raw["track"]
    if "playlist" in raw and raw["playlist"]:
        return "playlist", raw["playlist"]
    return None


def _crawl_user(
    conn,
    settings: SoundCloudSettings,
    uid: int,
    entities: frozenset[str],
    client: Any,
    rl: Any,
    cid: str,
    now: str,
    fx: Any,
    counts: dict,
) -> list[int]:
    """Process one user's own collections. Returns neighbor user ids discovered."""
    raw_root = settings.raw_root
    u_raw = fx.user(client, rl, cid, uid)
    ref = rawlake.write_snapshot(raw_root, "user", uid, now, [u_raw])
    store.upsert_user(conn, store.parse_user(u_raw, fetched_at=now, raw_ref=ref))
    counts["users"] += 1
    neighbors: list[int] = []

    if "likes" in entities and not store.is_done(conn, uid, "likes"):
        items = list(fx.user_likes(client, rl, cid, uid))
        ref = rawlake.write_snapshot(raw_root, "likes", uid, now, items)
        for it in items:
            tr = it.get("track", it)
            if not tr or "id" not in tr:
                continue
            store.upsert_track(conn, store.parse_track(tr, fetched_at=now, raw_ref=ref))
            store.add_like(conn, uid, int(tr["id"]), it.get("created_at"))
            counts["tracks"] += 1
            counts["likes"] += 1
        store.mark_done(conn, uid, "likes", now)

    if "reposts" in entities and not store.is_done(conn, uid, "reposts"):
        items = list(fx.user_reposts(client, rl, cid, uid))
        ref = rawlake.write_snapshot(raw_root, "reposts", uid, now, items)
        for it in items:
            parsed = _repost_item(it)
            if not parsed:
                continue
            kind, obj = parsed
            if kind == "track" and "id" in obj:
                store.upsert_track(
                    conn, store.parse_track(obj, fetched_at=now, raw_ref=ref)
                )
            store.add_repost(conn, uid, kind, int(obj["id"]), it.get("created_at"))
            counts["reposts"] += 1
        store.mark_done(conn, uid, "reposts", now)

    if "playlists" in entities and not store.is_done(conn, uid, "playlists"):
        items = list(fx.user_playlists(client, rl, cid, uid))
        ref = rawlake.write_snapshot(raw_root, "playlists", uid, now, items)
        for pl in items:
            store.upsert_playlist(
                conn, store.parse_playlist(pl, fetched_at=now, raw_ref=ref)
            )
            counts["playlists"] += 1
            for pos, tr in enumerate(pl.get("tracks", []) or []):
                if "id" not in tr:
                    continue
                store.upsert_track(
                    conn, store.parse_track(tr, fetched_at=now, raw_ref=ref)
                )
                store.add_playlist_track(conn, int(pl["id"]), int(tr["id"]), pos)
        store.mark_done(conn, uid, "playlists", now)

    if "tracks" in entities and not store.is_done(conn, uid, "tracks"):
        items = list(fx.user_tracks(client, rl, cid, uid))
        ref = rawlake.write_snapshot(raw_root, "tracks", uid, now, items)
        for tr in items:
            if "id" not in tr:
                continue
            store.upsert_track(conn, store.parse_track(tr, fetched_at=now, raw_ref=ref))
            counts["tracks"] += 1
        store.mark_done(conn, uid, "tracks", now)

    for phase, edge_from_seed in (("followings", True), ("followers", False)):
        if phase in entities and not store.is_done(conn, uid, phase):
            items = list(getattr(fx, f"user_{phase}")(client, rl, cid, uid))
            ref = rawlake.write_snapshot(raw_root, phase, uid, now, items)
            for nu in items:
                if "id" not in nu:
                    continue
                nid = int(nu["id"])
                store.upsert_user(
                    conn, store.parse_user(nu, fetched_at=now, raw_ref=ref)
                )
                counts["users"] += 1
                if edge_from_seed:
                    store.add_follow(conn, uid, nid)
                else:
                    store.add_follow(conn, nid, uid)
                counts["follows"] += 1
                neighbors.append(nid)
            store.mark_done(conn, uid, phase, now)

    return neighbors


def crawl(
    conn,
    settings: SoundCloudSettings,
    policy: CrawlPolicy,
    client: Any,
    rl: Any,
    cid: str,
    now: str,
    fetch_mod: Any = _fetch,
) -> dict:
    counts = {
        k: 0 for k in ("users", "tracks", "likes", "reposts", "playlists", "follows")
    }
    visited: set[int] = set()
    queue: deque[tuple[int, int]] = deque((uid, 0) for uid in policy.seed_user_ids)
    while queue:
        uid, depth = queue.popleft()
        if uid in visited:
            continue
        visited.add(uid)
        neighbors = _crawl_user(
            conn,
            settings,
            uid,
            policy.entity_types,
            client,
            rl,
            cid,
            now,
            fetch_mod,
            counts,
        )
        if depth + 1 < policy.depth:
            for nid in neighbors:
                if nid not in visited:
                    queue.append((nid, depth + 1))
    return counts
