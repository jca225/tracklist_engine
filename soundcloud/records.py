"""Frozen records for the SoundCloud data lake (nodes, edges, crawl policy)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ScUser:
    sc_user_id: int
    permalink: str
    username: str
    followers_count: int | None
    followings_count: int | None
    verified: bool
    city: str | None
    country: str | None
    description: str | None
    fetched_at: str
    raw_ref: str


@dataclass(frozen=True)
class ScTrack:
    sc_track_id: int
    title: str
    owner_sc_user_id: int
    genre: str | None
    tag_list: str | None
    duration_ms: int | None
    playback_count: int | None
    likes_count: int | None
    created_at: str | None
    permalink: str | None
    fetched_at: str
    raw_ref: str


@dataclass(frozen=True)
class ScPlaylist:
    sc_playlist_id: int
    title: str
    owner_sc_user_id: int
    track_count: int | None
    is_album: bool
    fetched_at: str
    raw_ref: str


DEFAULT_ENTITIES = frozenset(
    {"likes", "reposts", "playlists", "tracks", "followings", "followers"}
)


@dataclass(frozen=True)
class CrawlPolicy:
    seed_user_ids: tuple[int, ...]
    depth: int = 1
    entity_types: frozenset[str] = DEFAULT_ENTITIES
    rpm: int = 60
