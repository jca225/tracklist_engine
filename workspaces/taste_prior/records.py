"""Frozen records for taste warehouse."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ListenerRow:
    user_id: str
    platform: str
    handle: str
    mix_id: str
    sc_user_id: int | None
    first_seen_at: str
    source_evidence_json: str = "{}"
    username: str | None = None
    followers_count: int | None = None
    followings_count: int | None = None
    verified: bool = False
    city: str | None = None
    country_code: str | None = None


@dataclass(frozen=True)
class ScLikeRow:
    user_id: str
    mix_id: str
    sc_user_id: int
    liked_at: str
    track_id: int
    track_title: str
    track_permalink: str | None
    track_artist_username: str | None
    track_genre: str | None
    raw_json: str


@dataclass(frozen=True)
class ScMixCommentRow:
    """Comment on a mix upload — mix_position_ms is playhead in the mix."""

    user_id: str
    mix_id: str
    sc_user_id: int
    sc_track_id: int
    comment_id: int
    commented_at: str
    mix_position_ms: int | None
    body: str
    raw_json: str


@dataclass(frozen=True)
class ScPlaylistRow:
    user_id: str
    mix_id: str
    sc_user_id: int
    playlist_id: int
    title: str | None
    track_count: int | None
    track_ids_json: str
    created_at: str | None
    last_modified: str | None
    raw_json: str


@dataclass(frozen=True)
class ListenerBotScoreRow:
    user_id: str
    mix_id: str
    sc_user_id: int | None
    bot_score: float
    is_bot: bool
    reasons_json: str
    computed_at: str
