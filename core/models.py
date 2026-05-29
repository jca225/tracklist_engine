from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class MediaSource:
    """A resolved URL on one platform for a canonical 1001tracklists track."""
    platform: str            # 'youtube' | 'soundcloud' | 'spotify' | 'apple'
    player_id: str           # YT video id | SC track id | Spotify track id
    url: str


@dataclass(frozen=True)
class Track:
    """A canonical track as seen in the crawler DB."""
    track_id: str            # 1001tracklists data-trackid
    tlp_ids: tuple[str, ...] # per-set-row ids this track appears under
    sources: tuple[MediaSource, ...]

    def source_for(self, platform: str) -> MediaSource | None:
        for s in self.sources:
            if s.platform == platform:
                return s
        return None


@dataclass(frozen=True)
class AudioAsset:
    """A downloaded audio file on disk, linked to a canonical track."""
    track_audio_id: int | None  # None pre-insert
    track_id: str
    platform: str
    source_url: str
    player_id: str
    path: str
    sha256: str | None
    duration_s: float | None
    sample_rate: int | None
    codec: str | None
    bitrate_kbps: int | None
    # Which version this audio is: 'original' | 'acappella' | 'instrumental' |
    # 'remix' (maps to track_audio.variant_tag). Defaulted so the download
    # pipeline keeps emitting 'original'; variant sourcing sets it explicitly.
    variant_tag: str = "original"
    # Edit length (Variant axis): 'regular' (radio/album cut) | 'extended'.
    # Independent of variant_tag (Stem axis).
    edit_tag: str = "regular"


@dataclass(frozen=True)
class SetMediaLink:
    """A resolved set-level (full-mix) URL on one platform."""
    set_id: str
    platform: str               # 'youtube' | 'soundcloud' | 'mixcloud' | 'other'
    url: str


@dataclass(frozen=True)
class SetAudioAsset:
    """A downloaded full-mix audio file for a DJ set."""
    set_audio_id: int | None
    set_id: str
    platform: str
    source_url: str
    path: str
    sha256: str | None
    duration_s: float | None
    sample_rate: int | None
    codec: str | None
    bitrate_kbps: int | None


# URL builders — pure, no I/O.
def youtube_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def soundcloud_api_url(track_id: str) -> str:
    # yt-dlp accepts the api.soundcloud.com form for numeric IDs
    return f"https://api.soundcloud.com/tracks/{track_id}"


def spotify_track_url(track_id: str) -> str:
    return f"https://open.spotify.com/track/{track_id}"


def normalize_set_media_url(raw: str) -> str:
    """Unwrap SC widget URLs (`w.soundcloud.com/player/?url=...`) into the
    underlying api.soundcloud.com URL. Pass other URLs through unchanged.
    Pure — no I/O."""
    if not raw:
        return raw
    if "w.soundcloud.com/player" in raw:
        from urllib.parse import urlparse, parse_qs, unquote
        parsed = urlparse(raw)
        inner = parse_qs(parsed.query).get("url", [None])[0]
        if inner:
            return unquote(inner)
    return raw


# Platform preference order for downloads (higher index = lower priority).
DOWNLOAD_PLATFORM_PRIORITY: Final[tuple[str, ...]] = ("youtube", "soundcloud")
