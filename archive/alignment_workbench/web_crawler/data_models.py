from enum import IntEnum, Enum
from dataclasses import dataclass
from typing import Optional


try:
    from enum import StrEnum  # type: ignore
except Exception:  # pragma: no cover - py310 fallback
    class StrEnum(str, Enum):
        pass

@dataclass(frozen=True)
class DJSet:
    set_id: str
    set_url: Optional[str]

    title: str = ""
    date_played: str = ""
    artists: Optional[str] = None

    creator_name: Optional[str] = None
    creator_url: Optional[str] = None

    views: Optional[int] = None
    ided_tracks: Optional[int] = None
    total_tracks: Optional[int] = None
    likes: Optional[int] = None

    play_time: Optional[str] = None
    styles: Optional[str] = None
    stream_links: Optional[str] = None

    scraped_at: Optional[str] = None


@dataclass(frozen=True)
class DJSetCrawl:
    set_id: str
    set_url: str
    http_status: Optional[int] = None
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    html_sha256: Optional[str] = None
    html_path: Optional[str] = None


@dataclass(frozen=True)
class DJSetMediaLink:
    set_id: str
    platform: str
    url: Optional[str] = None
    id_item: Optional[str] = None
    id_source: Optional[str] = None


@dataclass(frozen=True)
class DJSetTrackMediaLink:
    set_id: str
    tlp_id: Optional[str]
    track_id: Optional[str]
    platform: str
    player_id: Optional[str]
    id_object: Optional[str] = None
    id_item: Optional[str] = None
    id_source: Optional[str] = None
    view_source: Optional[str] = None
    view_item: Optional[str] = None


@dataclass(frozen=True)
class ScrapeFailure:
    set_id: str
    stage: str
    set_url: Optional[str] = None
    track_title: Optional[str] = None
    track_id: Optional[str] = None
    tlp_id: Optional[str] = None
    params_json: Optional[str] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class DJSetRow:
    set_id: str
    row_index: int
    element_id: Optional[str]
    classes: Optional[str]
    data_attrs_json: Optional[str]
    text_excerpt: Optional[str]
    raw_html: Optional[str]

