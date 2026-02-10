from enum import IntEnum, Enum
from dataclasses import dataclass
from typing import Optional


try:
    from enum import StrEnum  # type: ignore
except Exception:  # pragma: no cover - py310 fallback
    class StrEnum(str, Enum):
        pass


class TrackType(StrEnum):
    SONG = "song"
    MASHUP = "mashup"
    UNRELEASED = "unreleased"


class PlayType(IntEnum):
    REGULAR = 0
    INSTRUMENTAL = 1
    VOCALS_ONLY = 2


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


@dataclass(frozen=True)
class Track:
    track_id: str
    track_type: TrackType


@dataclass(frozen=True)
class Song:
    song_id: str          # same as tracks.track_id
    url: Optional[str]
    full_title: Optional[str]
    genre: Optional[str]
    labels: Optional[str]
    stream_links: Optional[str]


@dataclass
# Commented out because if the track does not show up anywhere else, the mashup_id 
# defaults to its location in the set. Over 40,000 sets, this becomes subject to overwrites
class MainMashup:
    mashup_id: str        # same as tracks.track_id
    url: Optional[str]
    full_title: Optional[str]
    genre: Optional[str]
    labels: Optional[str]
    stream_links: Optional[str]


@dataclass(frozen=True)
class MashupComponent:
    """
    One row in mashup_components:
      - mashup_id: the mashup track id (tracks.track_id where track_type='mashup')
      - song_id:   the component song track id (tracks.track_id where track_type='song')
      - play_role: 0=full_track, 1=instrumental, 2=vocals_only
    """
    mashup_id: str
    song_id: str
    play_role: PlayType


@dataclass(frozen=True)
class UnreleasedTrack:
    unreleased_id: str    # same as tracks.track_id
    full_title: Optional[str]


@dataclass(frozen=True)
class UnreleasedTrackAliases:
    unreleased_id: str    # same as tracks.track_id
    set_id: str
    track_location: str

@dataclass(frozen=True)
class DJSetTrack:
    """This is essentially a flattened DJ Set containing links to all of the """
    set_id: str
    track_id: str

    main_index: int
    sub_index: int = 0

    time_played: str = ""
    type_played: Optional[PlayType] = None
    played_by: Optional[str] = None


def make_track(track_id: str, track_type: TrackType) -> Track:
    return Track(track_id=track_id, track_type=track_type)


def make_song(
    *,
    song_id: str,
    url: Optional[str] = None,
    full_title: Optional[str] = None,
    genre: Optional[str] = None,
    labels: Optional[str] = None,
    stream_links: Optional[str] = None,
) -> Song:
    return Song(
        song_id=song_id,
        url=url,
        full_title=full_title,
        genre=genre,
        labels=labels,
        stream_links=stream_links,
    )


def make_main_mashup(
    *,
    mashup_id: str,
    url: Optional[str] = None,
    full_title: Optional[str] = None,
    genre: Optional[str] = None,
    labels: Optional[str] = None,
    stream_links: Optional[str] = None,
) -> MainMashup:
    return MainMashup(
        mashup_id=mashup_id,
        url=url,
        full_title=full_title,
        genre=genre,
        labels=labels,
        stream_links=stream_links,
    )


def make_mashup_component(
    *,
    mashup_id: str,
    song_id: str,
    play_role: PlayType,
) -> MashupComponent:
    return MashupComponent(
        mashup_id=mashup_id,
        song_id=song_id,
        play_role=play_role,
    )


def make_unreleased_track(
    *,
    unreleased_id: str,
    full_title: Optional[str] = None,
) -> UnreleasedTrack:
    return UnreleasedTrack(unreleased_id=unreleased_id, full_title=full_title)


def make_unreleased_track_alias(
    *,
    unreleased_id: str,
    set_id: str,
    track_location: str
) -> UnreleasedTrackAliases:
    return UnreleasedTrackAliases(
        unreleased_id=unreleased_id,
        set_id=set_id,
        track_location=track_location
    )


def make_dj_set_track(
    *,
    set_id: str,
    track_id: str,
    main_index: int,
    sub_index: int = 0,
    time_played: str = "",
    type_played: Optional[PlayType] = None,
    played_by: Optional[str] = None,
) -> DJSetTrack:
    return DJSetTrack(
        set_id=set_id,
        track_id=track_id,
        main_index=main_index,
        sub_index=sub_index,
        time_played=time_played,
        type_played=type_played,
        played_by=played_by,
    )



def determine_playtype(title: str) -> PlayType:
    if "acappella" in title.lower() or "acapella" in title.lower() or "accappella" in title.lower():
        return PlayType.VOCALS_ONLY
    elif "instrumental" in title.lower():
        return PlayType.INSTRUMENTAL
    else:
        return PlayType.REGULAR
    
