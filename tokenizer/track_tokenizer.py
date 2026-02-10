from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from bs4 import BeautifulSoup, Tag


# -----------------------------
# Data model (pipeline-friendly)
# -----------------------------

@dataclass
class LabelRef:
    name: str
    href: Optional[str] = None


@dataclass
class UserRef:
    username: str
    profile_href: Optional[str] = None
    reputation_text: Optional[str] = None  # e.g. "(9.2k)"


@dataclass
class MediaFlags:
    youtube: bool = False
    soundcloud: bool = False
    spotify: bool = False
    apple: bool = False
    affiliate: bool = False


@dataclass
class TrackRow:
    # identity / position
    row_dom_id: Optional[str] = None            # e.g. "tlp_1347425"
    data_id: Optional[int] = None               # e.g. 1347425
    data_trno: Optional[int] = None             # e.g. 0, 1, 2...

    # deterministic flags (new)
    is_ided: bool = False                       # data-isided="true"
    is_concurrent: bool = False                 # class contains "con" or track number shows "w/"
    is_remixish: bool = False                   # remix/rework/acappella/alt version flag
    version_tag: Optional[str] = None           # "Remix" | "Rework" | "Acappella" | "AltVersion" | None

    # ordering / timing
    track_number_raw: Optional[str] = None      # "01", "w/", etc (as shown)
    cue_seconds: Optional[int] = None           # hidden input value
    cue_timecode: Optional[str] = None          # "10:34" etc
    cue_time_seconds: Optional[int] = None      # parsed from cue_timecode when present

    #  links
    track_key: Optional[str] = None             # e.g. "1hqwytzf" (string id from data-trackid / tr_*)
    track_page_href: Optional[str] = None       # "/track/.../index.html"
    google_query_href: Optional[str] = None     # the Google search link if present

    # schema/meta (stable)
    title: Optional[str] = None                 # track title
    artists: List[str] = field(default_factory=list)
    full_name: Optional[str] = None             # meta itemprop=name (often "Artist - Title")
    genre: Optional[str] = None
    duration_iso: Optional[str] = None          # e.g. "PT6M55S"
    duration_seconds: Optional[int] = None
    publisher_labels: List[LabelRef] = field(default_factory=list)

    # community signals
    plays: Optional[int] = None                 # total tracklist plays
    iders: List[UserRef] = field(default_factory=list)

    # media
    media_track_numeric_id: Optional[int] = None  # data-trackid from media row (numeric)
    is_remix_row: bool = False                    # data-remix="1" on media row
    media_flags: MediaFlags = field(default_factory=MediaFlags)
    spotify_cta_text: Optional[str] = None        # "Save 18", "Pre-Save 3", etc
    spotify_cta_count: Optional[int] = None


# -----------------------------
# Helpers
# -----------------------------

_INT_RE = re.compile(r"[-+]?\d+")


def _as_int(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    m = _INT_RE.search(s)
    return int(m.group(0)) if m else None


def _clean_text(s: str) -> str:
    # collapse whitespace and non-breaking spaces
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


def _meta_content(container: Tag, itemprop: str) -> Optional[str]:
    m = container.find("meta", attrs={"itemprop": itemprop})
    return m.get("content") if m and m.has_attr("content") else None


def parse_iso8601_duration_to_seconds(iso: Optional[str]) -> Optional[int]:
    if not iso:
        return None
    iso = iso.strip().upper()
    m = re.fullmatch(
        r"PT"
        r"(?:(?P<h>\d+)H)?"
        r"(?:(?P<m>\d+)M)?"
        r"(?:(?P<s>\d+)S)?",
        iso,
    )
    if not m:
        return None
    h = int(m.group("h") or 0)
    mi = int(m.group("m") or 0)
    se = int(m.group("s") or 0)
    return h * 3600 + mi * 60 + se


def parse_timecode_to_seconds(tc: Optional[str]) -> Optional[int]:
    if not tc:
        return None
    tc = _clean_text(tc)
    if not tc or ":" not in tc:
        return None
    parts = tc.split(":")
    if not all(p.isdigit() for p in parts):
        return None
    nums = [int(p) for p in parts]
    if len(nums) == 2:
        mm, ss = nums
        return mm * 60 + ss
    if len(nums) == 3:
        hh, mm, ss = nums
        return hh * 3600 + mm * 60 + ss
    return None


def _extract_track_key(row: Tag) -> Optional[str]:
    # Prefer outer data-trackid (string key like "1hqwytzf")
    v = row.get("data-trackid")
    if v:
        return str(v)

    # Fallback: span id="tr_<key>"
    tr = row.find(id=re.compile(r"^tr_"))
    if tr and tr.has_attr("id"):
        return tr["id"].split("_", 1)[1]
    return None


def _extract_artist_names(track_value_span: Optional[Tag]) -> List[str]:
    if not track_value_span:
        return []

    text = _clean_text(track_value_span.get_text(" ", strip=True))

    # Extract left side of "Artist - Title"
    artist_part = text.split(" - ", 1)[0] if " - " in text else text

    # Drop featuring suffix from artist-part for canonical artist list
    if " ft. " in artist_part:
        artist_part = artist_part.split(" ft. ", 1)[0]

    return [_clean_text(x) for x in re.split(r"\s*&\s*", artist_part) if _clean_text(x)]


def _derive_version_flags(row: Tag, media_row: Optional[Tag]) -> tuple[bool, Optional[str]]:
    """
    Returns (is_remixish, version_tag)
    version_tag: "Acappella" | "Rework" | "Remix" | "AltVersion" | None
    """
    remix_flag = bool(media_row is not None and (media_row.get("data-remix") or "") == "1")

    # Strong explicit rework marker (recycle icon link)
    has_recycle_rework = bool(row.select_one("a[title^='rework of track']"))

    # Text scan (cheap + robust)
    text_blob = row.get_text(" ", strip=True).lower()
    has_acappella = "acappella" in text_blob
    has_rework = " rework" in text_blob
    has_remix = " remix" in text_blob

    is_remixish = remix_flag or has_acappella or has_rework or has_recycle_rework or has_remix

    version_tag: Optional[str] = None
    if has_acappella:
        version_tag = "Acappella"
    elif has_rework or has_recycle_rework:
        version_tag = "Rework"
    elif has_remix:
        version_tag = "Remix"
    elif remix_flag:
        version_tag = "AltVersion"

    return is_remixish, version_tag


# -----------------------------
# Main extractor
# -----------------------------

def parse_track_row(row_html: str) -> TrackRow:
    """
    Parse a single 1001tracklists track row into a structured TrackRow.
    Only extracts fields currently present in TrackRow.
    """
    soup = BeautifulSoup(row_html, "html.parser")
    row = soup.find("div")
    if not row:
        return TrackRow()

    tr = TrackRow()

    # outer identity
    tr.row_dom_id = row.get("id")
    tr.data_id = _as_int(row.get("data-id"))
    tr.data_trno = _as_int(row.get("data-trno"))
    tr.track_key = _extract_track_key(row)

    # deterministic: IDed?
    tr.is_ided = (row.get("data-isided") == "true")

    # deterministic: "played together" / w/ ?
    #  - class contains "con"
    #  - OR bPlay tracknumber shows "w/"
    tr.is_concurrent = ("con" in set(list(row.get("class") or [])))

    # bPlay block: cue + track number
    bplay = row.find("div", class_="bPlay")
    if bplay:
        cue_input = bplay.find("input", id=re.compile(r"_cue_seconds$"))
        tr.cue_seconds = _as_int(cue_input.get("value") if cue_input else None)

        tracknum_span = bplay.find("span", id=re.compile(r"_tracknumber_value$"))
        if tracknum_span:
            tr.track_number_raw = _clean_text(tracknum_span.get_text(" ", strip=True)) or None
            if tr.track_number_raw == "w/":
                tr.is_concurrent = True

        cue_div = bplay.find("div", id=re.compile(r"^cue_"))
        if cue_div:
            tr.cue_timecode = _clean_text(cue_div.get_text(" ", strip=True)) or None
            tr.cue_time_seconds = parse_timecode_to_seconds(tr.cue_timecode)

    # content block (schema.org MusicRecording)
    content = row.find("div", itemtype=re.compile(r"schema\.org/MusicRecording"))
    if content:
        tr.full_name = _meta_content(content, "name")
        tr.genre = _meta_content(content, "genre")
        tr.duration_iso = _meta_content(content, "duration")
        tr.duration_seconds = parse_iso8601_duration_to_seconds(tr.duration_iso)
        tr.track_page_href = _meta_content(content, "url")

        track_value = content.find("span", class_=re.compile(r"\btrackValue\b"))
        if track_value:
            tv_text = _clean_text(track_value.get_text(" ", strip=True))
            if " - " in tv_text:
                _, title_part = tv_text.split(" - ", 1)
                tr.title = re.sub(r"\s*\(.+\)\s*$", "", title_part).strip() or None
            else:
                tr.title = tv_text or None

            tr.artists = _extract_artist_names(track_value)

        # labels: parse the labeldata block (can contain multiple labels + parentheses)
        label_block = content.find(id=re.compile(r"_labeldata$"))
        if label_block:
            for a in label_block.find_all("a", href=True):
                name = _clean_text(a.get_text(" ", strip=True))
                if name:
                    tr.publisher_labels.append(LabelRef(name=name, href=a["href"]))
            if not tr.publisher_labels:
                txt = _clean_text(label_block.get_text(" ", strip=True))
                if txt:
                    tr.publisher_labels.append(LabelRef(name=txt))

    # plays + IDer(s)
    wrow = row.find("div", class_=re.compile(r"\bwRow\b"))
    if wrow:
        play_span = wrow.find("span", class_=re.compile(r"pcTr"))
        if play_span:
            tr.plays = _as_int(_clean_text(play_span.get_text(" ", strip=True)))

        for ider_span in wrow.find_all("span", title=re.compile(r"IDer", re.I)):
            username = _clean_text(ider_span.get_text(" ", strip=True))

            # username text often includes "(9.2k)" inside, so split it out
            rep = None
            rep_m = re.search(r"\(\s*[\d\.]+\s*[kKmM]?\s*\)", username)
            if rep_m:
                rep = rep_m.group(0)
                username = _clean_text(username.replace(rep, ""))

            a = ider_span.find("a", href=True)
            href = a["href"] if a else None

            # avoid duplicates (span nesting can cause repeats)
            if username and all(u.username != username for u in tr.iders):
                tr.iders.append(UserRef(username=username, profile_href=href, reputation_text=rep))

    # media row: icons and ids
    media_row = row.find("div", class_=re.compile(r"\bmediaRow\b"))
    if media_row:
        tr.media_track_numeric_id = _as_int(media_row.get("data-trackid"))
        tr.is_remix_row = str(media_row.get("data-remix") or "") == "1"

        for i in media_row.find_all("i"):
            classes = set(i.get("class") or [])
            if "fa-video-camera" in classes:
                tr.media_flags.youtube = True
            if "fa-soundcloud" in classes:
                tr.media_flags.soundcloud = True
            if "fa-spotify" in classes:
                tr.media_flags.spotify = True
            if "fa-apple" in classes:
                tr.media_flags.apple = True
            if "fa-shopping-cart" in classes:
                tr.media_flags.affiliate = True

        # Spotify CTA badge text like "Save 18" / "Pre-Save 3"
        badge = media_row.find("span", class_=re.compile(r"badgeSpotify"))
        if badge:
            tr.spotify_cta_text = _clean_text(badge.get_text(" ", strip=True)) or None
            tr.spotify_cta_count = _as_int(tr.spotify_cta_text)

        # Google link
        g = media_row.find("a", href=re.compile(r"google\.com/search\?q=", re.I))
        if g and g.has_attr("href"):
            tr.google_query_href = g["href"]

    # deterministic: remix/rework/acappella flags
    tr.is_remixish, tr.version_tag = _derive_version_flags(row, media_row)

    return tr
