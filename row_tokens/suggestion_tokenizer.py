from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

from bs4 import BeautifulSoup, Tag  # pip install beautifulsoup4

# -----------------------------
# Helpers
# -----------------------------

_TIME_RE = re.compile(r"^\s*(\d+):(\d{2})(?::(\d{2}))?\s*$")  # mm:ss or h:mm:ss
_SUG_ID_RE = re.compile(r"^(?:sug_)?(\d+)$")
_TLP_CLASS_RE = re.compile(r"\btlp_(\d+)\b")
_TRACK_PAGE_RE = re.compile(r"^/track/([^/]+)/(.+?)/index\.html$")

def _strip(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s2 = re.sub(r"\s+", " ", s).strip()
    return s2 or None

def parse_time_to_seconds(t: Optional[str]) -> Optional[int]:
    """Parse 'mm:ss' or 'h:mm:ss' to seconds. Returns None if not parseable."""
    if not t:
        return None
    m = _TIME_RE.match(t.strip())
    if not m:
        return None
    a, b, c = m.groups()
    if c is None:
        mm, ss = int(a), int(b)
        return mm * 60 + ss
    hh, mm, ss = int(a), int(b), int(c)
    return hh * 3600 + mm * 60 + ss

def _parse_onclick_kwargs(onclick: str) -> Dict[str, str]:
    """
    Pulls simple key:'value' pairs out of JS object literals inside onclick.
    Example: playPosition(this, { cue: '2735',idPlayer: '',idPlayerSource: '' } );
    """
    if not onclick:
        return {}
    # very forgiving: key : 'value' OR key:'value'
    pairs = re.findall(r"(\w+)\s*:\s*'([^']*)'", onclick)
    return {k: v for k, v in pairs}

def _first(tag: Tag, selector: str) -> Optional[Tag]:
    out = tag.select_one(selector)
    return out if isinstance(out, Tag) else None

def _all(tag: Tag, selector: str) -> List[Tag]:
    return [t for t in tag.select(selector) if isinstance(t, Tag)]

def _get_int_attr(tag: Tag, attr: str) -> Optional[int]:
    v = tag.get(attr)
    if v is None:
        return None
    try:
        return int(str(v))
    except ValueError:
        return None

def _get_bool_attr(tag: Tag, attr: str) -> Optional[bool]:
    if attr not in tag.attrs:
        return None
    v = str(tag.get(attr)).strip().lower()
    if v in ("true", "1", "yes"):
        return True
    if v in ("false", "0", "no"):
        return False
    return None

# -----------------------------
# Output schema (pipeline-friendly)
# -----------------------------

@dataclass(frozen=True)
class SuggestionRow:
    # Identity / routing
    sug_id: Optional[int]
    data_type: Optional[int]          # from data-type
    tlp_id: Optional[int]             # from data-tlp or tlp_XXXX class
    pos: Optional[int]                # data-pos (position in setlist)
    is_track: Optional[bool]          # data-track
    has_value: Optional[bool]         # data-value
    nospam: Optional[bool]            # data-nospam

    # Suggester
    suggester_kind: Optional[str]     # "user" | "guest" | None
    suggester_user_id: Optional[int]  # data-user if present
    suggester_guest_id: Optional[int] # data-guest if present
    suggester_name: Optional[str]     # visible username, or "Guest"
    suggester_profile_path: Optional[str]  # /user/.../index.html
    suggestion_timestamp: Optional[str]     # "[YY-MM-DD hh:mm:ss]" string as shown

    # Cue / timing
    cue_text: Optional[str]           # displayed cue, e.g. "45:35"
    cue_seconds: Optional[int]        # parsed
    play_cue_seconds: Optional[int]   # from onclick playPosition cue:'...'

    # Track semantic payload (only when present)
    track_display: Optional[str]      # full visible line (artist - title + remix)
    artist_title: Optional[str]       # heuristically the same as track_display (minus labels)
    track_page_path: Optional[str]    # /track/<slug>/<name>/index.html
    track_slug: Optional[str]         # e.g. "2rxx9slp"
    track_id_numeric: Optional[int]   # mediaRow data-trackid if present

    # Labels (can be multiple)
    labels: Tuple[Tuple[str, Optional[str]], ...]  # (label_name, label_path)

    # Flags / special
    is_id_remix: Optional[bool]       # "(ID Remix)" marker or mediaRow data-type=200 or data-remix=1 (weak signal)
    is_remix: Optional[bool]          # mediaRow data-remix=1

    # Media availability
    has_youtube: bool
    has_soundcloud: bool
    has_spotify: bool
    has_apple: bool
    has_affiliate: bool
    has_live_video: bool

    # Polls (if present)
    poll_correct: Optional[int]
    poll_not_correct: Optional[int]
    poll_unsure: Optional[int]

    # Search / misc
    google_search_url: Optional[str]
    raw_text: Optional[str]           # fallback for debugging

# -----------------------------
# Core parser for one <div class="bItm ... sugTog ..."> row
# -----------------------------

def parse_suggestion_row(row: Tag) -> SuggestionRow:
    """
    Robustly parse 1001TL-ish suggestion rows (data-type distinguishes semantics).
    Works for:
      - type=4: suggested track insert ("w/ <track>")
      - type=14: corrected cue time ("correct cue time is <time>")
      - plus partial rows with missing media/track links.
    """
    if not isinstance(row, Tag):
        raise TypeError("row must be a bs4.Tag")

    # --- identity
    sug_id = None
    rid = row.get("id")
    if rid:
        m = _SUG_ID_RE.match(str(rid))
        if m:
            sug_id = int(m.group(1))

    data_type = _get_int_attr(row, "data-type")
    tlp_id = _get_int_attr(row, "data-tlp")
    pos = _get_int_attr(row, "data-pos")
    is_track = _get_bool_attr(row, "data-track")
    has_value = _get_bool_attr(row, "data-value")
    nospam = _get_bool_attr(row, "data-nospam")

    # sometimes tlp is only in class "tlp_XXXX"
    if tlp_id is None:
        classes = " ".join(row.get("class", []) or [])
        cm = _TLP_CLASS_RE.search(classes)
        if cm:
            tlp_id = int(cm.group(1))

    # --- suggester
    suggester_user_id = _get_int_attr(row, "data-user")
    suggester_guest_id = _get_int_attr(row, "data-guest")
    if suggester_user_id is not None:
        suggester_kind = "user"
    elif suggester_guest_id is not None:
        suggester_kind = "guest"
    else:
        suggester_kind = None

    # visible username + timestamp live in .wRow
    wrow = _first(row, "div.wRow")
    suggestion_timestamp = None
    suggester_name = None
    suggester_profile_path = None
    if wrow:
        ts = _first(wrow, "span.tlEdit")
        suggestion_timestamp = _strip(ts.get_text()) if ts else None

        # If there's an <a href="/user/..."> use that
        a_user = _first(wrow, "a[href^='/user/']")
        if a_user:
            suggester_profile_path = str(a_user.get("href"))
            # username often is in a preceding <span>, but safest: visible text nodes near it
        # Name heuristics:
        # - Many have <span>FoXy</span>
        # - Some are literal "Guest" as a text node
        spans = _all(wrow, "span")
        # filter out tlEdit and icon spans
        cand = []
        for sp in spans:
            if "tlEdit" in (sp.get("class") or []):
                continue
            txt = _strip(sp.get_text())
            if txt and not txt.startswith("[poll:"):
                cand.append(txt)
        if cand:
            # last non-timestamp span tends to be username
            suggester_name = cand[-1]
        else:
            # maybe "Guest" appears as a text node
            t = _strip(wrow.get_text(" ", strip=True))
            if t and "Guest" in t:
                suggester_name = "Guest"

    # --- cue display (two variants)
    cue_text = None
    cue_seconds = None
    # type=4 usually has <span class="cue ...">MM:SS</span>
    cue_span = _first(row, "div.bPlay span.cue")
    if cue_span:
        cue_text = _strip(cue_span.get_text())
        cue_seconds = parse_time_to_seconds(cue_text)

    # type=14: cue text shown inside #sugXXXX_value span.italic
    if data_type == 14:
        italic_time = _first(row, "div#{} span.italic".format(row.get("id", "")).replace("sug_", "sug"))  # not reliable
        if italic_time is None:
            italic_time = _first(row, "div.fontL span.italic")
        if italic_time:
            cue_text = _strip(italic_time.get_text())
            cue_seconds = parse_time_to_seconds(cue_text)

    # --- playPosition cue seconds in onclick (if any)
    play_cue_seconds = None
    play_icon = _first(row, "i[id^='tlp_play_sug_']")
    if play_icon and play_icon.get("onclick"):
        kwargs = _parse_onclick_kwargs(str(play_icon.get("onclick")))
        if "cue" in kwargs and kwargs["cue"].isdigit():
            play_cue_seconds = int(kwargs["cue"])

    # --- track display + link
    track_display = None
    artist_title = None
    track_page_path = None
    track_slug = None

    # Most type=4: within div.fontL#sugXXXX_value; track title often in span.blueTxt
    fontL = _first(row, "div.bCont div.fontL")
    if fontL:
        # "w/ <track>" header lives outside; fontL mostly the main payload
        track_display = _strip(fontL.get_text(" ", strip=True))
        artist_title = track_display

        # Find first /track/... link
        a_track = _first(fontL, "a[href^='/track/']")
        if a_track and a_track.get("href"):
            track_page_path = str(a_track.get("href"))
            tm = _TRACK_PAGE_RE.match(track_page_path)
            if tm:
                track_slug = tm.group(1)

    # Some rows (e.g. mashups) have no /track link; keep text only.

    # --- labels (may be multiple: e.g. MAD DECENT / FREE)
    labels: List[Tuple[str, Optional[str]]] = []
    if fontL:
        for lab in _all(fontL, "span.trackLabel"):
            name = _strip(lab.get_text(" ", strip=True))
            # label link might be inside the span
            a_lab = _first(lab, "a[href^='/label/']")
            path = str(a_lab.get("href")) if a_lab and a_lab.get("href") else None
            if name:
                labels.append((name, path))

    # --- media row + availability
    media_row = _first(row, "div.mediaRow")
    track_id_numeric = _get_int_attr(media_row, "data-trackid") if media_row else None
    is_remix = None
    if media_row and "data-remix" in media_row.attrs:
        is_remix = _get_bool_attr(media_row, "data-remix")

    # Detect ID Remix (strong signal is "(ID Remix)" marker; also sometimes mediaRow data-type=200)
    is_id_remix = None
    if fontL:
        if _first(fontL, "span.italic.redTxt") and "ID Remix" in fontL.get_text(" ", strip=True):
            is_id_remix = True
    if is_id_remix is None and media_row is not None:
        dt_media = _get_int_attr(media_row, "data-type")
        if dt_media == 200:
            is_id_remix = True
    # weak signal: remix flag in media row (don't equate remix == ID remix)
    if is_id_remix is None and is_remix is True:
        is_id_remix = False  # explicitly not ID remix, but it is a remix

    # Media presence heuristics by icon classes
    def _has_icon(sel: str) -> bool:
        return _first(row, sel) is not None

    has_youtube = _has_icon("i.fa-video-camera") or _has_icon("i.fa-youtube-play")
    has_soundcloud = _has_icon("i.fa-soundcloud")
    has_spotify = _has_icon("i.fa-spotify")
    has_apple = _has_icon("i.fa-apple")
    has_affiliate = _has_icon("i.fa-shopping-cart")  # affiliate player icon
    has_live_video = _has_icon("span[id^='sugplay_vl_'] i.fa-video-camera")

    # --- poll parsing
    poll_correct = poll_not_correct = poll_unsure = None
    poll_div = _first(row, "div[id^='pollres_']")
    if poll_div:
        # Looks like: [poll:<div class="greenTxt iB">3</div>/<div class="redTxt iB">0</div>/<div class="blueTxt iB">0</div>]
        g = _first(poll_div, "div.greenTxt")
        r = _first(poll_div, "div.redTxt")
        b = _first(poll_div, "div.blueTxt")

        def _num(t: Optional[Tag]) -> Optional[int]:
            if not t:
                return None
            # sometimes includes nested "(1)" guest polls; keep the leading integer
            m = re.search(r"(\d+)", t.get_text(" ", strip=True))
            return int(m.group(1)) if m else None

        poll_correct = _num(g)
        poll_not_correct = _num(r)
        poll_unsure = _num(b)

    # --- google search link (often in mediaRow <a href="https://www.google.com/search?q=...">)
    google_search_url = None
    a_google = _first(row, "a[href^='https://www.google.com/search?q=']")
    if a_google and a_google.get("href"):
        google_search_url = str(a_google.get("href"))

    raw_text = _strip(row.get_text(" ", strip=True))

    return SuggestionRow(
        sug_id=sug_id,
        data_type=data_type,
        tlp_id=tlp_id,
        pos=pos,
        is_track=is_track,
        has_value=has_value,
        nospam=nospam,
        suggester_kind=suggester_kind,
        suggester_user_id=suggester_user_id,
        suggester_guest_id=suggester_guest_id,
        suggester_name=suggester_name,
        suggester_profile_path=suggester_profile_path,
        suggestion_timestamp=suggestion_timestamp,
        cue_text=cue_text,
        cue_seconds=cue_seconds,
        play_cue_seconds=play_cue_seconds,
        track_display=track_display if data_type != 14 else None,  # type 14 isn't a track suggestion
        artist_title=artist_title if data_type != 14 else None,
        track_page_path=track_page_path if data_type != 14 else None,
        track_slug=track_slug if data_type != 14 else None,
        track_id_numeric=track_id_numeric if data_type != 14 else None,
        labels=tuple(labels) if data_type != 14 else tuple(),
        is_id_remix=is_id_remix if data_type != 14 else None,
        is_remix=is_remix if data_type != 14 else None,
        has_youtube=has_youtube,
        has_soundcloud=has_soundcloud,
        has_spotify=has_spotify,
        has_apple=has_apple,
        has_affiliate=has_affiliate,
        has_live_video=has_live_video,
        poll_correct=poll_correct,
        poll_not_correct=poll_not_correct,
        poll_unsure=poll_unsure,
        google_search_url=google_search_url,
        raw_text=raw_text,
    )
