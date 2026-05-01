from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Literal

from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel, Field

from ._parser import BS_PARSER


RowType = Literal[
    "artist_played",
    "headtext",
    "add_missing_live_video",
    "info_notice",
    "recycle_notice",
    "identical_tracklist_link",
    "identical_tracklist_part",
    "warning_title_wrong",
    "new_added_video",
    "unknown",
]


class TextRowToken(BaseModel):
    row_type: RowType
    raw_html: str
    text: str = ""
    root_attrs: Dict[str, Any] = Field(default_factory=dict)
    links: List[Dict[str, Any]] = Field(default_factory=list)
    icons: List[List[str]] = Field(default_factory=list)
    parsed: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


_RE_ANKER = re.compile(r"anker:\s*'([^']+)'")
_RE_ARTIST_DELETE = re.compile(r"\bartist(\d+)_tracks_delete\s*:\s*true\b")
_RE_USERSUGGEST = re.compile(r"new\s+UserSuggest\s*\(\s*this\s*,\s*\{([^}]*)\}\s*\)")
_RE_IDTL = re.compile(r"\bidTL\s*:\s*'([^']+)'")
_RE_IDTLP = re.compile(r"\bidTLP\s*:\s*(\d+)")
_RE_US_TYPE = re.compile(r"\btype\s*:\s*'([^']+)'")

_RE_IDENTICAL_START_END = re.compile(
    r"\bidentical\s+tracklist\s+(start|end)\b", re.IGNORECASE
)
_RE_IDENTICAL_PART = re.compile(
    r"\bpart:\s*identical\s+tracklist\s+(start|end)\s+track\s*#\s*(\d+)\s+of\b",
    re.IGNORECASE,
)


def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("\xa0", " ")).strip()


def _attrs(tag: Tag) -> Dict[str, Any]:
    # keep only JSON-ish safe attrs (class/id/title/href/translate/onclick/data-*)
    out: Dict[str, Any] = {}
    for k, v in tag.attrs.items():
        if k in {"class", "id", "title", "href", "translate", "onclick"} or k.startswith("data-"):
            out[k] = v
    return out


def _extract_links(root: Tag) -> List[Dict[str, Any]]:
    links = []
    for a in root.find_all("a"):
        links.append(
            {
                "text": _norm_space(a.get_text(" ", strip=True)),
                "href": a.get("href"),
                "translate": a.get("translate"),
                "class": a.get("class") or [],
            }
        )
    return links


def _icon_classes(root: Tag) -> List[List[str]]:
    return [i.get("class") or [] for i in root.find_all("i")]


def _detect_row_type(root: Tag, text: str) -> RowType:
    classes = set(root.get("class") or [])
    has_breakall = bool(root.select_one("span.breakAll"))
    has_frow = bool(root.select_one("span.fRow"))

    # Warning row
    if root.select_one("i.fa-exclamation-triangle"):
        return "warning_title_wrong"

    # Recycle notice
    if root.select_one("i.fa-recycle") or "identical tracklist(s)" in text.lower():
        return "recycle_notice"

    # "New added video track(s)"
    if text.lower() == "new added video track(s)":
        return "new_added_video"

    # Add missing live video CTA rows
    if "interaction" in classes and root.select_one("div.hOO") and "add missing live video" in text.lower():
        return "add_missing_live_video"

    # Info notice (info-circle) rows
    if root.select_one("i.fa-info-circle"):
        return "info_notice"

    # Artist played rows
    if has_frow and "played:" in text.lower() and root.select_one("i.fa-trash"):
        return "artist_played"

    # Headtext rows (stage labels, segment labels, etc.)
    if has_breakall:
        return "headtext"

    # Identical link rows (start/end with a tracklist link)
    if _RE_IDENTICAL_START_END.search(text) and root.select_one("a[href*='/tracklist/']"):
        return "identical_tracklist_link"

    # Identical part rows (start/end track #N of some tracklist)
    if _RE_IDENTICAL_PART.search(text) and root.select_one("a[href*='/tracklist/']"):
        return "identical_tracklist_part"

    return "unknown"


def _parse_artist_played(root: Tag) -> Dict[str, Any]:
    # Example:
    # <span class="fRow">
    #   <a href="/dj/martingarrix/">Martin Garrix</a> <span>&amp;</span><a ...>Alesso</a>
    #   <span>played:</span> <i class="... fa-trash ..." id="...artistTracksDel" onclick="... anker:'tlp_1347438', artist3215_tracks_delete:true ...">
    # </span>
    frow = root.select_one("span.fRow")
    artists = []
    if frow:
        # All DJ links in the fRow before "played:"; grabbing all <a> is OK here.
        for a in frow.find_all("a"):
            href = a.get("href")
            # only keep DJ links (your examples are /dj/<slug>/index.html)
            if href and "/dj/" in href:
                artists.append({"name": _norm_space(a.get_text(" ", strip=True)), "href": href})

    trash = root.select_one("i.fa-trash")
    onclick = trash.get("onclick") if trash else None
    title = trash.get("title") if trash else None
    icon_id = trash.get("id") if trash else None

    anker = None
    delete_artist_id = None
    if onclick:
        m = _RE_ANKER.search(onclick)
        if m:
            anker = m.group(1)
        m2 = _RE_ARTIST_DELETE.search(onclick)
        if m2:
            delete_artist_id = int(m2.group(1))

    return {
        "artists": artists,
        "played_label_present": True,
        "delete": {
            "icon_id": icon_id,
            "title": title,
            "anker": anker,
            "artist_id_for_delete": delete_artist_id,
            "onclick": onclick,
        },
    }


def _parse_add_missing_live(root: Tag) -> Dict[str, Any]:
    # Example:
    # onclick="new UserSuggest(this, { type:'add_videotrack', idTL: '5uy5c89', idTLP: 1000 } ).show();"
    box = root.select_one("div.hOO")
    onclick = box.get("onclick") if box else None
    el_id = box.get("id") if box else root.get("id")
    title = box.get("title") if box else root.get("title")

    payload = {"type": None, "idTL": None, "idTLP": None, "raw": None}
    if onclick:
        payload["raw"] = onclick
        mt = _RE_US_TYPE.search(onclick)
        if mt:
            payload["type"] = mt.group(1)
        midtl = _RE_IDTL.search(onclick)
        if midtl:
            payload["idTL"] = midtl.group(1)
        midtp = _RE_IDTLP.search(onclick)
        if midtp:
            payload["idTLP"] = int(midtp.group(1))

    return {
        "element_id": el_id,
        "title": title,
        "user_suggest": payload,
        "cta_text": _norm_space(root.get_text(" ", strip=True)),
    }


def _parse_headtext(root: Tag) -> Dict[str, Any]:
    span = root.select_one("span.breakAll")
    return {
        "headtext_id": span.get("id") if span else None,
        "text": _norm_space(span.get_text(" ", strip=True) if span else root.get_text(" ", strip=True)),
    }


def _parse_notice(root: Tag) -> Dict[str, Any]:
    # info/recycle/warning etc: keep icon classes + text
    return {
        "text": _norm_space(root.get_text(" ", strip=True)),
        "icons": _icon_classes(root),
    }


def _parse_identical_link(root: Tag) -> Dict[str, Any]:
    text = _norm_space(root.get_text(" ", strip=True))
    which = None
    m = _RE_IDENTICAL_START_END.search(text)
    if m:
        which = m.group(1).lower()

    a = root.select_one("a[href*='/tracklist/']")
    link = None
    if a:
        link = {
            "text": _norm_space(a.get_text(" ", strip=True)),
            "href": a.get("href"),
        }

    return {
        "which": which,  # "start" or "end"
        "link": link,
        "text": text,
    }


def _parse_identical_part(root: Tag) -> Dict[str, Any]:
    text = _norm_space(root.get_text(" ", strip=True))
    m = _RE_IDENTICAL_PART.search(text)

    which = track_no = None
    if m:
        which = m.group(1).lower()
        track_no = int(m.group(2))

    # Often: first <a> is DJ, second is tracklist
    dj = None
    a_dj = root.select_one("a[href*='/dj/']")
    if a_dj:
        dj = {"name": _norm_space(a_dj.get_text(" ", strip=True)), "href": a_dj.get("href")}

    a_tl = root.select_one("a[href*='/tracklist/']")
    tracklist = None
    if a_tl:
        tracklist = {"text": _norm_space(a_tl.get_text(" ", strip=True)), "href": a_tl.get("href")}

    return {
        "which": which,          # "start" or "end"
        "track_number": track_no,
        "dj": dj,
        "tracklist": tracklist,
        "text": text,
    }


def parse_bItmH_row(html: str) -> TextRowToken:
    """
    Tokenizes ONE <div class="bItmH ...">...</div> row into a structured object.
    If you pass a fragment that contains multiple root divs, use tokenize_bItmH_rows().
    """
    soup = BeautifulSoup(html, BS_PARSER)
    root = soup.find("div")
    if not root or not isinstance(root, Tag):
        return TextRowToken(
            row_type="unknown",
            raw_html=html,
            error="no_root_div",
        )

    text = _norm_space(root.get_text(" ", strip=True))
    row_type: RowType = _detect_row_type(root, text)

    base = TextRowToken(
        row_type=row_type,
        text=text,
        root_attrs=_attrs(root),
        links=_extract_links(root),
        icons=_icon_classes(root),
        raw_html=html,
    )

    if row_type == "artist_played":
        base.parsed = _parse_artist_played(root)
    elif row_type == "add_missing_live_video":
        base.parsed = _parse_add_missing_live(root)
    elif row_type == "headtext":
        base.parsed = _parse_headtext(root)
    elif row_type in ("info_notice", "recycle_notice", "warning_title_wrong", "new_added_video"):
        base.parsed = _parse_notice(root)
    elif row_type == "identical_tracklist_link":
        base.parsed = _parse_identical_link(root)
    elif row_type == "identical_tracklist_part":
        base.parsed = _parse_identical_part(root)
    else:
        base.parsed = {}

    return base
