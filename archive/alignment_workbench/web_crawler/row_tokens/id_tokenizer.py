from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from bs4 import Tag


_INT_RE = re.compile(r"-?\d+")
_TRROW_RE = re.compile(r"\btrRow(\d+)\b")
_PLAYPOS_RE = re.compile(r"playPosition\s*\([^,]+,\s*(\{.*\})\s*\)\s*;?", re.DOTALL)
_MEDIAVIEWER_RE = re.compile(r"new\s+MediaViewer\([^,]+,\s*null,\s*(\{.*\})\s*\)\s*;?", re.DOTALL)


@dataclass(frozen=True, slots=True)
class LinkedIDItem:
    user_name: Optional[str]
    user_href: Optional[str]
    user_followers_text: Optional[str]
    linked_tracklist_href: Optional[str]
    linked_tracklist_text: Optional[str]


@dataclass(frozen=True, slots=True)
class IDTrack:

    tlp_id: Optional[int]
    trno: Optional[int]
    trrow_index: Optional[int]
    is_concurrent: bool
    protected: bool
    rbcst: bool
    cue_seconds: Optional[int]
    cue_label: Optional[str]
    tracknum_display: Optional[str]
    track_text: Optional[str]
    track_tokens: List[str]
    watchers: Optional[int]
    spotify_presave_count: Optional[int]
    linked_count: int
    linked_items: List[LinkedIDItem]
    play_payload: Optional[Dict[str, Any]]
    pending_video: bool
    pending_payload: Optional[Dict[str, Any]]
    has_cue: bool
    has_play_icon: bool
    has_linked_ids: bool


def _safe_int(x: Optional[str]) -> Optional[int]:
    if x is None:
        return None
    x = x.strip()
    if not x:
        return None
    m = _INT_RE.search(x.replace(",", ""))
    return int(m.group(0)) if m else None


def _norm_ws(s: str) -> str:
    return " ".join(s.split())


def _parse_js_object_literal(js_obj: str) -> Dict[str, Any]:
    """
    Best-effort parser for simple JS object literals found in onclick attributes.
    Your samples are JSON-compatible already (single quotes around strings sometimes).
    We normalize to valid JSON and parse.
    """
    if not js_obj:
        return {}
    s = js_obj.strip()

    # Convert single-quoted strings to double-quoted strings (best-effort).
    # This is not a full JS parser, but works well for the shown payloads.
    # Also quote unquoted keys: { idPos: '123', pos: 10 } -> { "idPos": "123", "pos": 10 }
    s = re.sub(r"([{\s,])([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', s)
    s = re.sub(r"'\s*([^']*?)\s*'", lambda m: json.dumps(m.group(1)), s)

    # Remove trailing commas if any
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)

    try:
        return json.loads(s)
    except Exception:
        return {}


def _extract_trrow_index(classes: List[str]) -> Optional[int]:
    for c in classes:
        m = _TRROW_RE.search(c)
        if m:
            return int(m.group(1))
    return None


def _extract_text_or_none(el: Optional[Tag]) -> Optional[str]:
    if not el:
        return None
    t = _norm_ws(el.get_text(" ", strip=True))
    return t if t else None


def _extract_badge_number(badge: Tag) -> Optional[int]:
    # badges often contain an icon + a number, like "<i ...></i>4"
    t = _norm_ws(badge.get_text(" ", strip=True))
    return _safe_int(t)


def _tokenize_track_string(s: str) -> List[str]:
    """
    Lightweight, deterministic tokenization:
    - preserve separators as tokens
    - normalize whitespace
    - split into word-ish chunks and punctuation
    """
    s = _norm_ws(s)
    # Tokens: alphanumerics, @, #, /, :, ., -, parentheses, etc.
    return re.findall(r"[A-Za-z0-9_]+|[@#/.:()-]+", s)


def parse_track_row(outer_div: Tag) -> IDTrack:
    """
    Parse one 1001tracklists track-row <div ...> into an IDTrack dataclass.
    Assumes outer_div is the row div (tlpTog ... tlpItem ...).
    Works for ID rows (data-isid='true') and includes linked-ID metadata.
    """

    # ----- identity / flags -----
    classes = outer_div.get("class", []) or []
    tlp_id = _safe_int(outer_div.get("data-id"))
    trno = _safe_int(outer_div.get("data-trno"))
    is_concurrent = "con" in classes
    trrow_index = _extract_trrow_index(classes)

    protected = (outer_div.get("data-protected") or "").lower() == "true"
    rbcst = (outer_div.get("data-rbcst") or "").lower() == "true"

    # ----- cue seconds (hidden input) -----
    cue_input = outer_div.select_one("div.bPlay input[id^='tlp'][id$='_cue_seconds']")
    cue_seconds = _safe_int(cue_input.get("value") if cue_input else None)

    # ----- tracknumber display and w/ flag -----
    tracknum_span = outer_div.select_one("div.bPlay span[id^='tlp'][id$='_tracknumber_value']")
    tracknum_display = _extract_text_or_none(tracknum_span)

    # ----- cue timestamp label -----
    cue_div = outer_div.select_one("div.bPlay div.cue[id^='cue_']")
    cue_label = _extract_text_or_none(cue_div)  # e.g. "05:22", "1:30:50", or None

    # ----- playPosition payload (if play icon exists) -----
    play_icon = outer_div.select_one("i[id^='tlp_play_'][onclick*='playPosition']")
    play_payload: Dict[str, Any] = {}
    if play_icon:
        onclick = play_icon.get("onclick", "") or ""
        m = _PLAYPOS_RE.search(onclick)
        if m:
            play_payload = _parse_js_object_literal(m.group(1))

    # ----- track content text (the main "trackValue" string) -----
    track_value_span = outer_div.select_one("div.bCont div.fontL span.trackValue")
    track_text = _extract_text_or_none(track_value_span)
    track_tokens = _tokenize_track_string(track_text) if track_text else []

    # ----- watchers + spotify presave -----
    watchers_badge = outer_div.select_one(f"span.badge.hO#wID_{tlp_id}") if tlp_id else None
    watchers = _extract_badge_number(watchers_badge) if watchers_badge else None

    spotify_badge = outer_div.select_one("span.badgeSpotify")
    presave_count = None
    if spotify_badge:
        # commonly "Pre-Save 10"
        t = _norm_ws(spotify_badge.get_text(" ", strip=True))
        # last int is usually the count
        presave_count = _safe_int(t)

    # ----- linked IDs (hidden section) -----
    linked_items: List[LinkedIDItem] = []
    linked_container = outer_div.select_one("div.tgHid")
    if linked_container and tlp_id is not None:
        # tlLinkItem_{id} are the actual linked position blocks
        for item in linked_container.select(f"div.tlLinkItem_{tlp_id}"):
            user_a = item.select_one("span.noWrap a[href^='/user/']")
            user_name = _extract_text_or_none(user_a)
            user_href = user_a.get("href") if user_a else None

            followers_span = item.select_one("span.noWrap span.spL")
            user_followers = _extract_text_or_none(followers_span)  # e.g. "(1.5M)"

            link_a = item.select_one("a[href^='/tracklist/']")
            link_href = link_a.get("href") if link_a else None
            link_text = _extract_text_or_none(link_a)

            linked_items.append(
                LinkedIDItem(
                    user_name=user_name,
                    user_href=user_href,
                    user_followers_text=user_followers,
                    linked_tracklist_href=link_href,
                    linked_tracklist_text=link_text,
                )
            )

    linked_count = len(linked_items)

    # ----- pending video icon (MediaViewer pending: true) -----
    pending_video = False
    pending_payload: Dict[str, Any] = {}
    pen = outer_div.select_one("div#penvl_{}".format(tlp_id)) if tlp_id else None
    if pen:
        onclick = pen.get("onclick", "") or ""
        m = _MEDIAVIEWER_RE.search(onclick)
        if m:
            pending_payload = _parse_js_object_literal(m.group(1))
            pending_video = bool(pending_payload.get("pending") is True or pending_payload.get("pending") == "true")

    # Helpful derived features for ML:
    has_cue = bool(cue_label) or (cue_seconds is not None and cue_seconds > 0)
    has_play_icon = bool(play_payload)
    has_linked_ids = linked_count > 0

    return IDTrack(
        tlp_id=tlp_id,
        trno=trno,
        trrow_index=trrow_index,
        is_concurrent=is_concurrent,
        protected=protected,
        rbcst=rbcst,
        cue_seconds=cue_seconds,
        cue_label=cue_label,
        tracknum_display=tracknum_display,
        track_text=track_text,
        track_tokens=track_tokens,
        watchers=watchers,
        spotify_presave_count=presave_count,
        linked_count=linked_count,
        linked_items=linked_items,
        play_payload=play_payload or None,
        pending_video=pending_video,
        pending_payload=pending_payload or None,
        has_cue=has_cue,
        has_play_icon=has_play_icon,
        has_linked_ids=has_linked_ids,
    )


def id_track_to_dict(track: IDTrack) -> Dict[str, Any]:
    return asdict(track)

