from __future__ import annotations

import re
import json
from dataclasses import dataclass
from typing import Any, List, Optional

from bs4 import BeautifulSoup, Tag
import logging

from archive.alignment_workbench.web_crawler.config import Result
from archive.alignment_workbench.web_crawler.data_models import DJSetMediaLink, DJSetTrackMediaLink, ScrapeFailure, DJSetRow


@dataclass
class ScrapedSetData:
    """Data object returning the result of a set scrape."""
    set_media_links: list[DJSetMediaLink]
    track_media_links: list[DJSetTrackMediaLink]
    failures: list[ScrapeFailure]
    rows: list[DJSetRow]


MEDIA_LINK_ID_RE = re.compile(r"^mediaLink\d+$")

YOUTUBE_EMBED_RE = re.compile(
    r'(?:youtube(?:-nocookie)?\.com\/embed\/)([A-Za-z0-9_-]{11})'
)

# Extract src="..." from an iframe HTML string inside <script>
IFRAME_SRC_IN_SCRIPT_RE = re.compile(
    r"<iframe\b[^>]*\bsrc=['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)


def youtube_embed_to_watch(url: str) -> Optional[str]:
    m = YOUTUBE_EMBED_RE.search(url)
    if not m:
        return None
    video_id = m.group(1)
    return f"https://www.youtube.com/watch?v={video_id}"


def html_unescape(s: str) -> str:
    # bs4 can do this too, but keep it minimal
    return s.replace("&amp;", "&")


def extract_media_url_from_mediaLink_div(media_link_div: Tag) -> Optional[str]:
    """
    Given <div id="mediaLinkNNNN..."> ... </div>, returns a playable URL:
    - iframe src if present
    - schema.org meta embedUrl if present
    - iframe src embedded inside iFrameBuffer script
    - YouTube embed rewritten to watch URL; others unchanged
    """
    base_id = media_link_div.get("id")
    player_div = media_link_div.find("div", id=f"{base_id}_player") if base_id else None
    container = player_div if player_div else media_link_div

    iframe = container.find("iframe")
    if iframe and iframe.get("src"):
        src = html_unescape(iframe["src"])
        return youtube_embed_to_watch(src) or src

    embed_meta = container.select_one('meta[itemprop="embedUrl"]')
    if embed_meta and embed_meta.get("content"):
        src = html_unescape(embed_meta["content"])
        return youtube_embed_to_watch(src) or src

    for script in container.find_all("script"):
        text = script.string or script.get_text() or ""
        m = IFRAME_SRC_IN_SCRIPT_RE.search(text)
        if m:
            src = html_unescape(m.group(1))
            return youtube_embed_to_watch(src) or src

    return None


def extract_set_media_links(soup: BeautifulSoup, set_id: str) -> list[DJSetMediaLink]:
    refs: list[DJSetMediaLink] = []
    media_link_divs = soup.find_all("div", id=MEDIA_LINK_ID_RE)

    for media_link_div in media_link_divs:
        url = extract_media_url_from_mediaLink_div(media_link_div)
        if not url:
            continue
        refs.append(DJSetMediaLink(
            set_id=set_id,
            platform=infer_platform_from_url(url),
            url=url,
            id_item=media_link_div.get("data-idmedia"),
            id_source=media_link_div.get("data-idsource"),
        ))

    return refs


SOURCE_TYPE = {
    "1": "beatport",
    "2": "apple",
    "4": "traxsource",
    "10": "soundcloud",
    "13": "youtube",
    "36": "spotify",
    "affiliate": "affiliate",
}

DOWNLOAD_PRIORITY = {
    "spotify": 0,
    "youtube": 0,
    "soundcloud": 1,
    "beatport": 2,
    "traxsource": 3,
    "apple": 4,
}


def get_priority_groups() -> dict[int, list[str]]:
    groups: dict[int, list[str]] = {}
    for slug, prio in DOWNLOAD_PRIORITY.items():
        source_id = None
        for key, name in SOURCE_TYPE.items():
            if name == slug:
                source_id = key
                break
        if not source_id:
            continue
        groups.setdefault(prio, []).append(source_id)
    return groups


def parse_media_icon_params(track_div: Tag) -> dict[str, dict[str, str]]:
    media_icons = track_div.select(".mediaRow i.mAction")
    keys = ["idObject", "idItem", "idSource", "viewSource", "viewItem"]
    params_by_source: dict[str, dict[str, str]] = {}

    for icon in media_icons:
        onclick = icon.get("onclick", "")
        params: dict[str, str] = {}
        for key in keys:
            match = re.search(fr"{key}:\s*['\"]?([^'\",\s}}]+)['\"]?", onclick)
            if match:
                params[key] = match.group(1)
        source_id = params.get("idSource")
        if source_id:
            params["insertAfter"] = "true"
            params_by_source[source_id] = params

    return params_by_source


def request_ajax_media_link(params: dict[str, str], page: Any) -> Result[dict[str, str], str]:
    ajax_url = "https://www.1001tracklists.com/ajax/get_medialink.php"
    try:
        response = page.request.get(
            ajax_url,
            params=params,
            headers={
                "X-Requested-With": "XMLHttpRequest",
                "Referer": page.url,
                "Accept": "application/json, text/javascript, */*; q=0.01",
            },
        )

        if not response.ok:
            return Result.fail(f"HTTP {response.status} {response.status_text}")

        result_json = response.json()
        items = result_json.get("data", [])
        links: dict[str, str] = {}

        for item in items:
            player_id = item.get("playerId")
            src = str(item.get("source"))
            platform = SOURCE_TYPE.get(src, src)
            if player_id:
                links[platform] = str(player_id)

        return Result.success(links)

    except Exception as e:
        return Result.fail(str(e))


def get_full_title(track_div: Tag) -> str:
    track_val = track_div.select_one(".trackValue")
    return track_val.text.strip().replace('\xa0', ' ') if track_val else "Unknown"


def extract_media_links(track_div: Tag, page: Any, set_id: str) -> tuple[dict[str, str], list[ScrapeFailure]]:
    log = logging.getLogger("MediaLinks")
    source_params = parse_media_icon_params(track_div)
    priority_groups = get_priority_groups()
    track_title = get_full_title(track_div)
    failures: list[ScrapeFailure] = []

    for prio in sorted(priority_groups.keys()):
        group_results: dict[str, str] = {}
        any_success = False

        for source_id in priority_groups[prio]:
            params = source_params.get(source_id)
            if not params:
                continue

            res = request_ajax_media_link(params, page)
            if res.is_success:
                any_success = True
                group_results.update(res.value)
            else:
                # Log failed AJAX requests
                log.warning(
                    "AJAX media fetch failed for track=%s: %s params=%s",
                    track_title,
                    res.error,
                    params,
                )
                failures.append(ScrapeFailure(
                    set_id=set_id,
                    stage="ajax",
                    track_title=track_title,
                    track_id=track_div.get("data-trackid"),
                    tlp_id=track_div.get("data-id"),
                    params_json=json.dumps(params, ensure_ascii=False),
                    error=res.error,
                ))
                continue

        if any_success:
            return group_results, failures

    return {}, failures


def infer_platform_from_url(url: str) -> str:
    lowered = url.lower()
    if "youtube.com" in lowered or "youtu.be" in lowered:
        return "youtube"
    if "soundcloud.com" in lowered:
        return "soundcloud"
    if "open.spotify.com" in lowered:
        return "spotify"
    if "music.apple.com" in lowered:
        return "apple"
    if "beatport.com" in lowered:
        return "beatport"
    if "traxsource.com" in lowered:
        return "traxsource"
    return "other"


def scrape_dj_set(soup: BeautifulSoup, set_id: str, page: Any) -> Result[ScrapedSetData, str]:
    set_media_links = extract_set_media_links(soup, set_id=set_id)
    track_media_links: list[DJSetTrackMediaLink] = []
    failures: list[ScrapeFailure] = []
    rows: list[DJSetRow] = []

    tl_container = soup.find("div", id="tlTab")
    if tl_container:
        for idx, element in enumerate(tl_container.find_all("div", recursive=False)):
            data_attrs = {k: v for k, v in element.attrs.items() if k.startswith("data-")}
            rows.append(DJSetRow(
                set_id=set_id,
                row_index=idx,
                element_id=element.get("id"),
                classes=" ".join(element.get("class", [])) if element.get("class") else None,
                data_attrs_json=json.dumps(data_attrs, ensure_ascii=False) if data_attrs else None,
                text_excerpt=element.get_text(" ", strip=True)[:500],
                raw_html=str(element),
            ))

            classes = element.get("class", [])
            if "tlpItem" not in classes:
                continue

            tlp_id = element.get("data-id")
            track_id = element.get("data-trackid")

            links, link_failures = extract_media_links(element, page, set_id=set_id)
            failures.extend(link_failures)

            params_by_source = parse_media_icon_params(element)
            for platform, player_id in links.items():
                source_id = None
                for key, name in SOURCE_TYPE.items():
                    if name == platform:
                        source_id = key
                        break
                params = params_by_source.get(source_id or "", {})
                track_media_links.append(DJSetTrackMediaLink(
                    set_id=set_id,
                    tlp_id=tlp_id,
                    track_id=track_id,
                    platform=platform,
                    player_id=player_id,
                    id_object=params.get("idObject"),
                    id_item=params.get("idItem"),
                    id_source=params.get("idSource"),
                    view_source=params.get("viewSource"),
                    view_item=params.get("viewItem"),
                ))

    return Result.success(ScrapedSetData(
        set_media_links=set_media_links,
        track_media_links=track_media_links,
        failures=failures,
        rows=rows,
    ))
