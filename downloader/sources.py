from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup

from downloader.constants import USER_AGENT


def canonicalize_url(url: str, *, timeout: float = 10.0) -> str:
    """Return a deterministic canonical URL for a track."""
    if "soundcloud" in url:
        return resolve_soundcloud_url(url, timeout=timeout)
    if "spotify" in url:
        return url.split("?")[0].replace("embed/", "")
    return url


def resolve_soundcloud_url(url: str, *, timeout: float = 10.0) -> str:
    """Get the canonical SoundCloud link associated with the provided URL."""
    html = ""
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
        )
        html = response.text
    except Exception:
        return url

    soup = BeautifulSoup(html, "html.parser")
    links = soup.find_all("link")
    for link in links:
        href = link.get("href")
        if isinstance(href, str) and href.startswith("https://soundcloud.com/"):
            return href
    return url


def infer_provider(url: str) -> str:
    lowered = url.lower()
    if "spotify" in lowered:
        return "spotify"
    if "youtube.com" in lowered or "youtu.be" in lowered:
        return "youtube"
    if "soundcloud" in lowered:
        return "soundcloud"
    return "unknown"


def build_track_media_url(platform: str, player_id: str) -> str:
    value = (player_id or "").strip()
    if not value:
        return ""
    if value.startswith("http://") or value.startswith("https://"):
        return value
    if platform == "spotify":
        return f"https://open.spotify.com/track/{value}"
    if platform == "youtube":
        return f"https://www.youtube.com/watch?v={value}"
    if platform == "soundcloud":
        return f"https://w.soundcloud.com/player/?url=https://api.soundcloud.com/tracks/{value}"
    return ""


def iter_track_media_urls(conn: sqlite3.Connection) -> Iterable[dict[str, str]]:
    query = """
        SELECT set_id, track_id, platform, player_id
        FROM dj_set_track_media_links
        WHERE platform IN ('spotify', 'youtube', 'soundcloud')
          AND player_id IS NOT NULL
          AND TRIM(player_id) != ''
    """
    for row in conn.execute(query):
        raw_url = build_track_media_url(row["platform"], row["player_id"])
        if not raw_url:
            continue
        yield {
            "source": "dj_set_track_media_links",
            "set_id": str(row["set_id"] or ""),
            "track_id": str(row["track_id"] or ""),
            "platform": str(row["platform"] or ""),
            "url": raw_url,
        }


def iter_set_media_urls(conn: sqlite3.Connection) -> Iterable[dict[str, str]]:
    query = """
        SELECT set_id, platform, url
        FROM dj_set_media_links
        WHERE platform IN ('spotify', 'youtube', 'soundcloud')
          AND url IS NOT NULL
          AND TRIM(url) != ''
    """
    for row in conn.execute(query):
        yield {
            "source": "dj_set_media_links",
            "set_id": str(row["set_id"] or ""),
            "track_id": "",
            "platform": str(row["platform"] or ""),
            "url": str(row["url"] or ""),
        }


def get_music_url_rows(db_path: Path) -> list[dict[str, str]]:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = list(iter_track_media_urls(conn))
        rows.extend(iter_set_media_urls(conn))
        return rows
    finally:
        conn.close()
