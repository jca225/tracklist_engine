"""Spotify metadata adapter — pulls track metadata only, not audio.

Audio-features and audio-analysis endpoints are deprecated for new apps as of
Nov 27 2024 — do NOT call them from here. We only pull fields still available:
title, artists, album, release_date, duration_ms, ISRC (via external_ids),
popularity, explicit, track_number. Uses Client Credentials flow (server-side,
no user auth).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from ..errors import SpotifyApiError
from ..result import Err, Ok, Result


@dataclass(frozen=True)
class SpotifyTrackMeta:
    spotify_id: str
    title: str
    artists: tuple[str, ...]
    album: str
    release_date: str
    duration_ms: int
    isrc: str | None
    popularity: int | None
    explicit: bool


def _client():
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    cid = os.getenv("SPOTIFY_CLIENT_ID")
    csec = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not cid or not csec:
        raise RuntimeError("SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET not set in env")
    return spotipy.Spotify(auth_manager=SpotifyClientCredentials(cid, csec))


def fetch_track_meta(spotify_id: str) -> Result[SpotifyTrackMeta, SpotifyApiError]:
    try:
        sp = _client()
        t = sp.track(spotify_id)
    except RuntimeError as e:
        return Err(SpotifyApiError(kind="auth", detail=str(e)))
    except Exception as e:  # spotipy raises SpotifyException + http errors; normalize broadly
        msg = str(e)
        if "401" in msg or "403" in msg:
            return Err(SpotifyApiError(kind="auth", detail=msg))
        if "429" in msg or "rate" in msg.lower():
            return Err(SpotifyApiError(kind="rate_limit", detail=msg))
        if "404" in msg:
            return Err(SpotifyApiError(kind="not_found", detail=msg))
        return Err(SpotifyApiError(kind="network", detail=msg))

    return Ok(SpotifyTrackMeta(
        spotify_id=spotify_id,
        title=t["name"],
        artists=tuple(a["name"] for a in t.get("artists", [])),
        album=t.get("album", {}).get("name", ""),
        release_date=t.get("album", {}).get("release_date", ""),
        duration_ms=int(t.get("duration_ms") or 0),
        isrc=(t.get("external_ids") or {}).get("isrc"),
        popularity=t.get("popularity"),
        explicit=bool(t.get("explicit", False)),
    ))
