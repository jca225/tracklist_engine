"""Shared acquisition guards — the offline-testable verdict core.

Every guard here encodes a failure class the corpus has already paid for
(1,267 rows in `track_audio_correction`; the 2026-07-09 D2 retry campaign;
the BB10/BB11 labeling pulls):

- **HTML-interstitial payloads** saved as audio: Dropbox served login pages
  with HTTP 200 and 52 of them were cached as "downloaded" stems (D2).
- **Preview/teaser clips** installed as full assets: a 46 s preview clip
  replaced a full track and the rescue then fetched the wrong version
  (`project_wrong_version_preview_clip`, 106 suspects).
- **Stream rips wildly longer than the listed set** — the set-level analog
  (a multi-hour livestream VOD grabbed for a 1 h set).

Pure functions over recorded inputs — no network, no DB — so entrypoints
(`ingest.main` track + set-mix paths, the YT Music rescue, discord harvest,
`replace_track_audio`) can compose them and `tests/test_ingest_guards.py`
can replay real cases in CI. New acquisition code should call these instead
of re-growing inline checks.
"""

from __future__ import annotations

import re

# ── listed play time ─────────────────────────────────────────────────────────

_PLAY_TIME_RE = re.compile(r"^\s*(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?\s*$", re.IGNORECASE)


def parse_play_time(raw: str | None) -> int | None:
    """`dj_sets.play_time` ('1h', '59m', '1h 2m', '2h') → seconds, else None."""
    if not raw:
        return None
    m = _PLAY_TIME_RE.match(raw)
    if not m or (m.group(1) is None and m.group(2) is None):
        return None
    return int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60


def duration_sane(
    actual_s: float | None,
    listed_s: int | None,
    *,
    min_frac: float = 0.5,
    max_frac: float = 3.0,
) -> bool:
    """Is a downloaded asset's duration plausible against the listed length?

    Conservative by design: with no listed length (or no measured duration)
    there is nothing to verify, so pass — the guard only rejects what it can
    positively contradict. Rejects the preview-clip class (actual ≪ listed)
    and the stream-rip class (actual ≫ listed).
    """
    if actual_s is None or listed_s is None or listed_s <= 0:
        return True
    return min_frac * listed_s <= actual_s <= max_frac * listed_s


# ── payload sniff ────────────────────────────────────────────────────────────

# magic prefixes of every container the pipeline downloads (yt-dlp/spotdl
# output: m4a/mp4, opus/ogg, mp3, flac, wav, webm)
_AUDIO_MAGIC_PREFIXES: tuple[bytes, ...] = (
    b"ID3",  # mp3 with ID3 tag
    b"fLaC",  # flac
    b"OggS",  # ogg/opus
    b"RIFF",  # wav
    b"\x1aE\xdf\xa3",  # matroska/webm
    b"\xff\xfb",  # bare mp3 frame (MPEG1 L3)
    b"\xff\xf3",  # mp3 frame (MPEG2 L3)
    b"\xff\xf2",  # mp3 frame
    b"\xff\xf1",  # aac adts
)

_HTML_MARKERS: tuple[bytes, ...] = (b"<!doctype", b"<html", b"<head", b"<?xml")


def sniff_audio_payload(head: bytes, content_type: str | None = None) -> bool:
    """Does this look like audio (vs an HTML interstitial served as 200)?

    `head` = the first bytes of the payload (>= 12 recommended). The D2 class:
    Dropbox login pages with `content-type: text/html` and HTTP 200 were saved
    as stems. Reject declared-HTML content types and HTML/XML-marker heads;
    accept known audio/video container magics (mp4/m4a boxes have their magic
    at offset 4: `ftyp`).
    """
    if content_type and any(
        t in content_type.lower()
        for t in ("text/html", "text/plain", "application/xhtml")
    ):
        return False
    lowered = head[:64].lstrip().lower()
    if any(lowered.startswith(m) for m in _HTML_MARKERS):
        return False
    if any(head.startswith(m) for m in _AUDIO_MAGIC_PREFIXES):
        return True
    if len(head) >= 12 and head[4:8] == b"ftyp":  # mp4/m4a/mov box
        return True
    # unknown container: do not hard-reject on magic alone — only declared/
    # sniffed HTML is a positive contradiction
    return True
