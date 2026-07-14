"""Appleseed librarian — fulfills song_requests from Appleseed's SQLite by
fetching audio with the ingest stack into the app's library folder. Runs in
tracklist_engine (has yt-dlp); the product repo stays downloader-free.

    venvs/audio/bin/python -m scripts.appleseed_librarian \\
        --db ~/Desktop/mashup_compiler/server/state.db \\
        --library ~/Desktop/mashup_compiler/server/library [--once]

Live-run requirements
---------------------
- venvs/audio/bin/python (has yt-dlp, ytmusicapi, ingest stack)
- ffmpeg on PATH (for 44.1k transcode)
- Optional: cookies.txt for age-gated YouTube (see feedback_ytdlp_bot_detection_recipe)
- Optional: node/nodejs on PATH for yt-dlp n-challenge deobfuscation (required
  for ~all current YouTube URLs — without it yt-dlp returns only image formats)
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MIN_FULL_S = 60  # shorter than this = a preview, not the song


def _safe_stem(q: str) -> str:
    """Sanitize a query string into a safe filename stem (no slashes, bangs, etc.)."""
    return re.sub(r"[^\w\s\-\(\)&,']", "_", q).strip().rstrip("_").strip()[:80]


def _best_hit(hits):
    """Pick the shortest full-length studio hit from a list of search results.

    Filters out anything under MIN_FULL_S seconds (previews) and anything
    whose title contains 'live' or 'preview'. Returns None if no qualifying
    hit exists. Works with any object that has .duration_s and .title attrs
    (both YTMSearchHit and test fakes).
    """
    ok = [
        h
        for h in hits
        if getattr(h, "duration_s", 0)
        and h.duration_s >= MIN_FULL_S
        and "live" not in h.title.lower()
        and "preview" not in h.title.lower()
    ]
    if not ok:
        return None
    return min(
        ok, key=lambda h: h.duration_s
    )  # shortest full-length = likely studio cut


def _claim_one(db_path: Path):
    """Atomically claim one 'searching' row, flip it to 'fetching', return it.

    Uses BEGIN IMMEDIATE so two concurrent pollers cannot claim the same row.
    Returns None if no unclaimed row exists.
    """
    con = sqlite3.connect(db_path, timeout=10)
    con.row_factory = sqlite3.Row
    try:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute(
            "SELECT * FROM song_requests WHERE status='searching' ORDER BY id LIMIT 1"
        ).fetchone()
        if row is None:
            con.rollback()
            return None
        con.execute(
            "UPDATE song_requests SET status='fetching' WHERE id=?", (row["id"],)
        )
        con.commit()
        return row
    finally:
        con.close()


def _mark(db_path: Path, rid: int, status: str, error: str | None = None) -> None:
    con = sqlite3.connect(db_path, timeout=10)
    con.execute(
        "UPDATE song_requests SET status=?, error=? WHERE id=?", (status, error, rid)
    )
    con.commit()
    con.close()


def _transcode_to_wav(src: Path, dst: Path) -> None:
    """ffmpeg-transcode src to dst at 44.1 kHz WAV (same path as server upload)."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ar",
        "44100",
        "-ac",
        "2",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg transcode failed (exit {proc.returncode}): {proc.stderr[:300]}"
        )


def _fetch_to_library(row, library: Path) -> Path:
    """Resolve the request to a downloaded source file, transcode to 44.1k wav
    in `library`, return the wav path. Raises on failure.

    kind='url'  → download_one(url) via downloader adapter (YouTube/SoundCloud)
    kind='name' → ytmusic_adapter.search(query) → _best_hit → _ytdlp_download

    Real signatures (from discovery):
      search(query: str, limit: int = 5) -> Result[tuple[YTMSearchHit, ...], DownloadError]
        YTMSearchHit fields: .video_id (str), .title (str), .artists (tuple[str,...]),
                             .duration_s (int | None)
        NOTE: no .url field — YouTube URL is built as
              https://www.youtube.com/watch?v={hit.video_id}

      downloader.download_one(track_id: str, source: MediaSource, cfg: DownloadConfig)
        -> Result[AudioAsset, DownloadError]
        MediaSource(platform, player_id, url) — platform must be 'youtube'|'soundcloud'

      ytmusic_adapter._ytdlp_download(video_id, track_id, cfg, timeout_s, cookies_path)
        -> Result[Path, DownloadError]
        (lower-level; used directly here to avoid the duplicate search in
        ytmusic_adapter.download_one which re-searches instead of taking a hit)
    """
    from ingest.adapters import downloader as dl_mod
    from ingest.adapters import ytmusic_adapter as yt_mod
    from ingest.adapters.downloader import DownloadConfig
    from core.models import MediaSource
    from core.result import Ok, Err

    library.mkdir(parents=True, exist_ok=True)
    query: str = row["query"]
    kind: str = row["kind"]
    safe = _safe_stem(query)
    wav_path = library / f"{safe}.wav"

    with tempfile.TemporaryDirectory(prefix="appleseed_dl_") as tmp_str:
        tmp = Path(tmp_str)
        cfg = DownloadConfig(out_dir=tmp, audio_format="m4a", retries=2)

        if kind == "url":
            # Parse YouTube video ID or SoundCloud track ID from the URL.
            url: str = query
            yt_match = re.search(
                r"(?:youtube\.com/watch\?v=|youtu\.be/)([A-Za-z0-9_-]{11})", url
            )
            sc_match = re.search(
                r"soundcloud\.com/|api\.soundcloud\.com/tracks/(\d+)", url
            )
            if yt_match:
                video_id = yt_match.group(1)
                source = MediaSource(platform="youtube", player_id=video_id, url=url)
            elif sc_match:
                player_id = sc_match.group(1) or url.split("/")[-1]
                source = MediaSource(
                    platform="soundcloud", player_id=player_id, url=url
                )
            else:
                # Attempt direct yt-dlp anyway (e.g. youtu.be short links, SC full URLs)
                source = MediaSource(platform="youtube", player_id="direct", url=url)

            result = dl_mod.download_one(f"appleseed_{row['id']}", source, cfg)
            match result:
                case Err(err):
                    raise RuntimeError(f"download failed: {err.detail}")
                case Ok(asset):
                    src_path = Path(asset.path)

        elif kind == "name":
            search_result = yt_mod.search(query, limit=8)
            match search_result:
                case Err(err):
                    raise RuntimeError(f"YT Music search failed: {err.detail}")
                case Ok(hits):
                    pass

            hit = _best_hit(hits)
            if hit is None:
                raise RuntimeError(
                    f"no full-length match found for {query!r} "
                    f"(all hits were previews, live recordings, or too short)"
                )

            dl_result = yt_mod._ytdlp_download(
                video_id=hit.video_id,
                track_id=f"appleseed_{row['id']}",
                cfg=cfg,
                timeout_s=180.0,
            )
            match dl_result:
                case Err(err):
                    raise RuntimeError(f"yt-dlp download failed: {err.detail}")
                case Ok(src_path):
                    pass

        else:
            raise RuntimeError(f"unknown request kind {kind!r}")

        _transcode_to_wav(src_path, wav_path)

    return wav_path


def run(db_path: Path, library: Path, once: bool = False) -> int:
    while True:
        row = _claim_one(db_path)
        if row is None:
            if once:
                return 0
            time.sleep(5)
            continue
        try:
            _fetch_to_library(row, library)
            _mark(db_path, row["id"], "done")
            print(f"fulfilled #{row['id']}: {row['query']}")
        except Exception as e:  # noqa: BLE001 — surface any fetch failure to the UI
            _mark(db_path, row["id"], "failed", str(e)[:300])
            print(f"failed #{row['id']}: {e}", file=sys.stderr)
        if once:
            return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--once", action="store_true")
    a = p.parse_args(argv)
    return run(a.db, a.library, once=a.once)


if __name__ == "__main__":
    sys.exit(main())
