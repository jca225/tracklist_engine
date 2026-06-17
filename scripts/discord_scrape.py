#!/usr/bin/env python3
"""Scrape Discord channels for audio assets (instrumentals / acappellas / stem packs).

Self-contained Python equivalent of DiscordChatExporter: it talks to the Discord
REST API with *your* user token, walks every message in the configured channels,
then pulls down two kinds of asset:

  1. Discord-hosted attachments (direct CDN uploads), filtered to audio + archives.
  2. External download links (Google Drive, Dropbox, MediaFire, MEGA, WeTransfer,
     Hypeddit, …). Auto-resolved where we can; catalogued for manual fetch where
     we can't (JS-gated / ephemeral hosts).

Everything is recorded in a manifest (JSON + CSV) so a re-run never re-downloads
and you can see exactly what was found vs. what needs a human.

  ⚠️  Automating Discord with a *user* token violates Discord's ToS and carries a
      (small but real) account-ban risk. You chose this path knowingly. Be gentle:
      this script rate-limits itself and respects 429 Retry-After.

Usage
-----
    export DISCORD_USER_TOKEN='...'        # your user token (NOT prefixed "Bot ")
    venvs/audio/bin/python scripts/discord_scrape.py            # full run
    venvs/audio/bin/python scripts/discord_scrape.py --catalog-only   # no downloads
    venvs/audio/bin/python scripts/discord_scrape.py --channels 882743836097511424

Getting your user token: open Discord in a browser → DevTools → Network tab →
filter "messages" → click any request → Headers → copy the `authorization` value.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass, fields, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Optional

import requests

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

API = "https://discord.com/api/v10"

# channel id -> human label (becomes the output subfolder)
CHANNELS: dict[str, str] = {
    "882743836097511424": "instrumentals",
    "882744265216761857": "acappellas",
    "847115682390736957": "stem_packs",
}

OUT_ROOT = Path(os.environ.get("DISCORD_OUT", Path.home() / "discord_stems"))

# what counts as a downloadable audio / pack attachment
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".m4a", ".ogg", ".wma", ".alac"}
PACK_EXTS = {".zip", ".rar", ".7z", ".tar", ".gz"}
KEEP_EXTS = AUDIO_EXTS | PACK_EXTS

# catch-all: every URL in a message; classification happens in _host_of().
# Trailing junk (quotes, parens, Discord's zero-width chars) is stripped after.
ALL_URL_RE = re.compile(r"https?://[^\s<>\)\]\"']+", re.IGNORECASE)

REQUEST_PAUSE = 0.6  # seconds between message-page fetches (be polite)
MAX_RETRIES = 5

DISCORD_EPOCH = 1420070400000  # 2015-01-01, ms — base for snowflake timestamps


# --------------------------------------------------------------------------- #
# Snowflake / --since cutoff helpers
# --------------------------------------------------------------------------- #


def date_to_snowflake(date_str: str) -> int:
    """`YYYY-MM-DD` (UTC midnight) -> the minimum Discord snowflake for that instant."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    ms = int(dt.timestamp() * 1000)
    return (ms - DISCORD_EPOCH) << 22


def last_seen_id(assets: dict[str, "Asset"], channel_id: str) -> Optional[int]:
    """Newest message id already recorded for a channel (for `--since auto`)."""
    ids = [int(a.message_id) for a in assets.values() if a.channel_id == channel_id]
    return max(ids) if ids else None


def resolve_cutoff(
    since: Optional[str], assets: dict[str, "Asset"], channel_id: str
) -> Optional[int]:
    """Turn the --since value into a stop-after snowflake for one channel."""
    if not since:
        return None
    if since == "auto":
        return last_seen_id(assets, channel_id)
    return date_to_snowflake(since)


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Asset:
    channel_id: str
    channel_label: str
    message_id: str
    author: str
    timestamp: str
    kind: str  # "attachment" | "link"
    host: str  # "discord" | "drive" | "dropbox" | "mediafire" | ...
    source_url: str
    filename: str  # best-effort name (may be empty for opaque links)
    size: Optional[int] = None  # bytes, attachments only
    status: str = "pending"  # pending | downloaded | skipped | manual | error
    local_path: str = ""
    note: str = ""


# --------------------------------------------------------------------------- #
# Discord REST: message pagination
# --------------------------------------------------------------------------- #


def _session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "authorization": token,
            "user-agent": "Mozilla/5.0",  # user-token requests want a browser-ish UA
        }
    )
    return s


def _get(session: requests.Session, url: str, **kw) -> requests.Response:
    """GET with Discord rate-limit (429) handling."""
    for attempt in range(MAX_RETRIES):
        r = session.get(url, **kw)
        if r.status_code == 429:
            retry = float(r.headers.get("retry-after", r.json().get("retry_after", 1)))
            print(f"    rate-limited, sleeping {retry:.1f}s", file=sys.stderr)
            time.sleep(retry + 0.25)
            continue
        return r
    r.raise_for_status()
    return r


def fetch_messages(
    session: requests.Session,
    channel_id: str,
    stop_after_id: Optional[int] = None,
) -> Iterator[dict]:
    """Yield messages in a channel, newest→oldest via the `before` cursor.

    If `stop_after_id` is given (a snowflake), pagination halts as soon as a
    message at or below that id is reached — i.e. only messages *newer* than the
    cutoff are yielded. This is how `--since` keeps re-runs to just new posts.
    """
    before: Optional[str] = None
    seen = 0
    while True:
        url = f"{API}/channels/{channel_id}/messages?limit=100"
        if before:
            url += f"&before={before}"
        r = _get(session, url)
        if r.status_code == 403:
            print(
                f"  ! 403 forbidden on {channel_id} — token lacks access",
                file=sys.stderr,
            )
            return
        if r.status_code == 401:
            print("  ! 401 unauthorized — token is invalid/expired", file=sys.stderr)
            sys.exit(1)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            return
        for m in batch:
            if stop_after_id is not None and int(m["id"]) <= stop_after_id:
                print(f"    …{seen} messages (reached cutoff)", file=sys.stderr)
                return
            yield m
            seen += 1
        before = batch[-1]["id"]
        print(f"    …{seen} messages", end="\r", file=sys.stderr)
        time.sleep(REQUEST_PAUSE)


# --------------------------------------------------------------------------- #
# Asset extraction
# --------------------------------------------------------------------------- #


def _clean_url(url: str) -> str:
    """Trim trailing punctuation Discord/markdown leaves stuck to a pasted URL."""
    return url.rstrip(".,;:!?'\"`)>]}​⁠ ")


def _host_of(url: str) -> str:
    u = url.lower()
    if "cdn.discordapp.com" in u or "media.discordapp.net" in u:
        return "discord"  # inline attachment, right-click→Copy Link
    if "drive.google" in u:
        return "drive"
    if "dropbox.com" in u:
        return "dropbox"
    if "mediafire.com" in u:
        return "mediafire"
    if "mega.nz" in u:
        return "mega"
    if "we.tl" in u or "wetransfer" in u:
        return "wetransfer"
    if "hypeddit" in u:
        return "hypeddit"
    if "gofile.io" in u:
        return "gofile"
    if "pixeldrain" in u:
        return "pixeldrain"
    if "soundcloud.com" in u or "on.soundcloud" in u:
        return "soundcloud"
    if "youtube.com" in u or "youtu.be" in u:
        return "youtube"
    if "buzzheavier" in u:
        return "buzzheavier"
    if "sharing.wtf" in u or "filesharing.io" in u:
        return "sharingwtf"
    return "other"


def extract_assets(msg: dict, channel_id: str, label: str) -> list[Asset]:
    out: list[Asset] = []
    author = msg.get("author", {}).get("username", "?")
    ts = msg.get("timestamp", "")
    mid = msg["id"]

    for att in msg.get("attachments", []):
        name = att.get("filename", "")
        ext = Path(name).suffix.lower()
        if ext not in KEEP_EXTS:
            continue
        out.append(
            Asset(
                channel_id=channel_id,
                channel_label=label,
                message_id=mid,
                author=author,
                timestamp=ts,
                kind="attachment",
                host="discord",
                source_url=att["url"],
                filename=name,
                size=att.get("size"),
            )
        )

    for raw in ALL_URL_RE.findall(msg.get("content", "") or ""):
        url = _clean_url(raw)
        out.append(
            Asset(
                channel_id=channel_id,
                channel_label=label,
                message_id=mid,
                author=author,
                timestamp=ts,
                kind="link",
                host=_host_of(url),
                source_url=url,
                filename="",
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Paste parsing (no token): a copy-pasted channel dump -> Assets
# --------------------------------------------------------------------------- #

# Discord renders a message header as "Display Name — M/D/YY, H:MM AM".
AUTHOR_RE = re.compile(r"^(.+?)\s+—\s+\d{1,2}/\d{1,2}/\d{2,4},\s+\d")


def parse_paste(text: str, label: str) -> list[Asset]:
    """Extract every URL from a pasted channel dump, attributing author + a
    best-effort filename hint (the embed-title line that often follows a link).

    No Discord API / token involved — this reads text you copied yourself.
    Synthetic message ids are just a running counter (ordering, dedup).
    """
    out: list[Asset] = []
    author = "?"
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = AUTHOR_RE.match(line.strip())
        if m:
            author = m.group(1).strip()
        for raw in ALL_URL_RE.findall(line):
            url = _clean_url(raw)
            host = _host_of(url)
            # Discord CDN anchors include posted screenshots — keep only audio/archive
            if host == "discord":
                ext = Path(url.split("?")[0]).suffix.lower()
                if ext and ext not in KEEP_EXTS:
                    continue
            # filename hint: the first nearby line that looks like an audio file
            hint = ""
            for look in lines[i : i + 4]:
                t = look.strip()
                if any(t.lower().endswith(e) for e in AUDIO_EXTS | PACK_EXTS):
                    hint = t
                    break
            out.append(
                Asset(
                    channel_id=label,
                    channel_label=label,
                    message_id=str(len(out)),
                    author=author,
                    timestamp="",
                    kind="link",
                    host=host,
                    source_url=url,
                    filename=hint,
                )
            )
    return out


# --------------------------------------------------------------------------- #
# Downloaders (one per host). Return (status, local_path, note).
# --------------------------------------------------------------------------- #


def _safe_name(name: str, fallback: str) -> str:
    name = re.sub(r"[^\w.\-]+", "_", name).strip("_") or fallback
    return name[:180]


def _stream_to(session: requests.Session, url: str, dest: Path, **kw) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, stream=True, timeout=60, **kw) as r:
        r.raise_for_status()
        total = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                total += len(chunk)
    return total


def dl_discord(
    session: requests.Session, a: Asset, outdir: Path
) -> tuple[str, str, str]:
    url_name = Path(a.source_url.split("?")[0]).name  # cdn links carry the filename
    name = _safe_name(a.filename or url_name, a.message_id + ".bin")
    dest = outdir / name
    if dest.exists() and dest.stat().st_size == (a.size or -1):
        return "downloaded", str(dest), "already present"
    try:
        n = _stream_to(session, a.source_url, dest)
        return "downloaded", str(dest), f"{n} bytes"
    except Exception as e:  # signed CDN URLs expire; surface clearly
        return "error", "", f"{type(e).__name__}: {e}"


def dl_dropbox(
    session: requests.Session, a: Asset, outdir: Path
) -> tuple[str, str, str]:
    url = a.source_url.replace("?dl=0", "?dl=1")
    if "dl=" not in url:
        url += ("&" if "?" in url else "?") + "dl=1"
    name = _safe_name(Path(url.split("?")[0]).name, a.message_id + ".bin")
    try:
        n = _stream_to(session, url, outdir / name)
        return "downloaded", str(outdir / name), f"{n} bytes"
    except Exception as e:
        return "error", "", f"{type(e).__name__}: {e}"


def _drive_id(url: str) -> Optional[str]:
    """Pull the file/folder id out of any Drive URL shape (/file/d/ID, ?id=ID, …)."""
    for pat in (r"/file/d/([\w-]+)", r"/folders/([\w-]+)", r"[?&]id=([\w-]+)"):
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


def _drive_file(
    session: requests.Session, fid: str, outdir: Path
) -> tuple[str, str, str]:
    """Direct Drive file download via the uc endpoint + large-file confirm form.
    More robust than gdown on these links (gdown chokes on the virus-scan page)."""
    base = "https://drive.google.com/uc"
    r = session.get(
        base, params={"id": fid, "export": "download"}, stream=True, timeout=60
    )
    ctype = r.headers.get("content-type", "")
    if "text/html" in ctype:  # large-file "can't scan for viruses" interstitial
        body = r.text
        action = re.search(r'action="([^"]+)"', body)
        fields = dict(re.findall(r'name="([^"]+)" value="([^"]*)"', body))
        if not (action and fields):
            return "error", "", "drive link dead/private (no download form)"
        r = session.get(
            action.group(1).replace("&amp;", "&"),
            params=fields,
            stream=True,
            timeout=60,
        )
        if "text/html" in r.headers.get("content-type", ""):
            return "error", "", "drive returned HTML (quota/permission)"
    if r.status_code == 404:
        return "error", "", "drive 404 (file removed)"
    r.raise_for_status()
    cd = r.headers.get("content-disposition", "")
    m = re.search(r'filename="([^"]+)"', cd)
    name = _safe_name(m.group(1) if m else "", fid + ".bin")
    dest = outdir / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(dest, "wb") as f:
        for chunk in r.iter_content(1 << 16):
            f.write(chunk)
            total += len(chunk)
    return "downloaded", str(dest), f"{total} bytes"


def dl_drive(session: requests.Session, a: Asset, outdir: Path) -> tuple[str, str, str]:
    """Drive: requests for files (robust), gdown for whole folders."""
    fid = _drive_id(a.source_url)
    if not fid:
        return "manual", "", "could not parse a Drive id from the url"
    if "/folders/" in a.source_url:
        try:
            import gdown  # type: ignore
        except ImportError:
            return "manual", "", "pip install gdown to fetch Drive folders"
        try:
            outdir.mkdir(parents=True, exist_ok=True)
            paths = gdown.download_folder(
                id=fid, output=str(outdir), quiet=True, use_cookies=False
            )
            return (
                "downloaded" if paths else "error",
                str(outdir),
                f"{len(paths or [])} files",
            )
        except Exception as e:
            return "error", "", f"{type(e).__name__}: {e}"
    try:
        return _drive_file(session, fid, outdir)
    except Exception as e:
        return "error", "", f"{type(e).__name__}: {e}"


def dl_mediafire(
    session: requests.Session, a: Asset, outdir: Path
) -> tuple[str, str, str]:
    """MediaFire: scrape the direct-download href off the file page."""
    try:
        page = session.get(a.source_url, timeout=30).text
        m = re.search(r'href="(https://download[^"]+)"', page)
        if not m:
            return "manual", "", "no direct link found on page (may be a folder)"
        direct = m.group(1)
        name = _safe_name(Path(direct.split("?")[0]).name, a.message_id + ".bin")
        n = _stream_to(session, direct, outdir / name)
        return "downloaded", str(outdir / name), f"{n} bytes"
    except Exception as e:
        return "error", "", f"{type(e).__name__}: {e}"


def dl_ytdlp(session: requests.Session, a: Asset, outdir: Path) -> tuple[str, str, str]:
    """SoundCloud / YouTube: hand off to yt-dlp (handles private `s-…` share links)."""
    import shutil
    import subprocess

    ytdlp = shutil.which("yt-dlp") or str(Path(sys.executable).parent / "yt-dlp")
    if not Path(ytdlp).exists() and not shutil.which("yt-dlp"):
        return "manual", "", "yt-dlp not found on PATH / in venv"
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(
            [
                ytdlp,
                "-x",
                "--audio-format",
                "wav",
                "--no-playlist",
                "-o",
                str(outdir / "%(title)s.%(ext)s"),
                a.source_url,
            ],
            capture_output=True,
            text=True,
            timeout=900,
        )
        if r.returncode != 0:
            return "error", "", (r.stderr.strip().splitlines() or ["yt-dlp failed"])[-1]
        return "downloaded", str(outdir), "via yt-dlp"
    except Exception as e:
        return "error", "", f"{type(e).__name__}: {e}"


def dl_pixeldrain(
    session: requests.Session, a: Asset, outdir: Path
) -> tuple[str, str, str]:
    """pixeldrain: /u/<id> -> https://pixeldrain.com/api/file/<id>?download"""
    m = re.search(r"pixeldrain\.com/u/([A-Za-z0-9]+)", a.source_url)
    if not m:
        return "manual", "", "unrecognised pixeldrain url shape"
    direct = f"https://pixeldrain.com/api/file/{m.group(1)}?download"
    name = _safe_name(a.filename or (a.message_id + ".bin"), a.message_id + ".bin")
    try:
        n = _stream_to(session, direct, outdir / name)
        return "downloaded", str(outdir / name), f"{n} bytes"
    except Exception as e:
        return "error", "", f"{type(e).__name__}: {e}"


# hosts we can't sanely auto-fetch (JS-gated, ephemeral, or login-walled)
MANUAL_HOSTS = {"mega", "wetransfer", "hypeddit", "gofile", "sharingwtf", "other"}

DOWNLOADERS: dict[str, Callable[..., tuple[str, str, str]]] = {
    "discord": dl_discord,
    "dropbox": dl_dropbox,
    "drive": dl_drive,
    "mediafire": dl_mediafire,
    "soundcloud": dl_ytdlp,
    "youtube": dl_ytdlp,
    "pixeldrain": dl_pixeldrain,
}


# --------------------------------------------------------------------------- #
# Manifest persistence
# --------------------------------------------------------------------------- #


def load_manifest(path: Path) -> dict[str, Asset]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())
    return {r["source_url"]: Asset(**r) for r in rows}


def save_manifest(path: Path, assets: dict[str, Asset]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(a) for a in assets.values()]
    path.write_text(json.dumps(rows, indent=2))
    with open(path.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[fld.name for fld in fields(Asset)])
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--channels",
        nargs="*",
        default=list(CHANNELS),
        help="channel ids to scrape (default: all configured)",
    )
    ap.add_argument(
        "--catalog-only",
        action="store_true",
        help="walk messages + build manifest, but download nothing",
    )
    ap.add_argument("--out", type=Path, default=OUT_ROOT, help="output root dir")
    ap.add_argument(
        "--since",
        metavar="YYYY-MM-DD|auto",
        help="only fetch messages newer than this date, or 'auto' to resume from "
        "the newest message already in the manifest (per channel). Shrinks the "
        "per-run token footprint on re-runs.",
    )
    ap.add_argument(
        "--paste",
        type=Path,
        help="parse a copy-pasted channel dump (text file) instead of hitting the "
        "Discord API. No token needed — the safest path. Pair with --label.",
    )
    ap.add_argument(
        "--label",
        default="instrumentals",
        help="channel label / output subfolder for --paste mode "
        "(e.g. instrumentals, acappellas, stem_packs)",
    )
    args = ap.parse_args()

    # Per-channel manifest in paste mode so concurrent --label runs never race
    # on a single shared file. API mode keeps one multi-channel manifest.
    manifest_name = f"manifest_{args.label}.json" if args.paste else "manifest.json"
    manifest_path = args.out / manifest_name
    assets = load_manifest(manifest_path)
    print(f"loaded {len(assets)} known assets from manifest", file=sys.stderr)
    session = _session("")  # downloads to public file hosts need no Discord auth

    # --- Phase 1: accrue assets (paste file OR Discord API) -------------------
    if args.paste:
        text = args.paste.read_text(encoding="utf-8", errors="replace")
        n_new = 0
        for a in parse_paste(text, args.label):
            if a.source_url not in assets:
                assets[a.source_url] = a
                n_new += 1
        print(
            f"  +{n_new} new assets from {args.paste} [{args.label}]", file=sys.stderr
        )
        save_manifest(manifest_path, assets)
    else:
        token = os.environ.get("DISCORD_USER_TOKEN") or os.environ.get("DISCORD_TOKEN")
        if not token:
            sys.exit(
                "set DISCORD_USER_TOKEN, or use --paste FILE for the no-token path"
            )
        session = _session(token)
        for cid in args.channels:
            label = CHANNELS.get(cid, cid)
            cutoff = resolve_cutoff(args.since, assets, cid)
            tag = f" (since {args.since})" if args.since else ""
            print(f"\n== channel {cid} ({label}){tag} ==", file=sys.stderr)
            n_new = 0
            for msg in fetch_messages(session, cid, stop_after_id=cutoff):
                for a in extract_assets(msg, cid, label):
                    if a.source_url not in assets:
                        assets[a.source_url] = a
                        n_new += 1
            print(f"\n  +{n_new} new assets", file=sys.stderr)
            save_manifest(manifest_path, assets)

    # --- Phase 2: download ----------------------------------------------------
    if args.catalog_only:
        _summary(assets)
        print(f"\ncatalog-only: manifest at {manifest_path}", file=sys.stderr)
        return

    # Discord CDN links carry expiring signatures — fetch those first.
    ordered = sorted(assets.items(), key=lambda kv: 0 if kv[1].host == "discord" else 1)
    for url, a in ordered:
        if a.status in ("downloaded", "skipped"):
            continue
        outdir = args.out / a.channel_label / a.host
        if a.host in MANUAL_HOSTS:
            assets[url] = Asset(
                **{**asdict(a), "status": "manual", "note": f"{a.host}: fetch by hand"}
            )
            continue
        dl = DOWNLOADERS.get(a.host)
        if not dl:
            assets[url] = Asset(
                **{**asdict(a), "status": "manual", "note": "no downloader"}
            )
            continue
        print(f"  ↓ [{a.host}] {a.filename or url[:60]}", file=sys.stderr)
        status, path, note = dl(session, a, outdir)
        assets[url] = Asset(
            **{**asdict(a), "status": status, "local_path": path, "note": note}
        )
        save_manifest(manifest_path, assets)

    _summary(assets)
    print(f"\nmanifest: {manifest_path}", file=sys.stderr)


def _summary(assets: dict[str, Asset]) -> None:
    by_status: dict[str, int] = {}
    by_host: dict[str, int] = {}
    for a in assets.values():
        by_status[a.status] = by_status.get(a.status, 0) + 1
        by_host[a.host] = by_host.get(a.host, 0) + 1
    print("\n  status:", dict(sorted(by_status.items())), file=sys.stderr)
    print("  host:  ", dict(sorted(by_host.items())), file=sys.stderr)


if __name__ == "__main__":
    main()
