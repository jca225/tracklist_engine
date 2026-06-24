"""Single source of truth for the Mac yt-dlp invocation profile.

Three recurring YouTube failure modes each need a specific flag, and the fixes
used to be copy-pasted across the Mac rescue scripts. They live here once:

1. **No JS runtime** — YouTube's n-parameter obfuscation needs a JS interpreter
   to deobfuscate stream URLs. ``--js-runtimes node:<path>`` + ``--remote-components
   ejs:github`` (yt-dlp-ejs solver scripts; library default is deno, absent on the Mac).
   Without it, videos return only image formats.
2. **HTTP 403 PO-token gate** — YouTube 403s the default ``tv`` player_client's media
   URLs. ``--extractor-args youtube:player_client=web_safari`` serves the m4a stream
   with Safari cookies.
3. **Bot / age gate** — ``--cookies-from-browser safari`` supplies a live session.

The CLI form (``mac_ytdlp_base``) is for scripts that shell out to the yt-dlp
binary; the YoutubeDL-API form (``apply_mac_ytdlp_opts``) is for code using the
Python API (``ingest.adapters.downloader``, ``scripts.fetch_candidate_stems``).
Both must stay in sync — ``tests/test_ytdlp_profile.py`` pins the invariant.
"""

from __future__ import annotations

import shutil
from pathlib import Path

#: player_client that bypasses the PO-token 403 on the Mac (see module docstring).
MAC_PLAYER_CLIENT = "web_safari"
MAC_COOKIES_BROWSER = "safari"
#: Prefer a clean m4a stream, fall back to best available.
DEFAULT_AUDIO_FORMAT_FILTER = "ba[ext=m4a]/bestaudio[ext=m4a]/bestaudio/best"


def _resolve_node(node_bin: str | None = None) -> str:
    return (
        node_bin
        or shutil.which("node")
        or shutil.which("nodejs")
        or "/opt/homebrew/bin/node"
    )


def mac_ytdlp_base(
    ytdlp_bin: str | Path,
    node_bin: str | None = None,
    *,
    audio_format_filter: str = DEFAULT_AUDIO_FORMAT_FILTER,
) -> list[str]:
    """CLI arg prefix for shelling out to the yt-dlp binary on the Mac.

    Replaces the formerly-duplicated ``YTDLP_BASE`` lists. Append the URL (and
    any per-call flags like ``-o``) after this prefix.
    """
    return [
        str(ytdlp_bin),
        "--js-runtimes",
        f"node:{_resolve_node(node_bin)}",
        "--remote-components",
        "ejs:github",
        "--cookies-from-browser",
        MAC_COOKIES_BROWSER,
        "--extractor-args",
        f"youtube:player_client={MAC_PLAYER_CLIENT}",
        "-f",
        audio_format_filter,
    ]


def apply_mac_ytdlp_opts(
    opts: dict,
    *,
    cookies_browser: str = MAC_COOKIES_BROWSER,
    node_bin: str | None = None,
) -> dict:
    """Apply the same Mac profile to a YoutubeDL Python-API opts dict.

    Mutates a copy: sets the ``web_safari`` player_client, Safari cookie source,
    and node JS runtime. Use only on the Mac — pi-storage uses a ``cookies.txt``
    file and the default player_client.
    """
    out = dict(opts)
    extractor_args = {**out.get("extractor_args", {})}
    youtube = {**extractor_args.get("youtube", {})}
    youtube["player_client"] = [MAC_PLAYER_CLIENT]
    extractor_args["youtube"] = youtube
    out["extractor_args"] = extractor_args
    out["cookiesfrombrowser"] = (cookies_browser,)
    out["js_runtimes"] = {"node": {"location": _resolve_node(node_bin)}}
    return out
