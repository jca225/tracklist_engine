"""Pin the Mac yt-dlp profile invariants.

The three recurring-failure flags (web_safari player_client, Safari cookies,
EJS/node runtime) used to be copy-pasted across the Mac rescue scripts. They now
live in one place; these tests fail loudly if either the CLI or the Python-API
form drops one of them or the two drift apart.
"""

from __future__ import annotations

from ingest.ytdlp_profile import (
    MAC_COOKIES_BROWSER,
    MAC_PLAYER_CLIENT,
    apply_mac_ytdlp_opts,
    mac_ytdlp_base,
)


def test_cli_base_carries_all_three_fixes():
    args = mac_ytdlp_base("/bin/yt-dlp", "/usr/bin/node")
    joined = " ".join(args)
    assert args[0] == "/bin/yt-dlp"
    assert f"youtube:player_client={MAC_PLAYER_CLIENT}" in args
    assert MAC_COOKIES_BROWSER in args  # --cookies-from-browser safari
    assert "node:/usr/bin/node" in joined  # JS runtime
    assert "ejs:github" in args  # EJS solver components


def test_cli_base_node_fallback_when_unspecified():
    # No node passed -> still resolves to *some* node path, never empty.
    args = mac_ytdlp_base("/bin/yt-dlp")
    js = args[args.index("--js-runtimes") + 1]
    assert js.startswith("node:") and len(js) > len("node:")


def test_api_opts_carry_same_player_client_and_cookies():
    opts = apply_mac_ytdlp_opts({"format": "bestaudio/best"}, node_bin="/usr/bin/node")
    assert opts["extractor_args"]["youtube"]["player_client"] == [MAC_PLAYER_CLIENT]
    assert opts["cookiesfrombrowser"] == (MAC_COOKIES_BROWSER,)
    assert opts["js_runtimes"]["node"]["location"] == "/usr/bin/node"
    # Must not clobber caller-supplied opts.
    assert opts["format"] == "bestaudio/best"


def test_api_opts_does_not_mutate_input():
    src = {"format": "x"}
    apply_mac_ytdlp_opts(src)
    assert src == {"format": "x"}
