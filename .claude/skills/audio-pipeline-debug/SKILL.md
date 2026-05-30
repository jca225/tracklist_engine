---
name: audio-pipeline-debug
description: Recover the yt-dlp download pipeline on pi-storage when it fails with bot detection ("Sign in to confirm you're not a bot") or JS runtime errors ("No supported JavaScript runtime"). Both failure modes are recurring and have an exact 3-step recipe. Use when downloads start failing en masse, scripts/redownload_via_ytmusic.py errors out, audio_pipeline/main.py reports many failures, or the user mentions yt-dlp / cookies / JS runtime / "Sign in to confirm" errors. Triggers on "yt-dlp is broken", "downloads are failing", "yt-dlp bot detection", "refresh cookies", "Sign in to confirm" error.
---

# Audio Pipeline Debug — yt-dlp recovery

The yt-dlp downloader on pi-storage has two related failure modes that recur every few weeks. Both have a fixed recipe. **Do all three steps in order — don't skip any, even if only one error signature is visible.** Both failures often appear together.

## Failure signatures

1. `WARNING: [youtube] ...: No supported JavaScript runtime could be found. Only deno is enabled by default`
2. `ERROR: [youtube] ...: Sign in to confirm you're not a bot. Use --cookies-from-browser or --cookies for the authentication`

**Why these break:**
- **JS runtime**: yt-dlp 2026.05.03 dropped JS-less YouTube extraction. Pi-storage doesn't have `deno` installed (the new default) but has `/usr/bin/node`. Need to tell yt-dlp explicitly.
- **Cookies**: Google invalidates server-IP-originated sessions aggressively — days to a week or two. Pi-storage's `~/.config/yt-dlp/cookies.txt` becomes stale.

## Step 1 — Ensure pi-storage has a JS-runtime config

Check current state:
```bash
ssh pi-storage 'cat ~/.config/yt-dlp/config'
```

Required contents:
```
--js-runtimes node:/usr/bin/node
```

If missing or different, write it:
```bash
ssh pi-storage 'cat > ~/.config/yt-dlp/config <<EOF
--js-runtimes node:/usr/bin/node
EOF'
```

## Step 2 — Refresh cookies from the Mac's Safari

**Pre-req:** Terminal must have **Full Disk Access** in System Settings → Privacy & Security → Full Disk Access. The user has granted this before; if it has reverted (macOS updates sometimes reset this), grant it again.

Extract cookies on the Mac:
```bash
/Users/johnnycabrahams/Desktop/tracklist_engine/venvs/audio/bin/yt-dlp \
  --cookies-from-browser safari \
  --cookies /tmp/yt_cookies.txt \
  --skip-download \
  --print "%(id)s|%(title)s" \
  "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

Expected: `/tmp/yt_cookies.txt` is written (~250 KB, with ~25 youtube.com lines). The command itself **may warn** about "Signature solving failed" or "Only images available" — **that's fine**. Cookies are extracted before any format-resolution starts. Don't be alarmed.

Ship to pi-storage:
```bash
scp /tmp/yt_cookies.txt pi-storage:~/.config/yt-dlp/cookies.txt
```

## Step 3 — Verify end-to-end

```bash
ssh pi-storage '~/tracklist_engine/venvs/audio/bin/yt-dlp \
  --cookies ~/.config/yt-dlp/cookies.txt \
  --print "%(id)s|%(title)s" --skip-download \
  "https://www.youtube.com/watch?v=dQw4w9WgXcQ"'
```

Expected: prints `dQw4w9WgXcQ|Rick Astley - Never Gonna Give You Up...` (or whatever Rick-roll title) with no errors. If you still see bot-detection or JS-runtime errors, something in step 1 or 2 didn't take — re-check the config file contents and that cookies.txt actually got copied.

## After recovery — resume the pipeline

If a long-running script was failing, resume:

- **Main production downloads** ([audio_pipeline/main.py](audio_pipeline/main.py)): just re-run — idempotent over `track_audio`.
- **YT Music rescue** ([scripts/redownload_via_ytmusic.py](scripts/redownload_via_ytmusic.py)): re-run; Phase 1 is idempotent on `platform='youtube_music'` rows.
- **Spotify retry** ([audio_pipeline/main_retry.py](audio_pipeline/main_retry.py)): different failure profile — needs real `SPOTIFY_CLIENT_ID`/`SECRET` env vars. Not fixed by this recipe; see the inline comment at [main.py:65-75](audio_pipeline/main.py#L65) for why this path is slow.

## When this recipe is NOT the right fix

- If a single track fails but others succeed: the URL itself may be dead, age-gated without cookies covering it, or geographically blocked. Don't refresh cookies for one-off failures.
- If failures are 100% on `youtube_music` searches and the URLs themselves work: the YT Music search query is the problem, not yt-dlp. Check `full_name` handling in `redownload_via_ytmusic.py` (the remix/version-qualifier rule).
- If failures are on SoundCloud, not YouTube: this recipe is YouTube-specific. SoundCloud has its own failure modes.

## Anti-patterns

- ❌ Skipping step 1 because "only cookies look broken". The JS runtime config silently makes everything fall back to image-only formats, which then look like cookie failures downstream.
- ❌ Pulling cookies from Chrome on macOS — Safari is what's logged into YouTube on this Mac. Wrong browser = empty cookies file.
- ❌ Running the cookie-extract on pi-storage (no browser there). Cookies MUST be extracted on the Mac and shipped via scp.
- ❌ Updating yt-dlp expecting it to fix bot detection. yt-dlp updates don't refresh Google session cookies — only re-extracting from a browser does.
