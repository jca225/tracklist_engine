# Appleseed librarian — two-process runbook

The Appleseed mashup compiler (`~/Desktop/mashup_compiler`) lets you type a
song name or paste a YouTube/SoundCloud URL into its library UI. The product
repo deliberately contains no downloader — it writes a request record and
waits. A separate **librarian** process in this repo (`tracklist_engine`)
fulfils those requests using the ingest stack (yt-dlp + ytmusicapi).

## Architecture

```
mashup_compiler/server/   ←── product process (uvicorn)
  state.db                       song_requests table
  library/                       WAV files land here

tracklist_engine/         ←── librarian process (polls state.db)
  scripts/appleseed_librarian.py
```

Shared surface: the `state.db` path + `library/` folder. No network call
passes between the two processes — the librarian polls the DB, claims a row
atomically, downloads the audio, transcodes to 44.1 kHz WAV into `library/`,
and marks the row done. The server's background scan then picks up the new
file.

## Commands (John's Mac)

**Terminal 1 — server:**

```bash
cd ~/Desktop/mashup_compiler
venv/bin/uvicorn server.app:app --host 0.0.0.0 --port 8500
```

**Terminal 2 — librarian:**

```bash
cd ~/Desktop/tracklist_engine
venvs/audio/bin/python -m scripts.appleseed_librarian \
    --db ~/Desktop/mashup_compiler/server/state.db \
    --library ~/Desktop/mashup_compiler/server/library
```

The librarian runs a poll loop (5 s idle sleep). Pass `--once` to process a
single request and exit (useful for manual testing).

## Prerequisites

- **ffmpeg** on PATH — used for 44.1 kHz WAV transcode after download.
- **node/nodejs** on PATH — required for yt-dlp n-challenge deobfuscation;
  without it, yt-dlp returns only image formats for most current YouTube URLs.
- **ytmusicapi** — search is anonymous; no account required for name lookups.
- **cookies.txt (optional)** — needed for age-gated YouTube videos. See the
  yt-dlp bot-check recipe: `audio-pipeline-debug` skill or the project memory
  entry `feedback_ytdlp_bot_detection_recipe`. Place the cookies file at the
  path passed to yt-dlp (default: not used unless configured).

Both `ffmpeg` and `node` are already present on this Mac.

## Request lifecycle

1. **searching** — UI submits the name or URL; server writes a
   `song_requests` row with `status='searching'`.
2. **fetching** — librarian claims the row atomically (`BEGIN IMMEDIATE`,
   flips to `'fetching'` so a second process can't double-download), then:
   - `kind='url'`: parses the YouTube video ID or SoundCloud URL and calls
     `downloader.download_one`.
   - `kind='name'`: calls `ytmusic_adapter.search(query, limit=8)`,
     selects the shortest full-length studio result via `_best_hit` (filters
     previews, live recordings, anything < 60 s), then downloads with
     `_ytdlp_download`.
3. **wav lands** — downloaded audio is transcoded to 44.1 kHz stereo WAV and
   written to `library/<safe_name>.wav`.
4. **analyzing** — server background scan detects the new WAV, queues it for
   analysis. **The first time a song is added this takes ~30 minutes**: full
   Roformer stem separation (vocal + instrumental extraction) + BPM/key
   analysis. This one-time cost is expected — the results are cached for all
   future mashups with that song.
5. **ready** — analysis complete; the song appears in the library as mashable.

The `fetching` status is visible to the server's pending-requests list while
the download is in flight.

## Failure modes

- **"not found"** — no full-length hit found on YT Music (all results were
  previews, live recordings, or shorter than 60 s), or the yt-dlp download
  failed. The UI surfaces the `error` field from `song_requests`. If a name
  search fails, try pasting a direct YouTube or SoundCloud URL instead.
- **Age-gated content** — yt-dlp returns an error; provide a `cookies.txt`
  from a logged-in browser session (see bot-check recipe above).
- **Bot detection / JS runtime error** — ensure `node` is on PATH; if yt-dlp
  still fails, refresh browser cookies and re-export. See
  `feedback_ytdlp_bot_detection_recipe` in project memory.

## Known soft spots (tracked in mashup_compiler BACKLOG)

- SoundCloud slug-URL `player_id` extraction is cosmetically wrong when the
  URL is `soundcloud.com/artist/slug` rather than
  `api.soundcloud.com/tracks/<id>` — download works, filename is off. Fix in
  `scripts/appleseed_librarian.py`.
- Two requests for the same song title produce the same WAV filename and the
  second silently overwrites the first. Fix: add a uniqueness suffix
  (`_<row_id>`).
