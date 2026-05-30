# ingest/ — audio acquisition (download stage)

Downloads the audio behind each scraped track into the canonical store. Keyed
on `track_id` / `recording_id` (minted by [tokenizer/](../tokenizer/CLAUDE.md));
idempotent over `track_audio` (`stem`, `variant` on each row). Inserts go through
`core.db.insert_audio_or_reap()` so a failed DB write does not strand files on
disk. Lands files at
`{audio_root}/objects/{track_id}/{track_id}__{platform}__{player_id}.{ext}`.

> Consumes the tokenizer's remix/version-qualifier rule — the remixer qualifier
> in `full_name` is what makes YT Music searches resolve the right release.
> That rule lives in [tokenizer/CLAUDE.md](../tokenizer/CLAUDE.md); don't restate
> it here.

## Download topology

This is *not* "yt-dlp + spotdl in one chain" — it's a yt-dlp main path, a spotdl
retry pass, and a YT Music rescue path. Three distinct entrypoints:

| Tool | Source for URLs | Fallback chain | When to use |
|---|---|---|---|
| [main.py](main.py) | scraped `dj_set_track_media_links` | `youtube → soundcloud` (see [main.py:76](main.py#L76)) | Production. Idempotent over `track_audio` |
| [main_retry.py](main_retry.py) | scraped Spotify URLs | spotdl only | Targeted retry on tracks with a Spotify URL but no `track_audio` row. Slow; needs real `SPOTIFY_CLIENT_ID`/`SECRET` (bundled spotdl creds are globally rate-limited) |
| [scripts/redownload_via_ytmusic.py](../scripts/redownload_via_ytmusic.py) | metadata search (`full_name`) | YT Music → yt-dlp | Two-phase rescue: Phase 1 inserts `platform='youtube_music'` rows alongside existing yt-dlp ones; Phase 2 (gated by `--no-replace` default-off) deletes the noisy yt-dlp rows + cascades + unlinks files. Use after a corpus run to upgrade noisy 1001tracklists scrape URLs to clean Topic-channel masters |

**Why spotdl is not in the main chain:** a 14h production run produced **zero**
successes and 174 timeouts ([main.py:65-75](main.py#L65-L75)) — spotdl's
anonymous YT Music search is rate-limited and slow without real Spotify creds.
Inline comment explains the move.

## yt-dlp specifics

- Needs Netscape `cookies.txt` for ~5–15% age-gated YouTube ([downloader.py:61](adapters/downloader.py#L61)).
- Needs a JS runtime (`node` or `nodejs` in PATH) to deobfuscate YouTube's
  n-parameter, otherwise stream URLs return only image formats
  ([downloader.py:35-43](adapters/downloader.py#L35-L43)).

When downloads fail en masse with "Sign in to confirm you're not a bot" or
"No supported JavaScript runtime", the `feedback_ytdlp_bot_detection_recipe`
memory (and the `audio-pipeline-debug` skill) have the exact 3-step recovery.

## One-off surgery

[scripts/replace_track_audio.py](../scripts/replace_track_audio.py) — swap one
track's audio by URL or local file. Destructive when replacing an existing row
(deletes old row + cascades); `--promote-reference` / `--purge-siblings` for
inventory hygiene. [scripts/acquire_variant.py](../scripts/acquire_variant.py)
**adds** a sibling row with `stem=acappella|instrumental` (does not replace the
`regular` reference). Corrections log to `track_audio_correction` by axis
(`version` | `variant` | `stem`) via [corrections.py](corrections.py).

[scripts/reconcile_orphans.py](../scripts/reconcile_orphans.py) — disk↔DB
orphan routing (dry-run default). See
[docs/identity_and_inventory_plan.md](../docs/identity_and_inventory_plan.md) and
[docs/agent_handoff_reconcile_20260530.md](../docs/agent_handoff_reconcile_20260530.md)
before re-running `--apply` on pi-storage.

See the `replace-track-audio` skill for the full coordinated workflow.

## Deploy caveat

pi-storage systemd units that ran `python -m audio_pipeline.main` must be
updated to `python -m ingest.main` (the module was renamed out of
`audio_pipeline/`) before `make deploy`, or the service won't restart.
