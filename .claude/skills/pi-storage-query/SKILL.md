---
name: pi-storage-query
description: Query the canonical Tracklist Engine state on pi-storage over SSH. The repo's local data/db/music_database.db is a STALE DEV COPY — never the source of truth. Use this skill whenever the user asks about current state of the corpus (track counts, audio coverage, stems status, recent analyses, scrape progress, set metadata, anything from track_audio / dj_sets / set_audio / track_analysis / track_audio_features / set_measures etc.) or asks "do we have X". Triggers on phrases like "how many tracks have stems", "what's the scrape failure count", "check pi-storage for X", "query the canonical DB", "is set Y downloaded", or any question whose answer would change as services run on pi-storage.
---

# Pi-Storage Canonical-DB Query

## Critical rule — read this first

The local file `data/db/music_database.db` in this repo is a **stale dev copy**. Services on pi-storage write to the canonical DB continuously; the local copy diverges within hours. **Never** use the local copy to answer "what's the current state" questions. Always go to pi-storage.

## Canonical paths

| Resource | Path on pi-storage |
|---|---|
| DB | `/mnt/storage/data/db/music_database.db` |
| Track audio files | `/mnt/storage/objects/{track_id}/{track_id}__{platform}__{player_id}.{ext}` |
| Demucs stems | `/mnt/storage/stems/{track_audio_id}/{vocals,drums,bass,other,instrumental}.{ext}` |

## The one-liner pattern

```bash
ssh pi-storage 'sqlite3 /mnt/storage/data/db/music_database.db "SELECT ..."'
```

Quote rules: outer single quotes for the ssh command body, double quotes inside for the SQL. If the SQL itself contains a single quote (e.g. `LIKE 'foo'`), switch the ssh body to a heredoc:

```bash
ssh pi-storage <<'EOF'
sqlite3 /mnt/storage/data/db/music_database.db <<'SQL'
SELECT track_id FROM dj_set_rows WHERE artists LIKE '%Garrix%' LIMIT 5;
SQL
EOF
```

For column headers + nice formatting, prefer `-cmd` flags:

```bash
ssh pi-storage 'sqlite3 -header -column /mnt/storage/data/db/music_database.db "SELECT ..."'
```

## Common queries (copy these)

**Track audio coverage:**
```bash
ssh pi-storage 'sqlite3 /mnt/storage/data/db/music_database.db "SELECT COUNT(DISTINCT track_id) FROM track_audio"'
```

**Stems coverage:**
```bash
ssh pi-storage 'sqlite3 /mnt/storage/data/db/music_database.db "SELECT COUNT(DISTINCT track_audio_id) FROM track_stems"'
```

**Scrape failure queue depth:**
```bash
ssh pi-storage 'sqlite3 /mnt/storage/data/db/music_database.db "SELECT COUNT(*) FROM scrape_failures"'
# Or just: make queue
```

**MERT embeddings coverage:**
```bash
ssh pi-storage 'sqlite3 /mnt/storage/data/db/music_database.db "SELECT COUNT(*) FROM track_mert_sections"'
```

**Track count for a specific set:**
```bash
ssh pi-storage 'sqlite3 /mnt/storage/data/db/music_database.db "SELECT COUNT(*) FROM dj_set_rows WHERE set_id = '\''<SET_ID>'\''"'
```

(Note the `'\''` quote-escape dance — easier to use the heredoc form for SQL with embedded quotes.)

## Joining main DB with aux.db

`aux.db` (research signals: Last.fm, Billboard, Spotify Charts, release years) lives **locally** at `data/analysis/aux.db`, not on pi-storage. For joins, two options:

**A. Pull a slice from pi-storage, then join locally** (preferred for one-off questions):
```bash
ssh pi-storage 'sqlite3 -csv /mnt/storage/data/db/music_database.db \
  "SELECT track_id, full_name FROM track_metadata"' > /tmp/track_meta.csv

sqlite3 data/analysis/aux.db <<'SQL'
.mode csv
.import /tmp/track_meta.csv track_meta_remote
SELECT m.track_id, c.peak_position
FROM track_meta_remote m
JOIN track_chart_match c ON c.track_id = m.track_id
LIMIT 10;
SQL
```

**B. Use the FastAPI jobqueue** (for structured queries the API exposes). Check `make logs-jobqueue` to see if it's running; otherwise `make restart-jobqueue` to bring it up.

## Identifier keys (cross-table joins)

- `track_id` — 1001tracklists identifier. Joins: `dj_set_rows`, `track_metadata`, `track_audio`, `track_analysis`, `track_audio_features`, `canonical_track_cue_points`, `track_fingerprints`, and **aux.db tables** (`track_meta`, `track_lastfm`, `track_chart_match`, `track_spotify_charts`).
- `track_audio_id` — auto-increment in `track_audio`. Joins to: `track_stems`, `track_measures`, `track_mert_sections`, `track_sections`, `set_section_alignment`, `measure_alignment`.
- `set_id` — 1001tracklists set identifier. Joins: `dj_sets`, `dj_set_rows`, `dj_set_media_links`, `set_audio`, `set_stems`, `set_measures`, `set_fingerprint_hits`, `set_section_alignment`, `set_views` (aux.db).

## Filesystem-side checks (don't trust schema alone)

If the question is "do we have audio for X" or "are stems present", DB row presence is not always ground truth — sometimes files exist without rows or vice versa. Check the filesystem too:

```bash
ssh pi-storage 'ls /mnt/storage/objects/<track_id>/ 2>/dev/null'
ssh pi-storage 'ls /mnt/storage/stems/<track_audio_id>/ 2>/dev/null'
```

(See memory `feedback_check_filesystem` — established rule: for "do we have X" audio questions, ls the drive AND check the DB.)

## What to report back

After running a query, present results as a short text answer to the user — not a SQL dump. If the result is more than ~10 rows, summarize (counts, ranges, distribution) rather than pasting all of it. Reserve raw output for when the user asks "show me the rows".

## Anti-patterns

- ❌ Querying `data/db/music_database.db` (local) for "current state" questions. It's stale.
- ❌ Forgetting to escape single quotes in the SQL inside the ssh command — use heredoc instead.
- ❌ Trying to query aux.db on pi-storage. It only exists locally on the Mac.
- ❌ Running long-running queries (full table scans on large tables) over the ssh pipe without a `LIMIT` — slow and ties up the terminal.
