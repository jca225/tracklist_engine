---
name: replace-track-audio
description: Replace the downloaded audio for a track when 1001tracklists' artist+title identification is correct but the file content is the wrong version (cover, edit, original instead of the named remix, label-mismatched master, etc.). Use when the user says "the audio is wrong for set X position N", "swap in this YouTube URL for [artist - title]", "fix the [Manse/Madison Mars/whatever] remix", "replace track [track_id] with [URL]", or "this file isn't the right version". Coordinates pi-storage DB + canonical file replacement, post-replace promotion/cleanup of stale rows, local `~/aligning/<set>/` refresh, optional Demucs re-stem on Mac MPS, and optional Essentia feature recompute + iTunes-tag refresh.
---

# Replace Track Audio Workflow

This skill encodes the multi-step audio replacement recipe. The single biggest failure mode is **stopping after the new row is inserted** — without promote + cleanup, the pull script still picks the stale YT-Music / Spotify version and the local aligning folder keeps the wrong audio.

## CRITICAL: don't skip phases

Phases 1–3 are mandatory. 4–5 are optional but commonly wanted.

| Phase | What | Where |
|---|---|---|
| 1. Identify | Resolve to a `track_id` + verify the target | Mac, via pi-storage SQL |
| 2. Replace | Insert new `track_audio` row at canonical path | pi-storage |
| 3. Promote + purge | `is_reference=1` on new, delete other rows + their files/stems | pi-storage |
| 4. Refresh aligning | Overwrite local m4a, kill `.asd`, delete stale stem subdir | Mac |
| 5. Re-stem + re-tag | Demucs + Essentia + iTunes tag + filename `[NNbpm KK]` | Mac |
| 6. Log the case | Append the acquisition-case attempt to the Mac corpus | Mac |

## Phase 1: Identify the target track

Disambiguate "position" first — there are three:

- **Tracklist section number** (1001tl's published "01 / 02 / ... 22 ...") — what users usually mean
- **File label in aligning folder** (`076__Artist - Title.m4a` after the new naming scheme this is `022__...`)
- **`row_index` in `dj_set_rows`** (raw scrape row, includes non-track header rows)

For "position N in set X", resolve via published section_no:

```bash
ssh pi-storage 'sqlite3 /mnt/storage/data/db/music_database.db "
SELECT row_index, SUBSTR(text_excerpt, 1, 100),
       json_extract(data_attrs_json, \"\$.\"\"data-trackid\"\"\") AS track_id
FROM dj_set_rows
WHERE set_id = \"<SET_ID>\" AND data_attrs_json LIKE \"%trackid%\"
  AND text_excerpt LIKE \"<NN> %\"
ORDER BY row_index LIMIT 5;"'
```

The section-number prefix in `text_excerpt` (`"22 30:16 Thomas Gold ..."`) is the published numbering. Confirm with the user before destroying audio.

## Phase 2: Replace via existing script

Run on pi-storage (the script writes to `/mnt/storage/`):

```bash
ssh pi-storage 'cd /home/johncabrahams/tracklist_engine && \
  venvs/audio/bin/python -m scripts.replace_track_audio \
    --track-id <TRACK_ID> --url <YOUTUBE_URL>'
```

If yt-dlp on pi-storage hits "Sign in to confirm you're not a bot" or "No supported JavaScript runtime": **fall back to Mac-side download** (different IP often passes without cookies):

```bash
# On Mac:
./venvs/audio/bin/yt-dlp -f "bestaudio[ext=m4a]/bestaudio" \
  -o "/tmp/replacement.%(ext)s" "<URL>"
scp /tmp/replacement.m4a pi-storage:/tmp/

# Then on pi-storage in --file mode (use the YT video ID as player-id
# so the canonical filename still reveals provenance):
ssh pi-storage 'cd /home/johncabrahams/tracklist_engine && \
  venvs/audio/bin/python -m scripts.replace_track_audio \
    --track-id <TRACK_ID> --file /tmp/replacement.m4a \
    --player-id <YT_VIDEO_ID>'
```

(For the cookie-refresh path itself, see the `feedback_ytdlp_bot_detection_recipe` memory.)

The script inserts a new row and prints `taid=<NEW_TAID>`. Capture it.

**Correction ledger (automatic):** on success the script appends a row to
`track_audio_correction` — the training signal for the acquisition gates
(version / variant / stem). Defaults to `axis=version`; enrich with
`--axis {version|variant|stem}`, `--reason "..."`, `--set-id`, `--position`.
It records `action=replace` only when you pass `--track-audio-id` (so it can
snapshot + delete the retired row); with `--track-id` alone it records
`action=add`. Pass `--no-log` to suppress (e.g. you'll log manually in Phase 3
for the multi-row purge).

## Phase 3: Promote + purge stale rows (the easy-to-miss part)

`scripts/replace_track_audio.py` only deletes the old row if you pass `--track-audio-id`. Even then, it deletes one row, not all the stale ones. And it never sets `is_reference=1` on the new row. The pull script orders by `is_reference DESC, platform=manual sorts last` — so without this phase, the OLD YT-Music row still wins on the next pull.

**3a. Log the correction first** (while the stale rows still exist, so
`--old-taid` resolves) — authoritative for the multi-row purge flow. Use this
when you ran Phase 2 with `--no-log`:

```bash
ssh pi-storage 'cd /home/johncabrahams/tracklist_engine && \
  venvs/audio/bin/python -m ingest.corrections \
    --track-id <TRACK_ID> --axis version --action replace \
    --old-taid <OLD_TAID> --new-taid <NEW_TAID> \
    --set-id <SET_ID> --position <NN> --reason "<why the old audio was wrong>"'
```

(`--axis variant` for wrong edit-length, `--axis stem` for wrong acappella/instrumental.)

**3b. Promote + purge**, one shot, on pi-storage:

```bash
ssh pi-storage 'sqlite3 /mnt/storage/data/db/music_database.db <<SQL
PRAGMA foreign_keys = ON;
BEGIN;
UPDATE track_audio SET is_reference=0 WHERE track_id="<TRACK_ID>";
UPDATE track_audio SET is_reference=1 WHERE track_audio_id=<NEW_TAID>;
DELETE FROM track_audio WHERE track_id="<TRACK_ID>" AND track_audio_id != <NEW_TAID>;
COMMIT;
SELECT track_audio_id, platform, is_reference, path
  FROM track_audio WHERE track_id="<TRACK_ID>";
SQL'
```

The `DELETE` cascades through FKs to `track_stems`, `track_audio_features`, `track_analysis`, `track_identity`, `track_mert_sections`, `track_measures`. **The on-disk files don't cascade.** Inventory them, then remove:

```bash
ssh pi-storage 'ls /mnt/storage/objects/<TRACK_ID>/ && \
  ls -d /mnt/storage/stems/<OLD_TAID_1> /mnt/storage/stems/<OLD_TAID_2> 2>/dev/null'
ssh pi-storage 'rm -f /mnt/storage/objects/<TRACK_ID>/<TRACK_ID>__youtube_music__*.m4a \
                       /mnt/storage/objects/<TRACK_ID>/<TRACK_ID>__spotify__*.m4a && \
                 rm -rf /mnt/storage/stems/<OLD_TAID_1> /mnt/storage/stems/<OLD_TAID_2>'
```

Only the new manual file should remain under `objects/<TRACK_ID>/`.

## Phase 4: Refresh local `~/aligning/<set>/`

If a user has an active alignment session for this set, the wrong audio is sitting at the old file path. Find it:

```bash
ls "/Users/johnnycabrahams/aligning/<SET_PREFIX>__"*/tracks/<LABEL>__*.m4a
```

`<LABEL>` is whatever the pull script assigned (new scheme: `022` / `022w1`; legacy folders may have `076` etc. — check what's actually on disk). User-tag suffix like `[128bpm 3B]` is optional and stays in the filename.

Copy fresh audio over the existing file, kill the `.asd` waveform cache, and remove the stale stem subdir (those stems were Demucsed from the WRONG audio):

```bash
DIR="/Users/johnnycabrahams/aligning/<SET_PREFIX>__<sanitized-title>"
SLOT="<LABEL>__<Artist> - <Title> (<VersionTag>)"  # match the filename minus ext + user tag

# 1. Copy new audio over existing file (preserves filename and any user-tag suffix):
rsync -L pi-storage:/mnt/storage/objects/<TRACK_ID>/<TRACK_ID>__manual__*.m4a "$DIR/tracks/${SLOT}*.m4a"
# (if the glob is ambiguous, ls the exact target first)

# 2. Kill .asd so Ableton regenerates the waveform:
rm -f "$DIR/tracks/${SLOT}"*.m4a.asd

# 3. Delete stale stem subdir (mirror of pi-storage delete):
rm -rf "$DIR/stems/${SLOT}"*
```

**Ableton risk**: if the set is open in Ableton with the file referenced, the audio in the current session keeps playing from buffered state — the swap is invisible until next reload. The clip's waveform display refreshes once `.asd` is regenerated. No `.als` relink needed (filename is unchanged).

## Phase 5 (optional): Re-stem + re-tag

The Phase 3 cascade deleted the old `track_audio_features` row, so the track has no BPM/key in the DB and no stems anywhere. Re-running:

**Demucs on Mac MPS** (~22s for a 3.5 min track):

```bash
SRC="$DIR/tracks/${SLOT}*.m4a"  # the local file you just refreshed
DST="$DIR/stems/${SLOT}"
TMP=$(mktemp -d)
./venvs/audio/bin/python -m demucs --two-stems vocals --flac -d mps -o "$TMP" "$SRC"
mv "$TMP/htdemucs/$(basename "$SRC" .m4a)/vocals.flac" "$DST/vocals.flac"
mv "$TMP/htdemucs/$(basename "$SRC" .m4a)/no_vocals.flac" "$DST/instrumental.flac"
rm -rf "$TMP"
```

(Stems aren't pushed back to pi-storage `/mnt/storage/stems/<NEW_TAID>/` automatically — the analyze loop will populate that path the next time it processes this taid.)

**Essentia features** (Mac venv subprocess, ~60s):

```bash
DYLD_LIBRARY_PATH="/opt/homebrew/opt/expat/lib:${DYLD_LIBRARY_PATH:-}" \
  PYTHONPATH="$(pwd)" \
  ./venvs/essentia/bin/python -m analysis.adapters.essentia_worker "$SRC"
```

Outputs JSON on stdout. Pull `rhythm.bpm`, `key.tonic`, `key.mode`, `danceability_tf`, `valence`. Map to Camelot:

```
KEY_PC_TO_CAMELOT_MAJOR = ["8B","3B","10B","5B","12B","7B","2B","9B","4B","11B","6B","1B"]
KEY_PC_TO_CAMELOT_MINOR = ["5A","12A","7A","2A","9A","4A","11A","6A","1A","8A","3A","10A"]
# pitch class for "C"=0, "C#"=1, ..., "B"=11
```

Persist to pi-storage so the DB matches what the local m4a tags claim:

```bash
ssh pi-storage 'sqlite3 /mnt/storage/data/db/music_database.db "
INSERT INTO track_audio_features
  (track_audio_id, source, key_pc, key_mode, bpm, danceability, valence, key_strength)
VALUES (<NEW_TAID>, \"essentia_v2\", <PC>, \"<major|minor>\", <BPM>, <DNCE>, <VAL>, <KEY_STRENGTH>);"'
```

Write iTunes tags + rename file with `[NNbpm KK]` suffix (matches the annotator rename convention):

```python
from mutagen.mp4 import MP4
from pathlib import Path
f = Path("<SRC>")
audio = MP4(f)
audio["tmpo"] = [round(bpm)]
audio["----:com.apple.iTunes:initialkey"] = [camelot.encode("utf-8")]
audio["\xa9cmt"] = [f"essentia bpm={round(bpm)} key={camelot} dnce={dnce:.2f} val={val:.2f}"]
audio.save()
f.rename(f.with_name(f"{f.stem} [{round(bpm)}bpm {camelot}]{f.suffix}"))
# Also rename the stem dir to match:
stem = Path("<DST>")
stem.rename(stem.with_name(f"{stem.name} [{round(bpm)}bpm {camelot}]"))
```

## Phase 6: Log the acquisition case (Mac)

The acquisition-case corpus (`data/acquisition_cases/{set_id}.jsonl`) records
*why* each slot's audio was fixed — the decision trace the future ingest harness
trains on. It lives on the **Mac**; `replace_track_audio.py` runs on pi-storage,
so on success it **emits** a `ACQUISITION_CASE\t<json>` line on stdout that the
Mac-side logger persists. Pipe the ssh output straight through it:

```bash
ssh pi-storage 'cd /home/johncabrahams/tracklist_engine && \
  venvs/audio/bin/python -m scripts.replace_track_audio \
    --track-audio-id <OLD_TAID> --url <URL> \
    --set-id <SET_ID> --position <SLOT> --axis version \
    --reason "original, not the <Remixer> remix"' \
  | venvs/audio/bin/python scripts/log_acquisition.py --from-stdin
```

The emit only fires when `--set-id` + `--position` are passed (the case is keyed
per set/slot). `--axis version` → problem class `wrong_version`; a stem swap →
`suboptimal_stem`. To log without re-running the replace (e.g. a SQL-only fix),
call the logger directly:

```bash
venvs/audio/bin/python scripts/log_acquisition.py \
  --set-id <SET_ID> --position <SLOT> --recording-id <TRACK_ID> \
  --problem wrong_version --url <URL> --reason "..."
```

This is the full-track sibling of the stem driver's built-in `--no-case-log`
hook ([ingest_stem_url.py](../../../scripts/ingest_stem_url.py)); both feed one
Mac-side corpus.

## Examples

**"The audio for BB12 position 22 is wrong, should be this URL"**
1. Phase 1: section 22 of `1fsnxchk` → `track_id=1r8f4fc5` (Thomas Gold - Saints & Sinners Manse Remix).
2. Phase 2: yt-dlp on pi-storage hits bot detection → Mac download → scp → `--file` mode → taid=20758.
3. Phase 3: set is_reference=1 on 20758, delete the stale yt_music (4048) and spotify (3729) rows + their files + their stems dirs.
4. Phase 4: rsync new file into `aligning/1fsnxchk__.../tracks/076__Thomas Gold - Saints & Sinners (Remix) [128bpm 3B].m4a`; remove `.asd`; remove stems subdir.
5. Phase 5: Demucs → vocals.flac + instrumental.flac in aligning stems dir; Essentia → 128 BPM C# major → 3B → INSERT features, tag M4A, file already has correct `[128bpm 3B]` suffix.

## Troubleshooting

- **yt-dlp "Sign in to confirm you're not a bot"** on pi-storage → see `feedback_ytdlp_bot_detection_recipe` memory; or fall back to Mac download (Mac IP often passes anonymously).
- **`replace_track_audio.py` reports success but pull_set_for_alignment still picks old audio** → you skipped Phase 3. The new row is `platform=manual` which sorts dead last unless `is_reference=1`.
- **`dst exists, skipping` from `migrate_aligning_naming.py` after a replace** → the OLD file at the OLD label is still there; safe to delete since new audio is at the new path.
- **Filename has stale `[NNbpm KK]` suffix** after audio swap → the old BPM/key were computed from the wrong audio. Either drop the suffix immediately (Phase 4 rename) or compute fresh features in Phase 5 and overwrite.
- **Aligning folder isn't user-tagged yet** → the file at the old position may already match the new naming scheme; just verify what's actually on disk vs the manifest, since `manifest.json` lags reality after migrations.

## Related skills

- `alignment-pull` — initial set pull + tag workflow; this skill is the inverse-correction.
- `pi-storage-query` — read-only DB inspection patterns for Phase 1 identification.
