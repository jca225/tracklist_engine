# Stem discovery playbook — acappella / instrumental re-source

**Status:** 2026-05-30  
**Coordination:** [agent_handoff_identity_rollout_20260530.md](agent_handoff_identity_rollout_20260530.md) — do not start a second `tokenizer.materialize`.

**Related:** [alignment_review_20260530.md](alignment_review_20260530.md), [alignment_objective.md](alignment_objective.md).

**Reference:** [UVR vocal/instrumental extraction chain](assets/UVR_Instructions.png) — Ultimate Vocal Remover model pipeline (Kim Vocals 2 → Karaoke ensemble → Reverb/De-Echo/Denoise; Demucs htdemucs v4 for instrumental). External community recipe (credit: LucidFir, soucarlosxavier).

---

## URL-first (primary path)

**pi-storage is the library; `~/aligning/` is the desk.** Paste a YouTube URL on the Mac; pi downloads into `objects/`; optional pull refreshes Ableton files. **Downloads is not on the critical path.**

```bash
# Add a new stem (does NOT promote by default — pass --promote if this track_id is stem-only in the set)
venvs/audio/bin/python scripts/ingest_stem_url.py \
  --url 'https://www.youtube.com/watch?v=...' \
  --track-id TRACK_ID \
  --role acappella \
  --set-id SET_ID --position 022 \
  --reason 'quality:good|identity:OK|note:studio YT acapella' \
  --fail-on fallback,wrong_song

# Replace a bad stem row (promotes by default)
venvs/audio/bin/python scripts/ingest_stem_url.py \
  --url '...' \
  --track-audio-id OLD_TAID \
  --set-id SET_ID --position 022 \
  --reason 'quality:good|...'

# Refresh ~/aligning/ after ingest (use --aligning-dest ~/aligning-v2 if old .als still points at a frozen folder)
venvs/audio/bin/python scripts/ingest_stem_url.py ... --pull --set-id SET_ID

# Preview remote command only
venvs/audio/bin/python scripts/ingest_stem_url.py ... --dry-run
```

**Promote policy:** add → no `is_reference` unless `--promote`. replace → promote on. See decision tree below.

**If pi yt-dlp fails:** script prints audio-pipeline-debug hints; Mac fallback:

```bash
venvs/audio/bin/yt-dlp -f 'bestaudio[ext=m4a]/bestaudio' -o /tmp/stem.m4a 'URL'
venvs/audio/bin/python scripts/ingest_stem_url.py --file /tmp/stem.m4a \
  --track-id TRACK_ID --role acappella --reason '...'
```

Pi must be at `507b08d+`: `ssh pi-storage 'cd ~/tracklist_engine && git pull'`.

---

## Human walkthrough

1. **Listen** — search YouTube (`{artist} {title} acapella`), pick URL by ear.
2. **Add vs replace** — bad row already in DB? use `--track-audio-id`. First stem? use `--track-id` + `--role`.
3. **Same track_id as regular elsewhere in set?** — add **without** `--promote`; pull may still give regular for other slots until pull v2. Use GT `claimed_stem` for truth. If stem-only track_id in set, add `--promote`.
4. **Run** `ingest_stem_url.py` (above).
5. **Read fingerprint** — `FALLBACK_TO_ORIGINAL` / `WRONG_SONG` → try another URL with replace mode.
6. **Optional `--pull`** — drag from `~/aligning/<set>/tracks/` (filenames include `(Acappella)` when compound suffix applies).
7. **GT** — `claimed_stem` + timings → `write_back_ground_truth.py`.

---

## Baby rule

- One file per slot under `~/aligning/<set>/tracks/`.
- Acappella/instrumental plays should be `track_audio.stem` + usually `is_reference=1` for that `track_id` when pull should serve the stem.
- Demucs `stems/vocals` are separation of the reference `track_audio_id`, not a substitute for a downloaded acapella master.

---

## Three places audio lives

| Place | Role |
|-------|------|
| Downloads / inbox | Scratch — not registered |
| pi-storage `objects/` + DB | **Canonical library** |
| `~/aligning/<set>/` | Ableton working copy |

---

## QA layers

**Layer A — identity (chromaprint):** `FALLBACK_TO_ORIGINAL`, `WRONG_SONG`, `DURATION_MISMATCH`, `WEAK_SIGNAL` (acappella — confirm by ear).

**Layer B — quality (human):** structured `reason` on `track_audio_correction` (`quality:`, `identity:`, `note:`).

---

## Direct pi commands (without Mac wrapper)

| Task | Command |
|------|---------|
| Add stem | `scripts/acquire_variant.py URL --role acappella --track-id …` (+ `--no-promote-reference` default off promote) |
| Replace stem | `scripts/replace_stem_audio.py --track-audio-id … --url … --reason …` |
| Wrong version | `scripts/replace_track_audio.py` |
| Pull | `labeling/pull_set_for_alignment.py <set_id>` |

---

## YouTube search

[ingest/search_query.py](../ingest/search_query.py) strips vocal qualifiers from YT Music auto-search. Search YouTube explicitly for stem cuts.

---

## After materialize

Audit `set_track_slots.claimed_stem` for your set. Re-source on pi. Fresh pull. GT write-back. Do not `--prune` legacy aligning trees tied to old `.als` files.

---

## Do not (pi-storage)

- Second `tokenizer.materialize`
- `reconcile_orphans.py --apply` (done)
- `reconcile_orphans.py --apply-promotions` (blocked on `5uzdn35`)
- Re-run identity migrations
- Replace `9hp84x` acappella ref (`taid=758`) without intent

---

## Verify materialize done

```bash
ssh pi-storage 'tail -3 /tmp/materialize.log'
ssh pi-storage 'sqlite3 /mnt/storage/data/db/music_database.db "SELECT COUNT(*) FROM track_metadata;"'
```
