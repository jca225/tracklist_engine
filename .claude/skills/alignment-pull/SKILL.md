---
name: alignment-pull
description: Pull or refresh a DJ set's mix + reference tracks + stems into ~/aligning/ on the Mac, ready for ground-truth alignment in Ableton. Wraps labeling/pull_set_for_alignment.py with knowledge of the consistency model (rsync delta refresh, --prune for orphans), the annotator rename convention ([NNNbpm KK] tags are user territory, never pruned), and downstream tagging (tag_aligning_folder.py to inject BPM+key into M4A tags). Use when the user wants to set up alignment for a set, refresh an existing aligning folder, prune orphan files, list candidate sets, or tag a pulled folder's audio with Essentia features. Triggers on phrases like "pull set X for alignment", "set up aligning for X", "refresh aligning folder", "prune the aligning folder", "tag the aligning audio".
---

# Alignment Pull Workflow

The `~/aligning/<set>/` folder is a **read-replica of pi-storage**: only `pull_set_for_alignment.py` writes to it, pi-storage's DB is the source of truth, and the folder is ephemeral (delete it once alignment data has been written back).

## The four operations

### 1. List recent candidate sets

```bash
python labeling/pull_set_for_alignment.py --list-recent
```

Use this when the user hasn't given a specific `set_id` and wants to browse what's ready to align.

### 2. Initial pull (or first-time delta refresh)

```bash
python labeling/pull_set_for_alignment.py <set_id>
```

This creates `~/aligning/<set_id>__<sanitized-title>/` with:
- `mix.<ext>` — the DJ set recording
- `tracks/NNN__Artist - Title.<ext>` — reference tracks (3-digit index = appearance order)
- `stems/NNN__Artist - Title/{vocals,drums,bass,other,instrumental}.<ext>` — Demucs stems
- `manifest.json` — set_id, track ids, paths, durations (source of truth for what *should* be there)

Re-running the same command is a **delta refresh** (rsync archive mode), only transferring files that changed on pi-storage. Safe and idempotent.

### 3. Refresh with orphan removal (`--prune`)

Use when pi-storage has diverged by *removal* — re-resolved track_audio_id, replaced audio with different codec, regenerated stems with a different subdir name. Without `--prune`, stale local files accumulate silently.

```bash
python labeling/pull_set_for_alignment.py <set_id> --prune
```

**Always preview first:**
```bash
python labeling/pull_set_for_alignment.py <set_id> --prune --dry-run
```

The prune is scoped — it only deletes audio-extension files inside `tracks/` and inside stem subdirs the current plan owns. It will NOT touch:

- User-renamed files matching `[NNNbpm KK]` or `[no-features]` tags (annotator territory — see below)
- Stem subdirs whose name doesn't match the current plan (user-renamed stem dirs and their contents)
- Ableton artifacts (`.asd`, `.als`), `manifest.json`, or anything at the folder root

This is gated behind `--prune` so a fat-finger can't wipe in-flight alignment work.

### 4. Inject BPM + key into M4A tags

After pulling, run the tagger so Ableton's clip browser shows tempo and Camelot key:

```bash
python labeling/tag_aligning_folder.py ~/aligning/<set_id>__<sanitized-title>
```

The tagger queries pi-storage's `track_audio_features` for BPM + Camelot key + feature comment, then writes them to each M4A's iTunes tags. Files without Essentia rows are skipped (and flagged with `[no-features]` rename later if the annotator does the rename pass).

## The annotator rename convention (DO NOT undo)

The human annotator renames files inline to expose tempo + key during alignment work:

- `tracks/030__Going Deeper - Little Big Adventure [126bpm 8B].m4a` — tempo + Camelot key
- `stems/001__Carmen Twillie - Circle Of Life [84bpm 6B]/` — same on stem subdirs
- `[no-features]` — flags tracks without Essentia rows on pi-storage

These renames are **one-sided, Mac-only mutations** — they never propagate back to pi-storage (canonical names there stay `{Artist} - {Title}.{ext}`). The prune logic explicitly recognizes the tag patterns and treats tagged files/subdirs as user territory.

**Consequence:** re-pulling a set will deposit *fresh un-tagged copies* of files the annotator previously renamed. That's expected — the annotator either re-runs the rename pass or ignores the duplicates. There's no automatic re-tag-on-refresh today.

## Workflow stages a typical alignment goes through

1. `--list-recent` → pick a set
2. `pull_set_for_alignment.py <set_id>` → initial pull
3. `tag_aligning_folder.py ~/aligning/<set>` → inject BPM + key into M4A tags
4. Drag into Ableton, do the alignment work
5. (sometime later) `pull_set_for_alignment.py <set_id> --prune --dry-run` then `--prune` → refresh if pi-storage state has changed
6. Write back GT via `python -m labeling.write_back_ground_truth --db ... --yaml ...`, then delete the folder

## Phase-cancel instrumental extraction

If the user wants to extract a clean instrumental for a track in the folder (used when Demucs stems aren't clean enough on their own), see `~/aligning/phase-cancel/`. The winner config is:

```
cancel.py adaptive --smooth 0.5 --fft 4096 --cap 4
```

`--mode simple` is broken on AI stems (negative gain). Don't use it.

## Anti-patterns

- ❌ Running `--prune` without `--dry-run` first when uncertain. Always preview.
- ❌ Manually renaming files back to canonical names ("cleaning up" the annotator's tags). The tags are intentional — Ableton shows them in the browser, dramatically speeding alignment.
- ❌ Treating the `~/aligning/<set>/` folder as a permanent archive. It's ephemeral; delete after write-back.
- ❌ Editing `manifest.json` by hand. It's regenerated on every pull.
- ❌ Writing alignment results back to local `data/db/music_database.db` — that's the stale dev copy. Write-back goes to pi-storage via `labeling.write_back_ground_truth`.
