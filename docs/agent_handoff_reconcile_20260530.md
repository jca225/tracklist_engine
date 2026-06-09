# Agent handoff — orphan reconcile (2026-05-30)

**From:** reconcile / ingest session (Cursor agent on Mac)  
**To:** any parallel agent touching `tokenizer/`, `identity_axes`, `analysis/`, or `ingest/`  
**Canonical state:** pi-storage @ `d459583` after `git pull`; DB + disk already mutated.

> Durable ops + three-axis vocabulary: root [CLAUDE.md](../CLAUDE.md) ("Track identity").
> This file is **session-specific** reconcile state — archive/delete when follow-ups are done.

> **Do not re-run** `reconcile_orphans.py --apply` unless you have re-dry-run and understand
> the delta. Pass-1 manual deletes + pass-2 apply are **done**.

---

## Completed on pi-storage

| Action | Count / outcome |
|--------|-----------------|
| `scripts/reconcile_pass1_manual.sh --execute` | 25 manual deletes (dup clusters + safe coexist dupes) |
| `git pull` | `925a467` → `d459583` (`insert_audio_or_reap` live) |
| `reconcile_orphans.py --apply` | **52 DELETE**, **42 REGISTER** |
| Left in REVIEW | **5** rows only (see below) |

**Notable REGISTER:** `gnsjmf` → `track_audio_id=20818`, `youtube_music` / `Jb6gcoR266U` (~430s full *Around The World*).  
**Preserved:** `9hp84x` acappella ref (`taid=758`, ~107s) — do not promote/replace without explicit intent.

**Correction ledger:** 42 rows in `track_audio_correction` with `source='reconcile_orphans'`.

**Scripts on pi:** `scripts/reconcile_pass1_manual.sh` (scp'd), `scripts/reconcile_orphans.py` (from git).

---

## For the other agent — optional follow-ups

### 1. Promotions / manual (5 REVIEW rows only)

These are the **only** remaining disk orphans. None were auto-deleted or registered.

| track_id | full_name (abbrev) | Why REVIEW | Suggested action |
|----------|-------------------|------------|------------------|
| `21wfxm45` | Mathew Jonson - Marionette (Stephan Bodzin Remix) | orphan 683s vs ref 346s; not acappella-shaped | **DEFER** — listen or `replace_track_audio` |
| `29c2lftf` | MRAK - One | orphan ffprobe failed; ref 423s ytm | **DEFER** — check file integrity |
| `5uzdn35` | Marvin Gaye - Got To Give It Up (Acappella) | orphan 713s vs ref 360s; acappella smell but ref > 120s | **Manual** — won't auto-PROMOTE (`PROMOTE_MAX_REF_S=120`). Use `replace-track-audio` skill or listen; do not blind `--apply-promotions` |
| `x25swf` | Synergy - Hello Strings | orphan 697s vs ref 241s | **DEFER** |
| `xf5gs8x` | Binary Finary - 1998 (Jose De Mara Remix) | orphan 479s vs ref 189s | **DEFER** |

TSV on pi: `/tmp/reconcile_pass3.tsv` (5 lines).

```bash
# Re-check anytime (dry-run only)
ssh pi-storage 'cd ~/tracklist_engine && \
  venvs/audio/bin/python scripts/reconcile_orphans.py --review-tsv /tmp/reconcile_pass3.tsv'
```

**Do not** run `--apply-promotions` without resolving `5uzdn35` first.

---

### 2. Re-download queue (folders emptied — no `track_audio`)

Acquire via `scripts/redownload_via_ytmusic.py` (or ingest) when convenient:

| track_id | full_name |
|----------|-----------|
| `4dtxu75` | Green Velvet - Flash (Nicky Romero Remix) |
| `19bg4m9p` | Daft Punk - Around The World (Dimitri Vegas & Like Mike Remix) |
| `1fw35fxp` | GTA & Astronomar - Heavy Thunder |
| `1mlz2hg5` | KURA - Thunder |
| `1uz8820p` | deadmau5 - Strobe (Lane 8 Remix) |
| `1wws7mtf` | Hardwell & Armin van Buuren - Boundaries (AMF 2017…) |
| `hm0pvnp` | Armin van Buuren - A State Of Trance Year Mix 2021 |

Pass-1 removed wrong 75m/650m search hits from these folders. **No analysis rows**
to cascade-delete for most (never registered). **`1fw35fxp` is SoundCloud-only**
in scrape links — acquire via `ingest.main` or `replace_track_audio` with an
`api.soundcloud.com/tracks/<id>` URL, not YT Music rescue.

---

### 3. Commits / merge hygiene (Mac repo)

**Ready to commit (reconcile session, low conflict risk):**

- `scripts/reconcile_pass1_manual.sh` — new, standalone
- `scripts/reconcile_orphans.py` — **2-line unicode fix** (em-dash → ASCII; fixes `UnicodeEncodeError` over SSH latin-1 locale on pi). Diff:

```diff
- (dry-run — nothing changed...
+ (dry-run - nothing changed...
- (promotions skipped — pass...
+ (promotions skipped - pass...
```

**Coordinate before editing:**

- `scripts/reconcile_orphans.py` — if you are refactoring reconcile, **merge unicode fix first** or preserve ASCII-only punctuation in print paths.
- Do **not** revert pi DB/disk changes without operator sign-off.

**Unrelated WIP on Mac (other agent — do not assume deployed to pi):**

- UVR chain, `tokenizer/identity_axes`, `ingest/search_query.py`, `docs/identity_and_inventory_plan.md`, etc.

---

## Conflict avoidance

| Area | Status |
|------|--------|
| pi `~/tracklist_engine` | At `d459583`; `insert_audio_or_reap` in `core/db.py` |
| `/mnt/storage/objects/` | Pass-1 + apply already ran; 5 REVIEW orphans remain on disk |
| `track_metadata` / materialize | Unaffected by reconcile; safe to continue identity-axis work |
| Analysis loop | 42 new `track_audio` rows may enqueue only if tracks are in loop's set-id filter (BB caveat) |

---

## Quick verification commands

```bash
# REVIEW count should be 5
ssh pi-storage 'cd ~/tracklist_engine && venvs/audio/bin/python scripts/reconcile_orphans.py 2>/dev/null | tail -3'

# gnsjmf registered
ssh pi-storage 'sqlite3 /mnt/storage/data/db/music_database.db \
  "SELECT * FROM track_audio WHERE track_id=\"gnsjmf\""'
```

---

*Delete or archive this file after follow-ups are done.*
