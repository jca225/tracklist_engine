# Agent handoff — identity axes rollout (2026-05-30)

**From:** identity / ingest / docs session (Cursor agent on Mac)  
**To:** any parallel agent touching ingest, tokenizer, labeling, analysis, or pi-storage ops  
**Canonical code:** `origin/main` @ **`f07b9aa`** (pi-storage and pi-worker should match after `git pull`)

> **Durable vocabulary:** root [CLAUDE.md](../CLAUDE.md) section **"Track identity (three axes)"**.  
> **Prior reconcile session:** [agent_handoff_reconcile_20260530.md](agent_handoff_reconcile_20260530.md) — **do not re-run** `reconcile_orphans.py --apply`.

---

## What shipped (commits since `d459583`)

| Commit | Summary |
|--------|---------|
| `1d12f0f` | Reconcile handoff doc + `reconcile_pass1_manual.sh` |
| `3b1a69f` | **Identity axes:** `core/identity.py`, DB renames, migrations, tokenizer/materialize, ingest/analysis/labeling renames, tests, `write_back_ground_truth.py` |
| `0104034` | Optional UVR stem backend (Demucs still default) |
| `38188d4` / `d454ce0` | Migration fixes (`set_track_slots` CREATE, fingerprint column) |
| `f07b9aa` | **materialize perf:** skip full `schema.sql` replay on canonical DB |

---

## Three axes (do not conflate)

| Axis | Values | DB |
|------|--------|-----|
| **version** | original, remix, rework, altversion, … | `track_metadata.version` |
| **stem** | **regular**, acappella, instrumental | `track_audio.stem` (was `variant_tag`; `original`/`full` → **regular**) |
| **variant** | regular, extended | `track_audio.variant` (was `edit_tag`) |

**Key:** `version__stem__variant` via `RecordingAxes.key()` in `core/identity.py`.  
**Demucs** `track_stems.stem_name` (vocals, drums, …) is unrelated to identity `stem`.

**Baby rule:** one full file in `~/aligning/.../tracks/`; acappella/instrumental → use Demucs `stems/vocals` or `stems/instrumental`.

---

## Pi-storage state (as of handoff)

### Done

| Item | Detail |
|------|--------|
| `git pull` | Repo @ `f07b9aa` on `~/tracklist_engine` |
| DB backup | `/mnt/storage/data/db/music_database.db.bak-20260530-identity` (~8.1G) |
| `migrate_identity_axes.sql` | Applied — `stem` / `variant` / `version` columns |
| `migrate_phase4_recording.sql` | Applied — `work`, `recording`, `set_track_slots` table, `set_ground_truth`, `identity_mismatch` view |
| `recording` backfill | ~17,838 rows (1:1 with legacy `track_id` for now) |
| Reconcile dry-run | **5 REVIEW orphans only** — same as reconcile handoff |

### In progress — `tokenizer.materialize`

**Last checked:** 2026-05-30 ~10:28 pi-storage local — **10%** (150k/1.4M rows), PID **59888** alive, ~151k `set_track_slots`, `track_metadata` still 0 (written at end). ETA **~5–6 h** from 09:53 start if rate holds.

```bash
# Running in background (verify PID still alive):
pgrep -af "python -m tokenizer.materialize"
tail -f /tmp/materialize.log
```

- **Log:** `/tmp/materialize.log`
- **Command:** `venvs/web_crawler/bin/python -m tokenizer.materialize --db /mnt/storage/data/db/music_database.db`
- **Workload:** 1,401,879 `dj_set_rows` (BeautifulSoup per row; **1–2+ hours** on Pi)
- **Progress logs:** every 5% in `/tmp/materialize.log`
- **`track_metadata`:** written only at **end** (in-memory aggregate until DONE)
- **`set_track_slots`:** flushed every 1,000 slots — check `SELECT COUNT(*) FROM set_track_slots` while running

**Do not start a second materialize** (DB lock). If dead, restart single `nohup` after confirming no other holder:

```bash
ssh pi-storage 'cd ~/tracklist_engine && nohup venvs/web_crawler/bin/python -m tokenizer.materialize \
  --db /mnt/storage/data/db/music_database.db > /tmp/materialize.log 2>&1 &'
```

**When DONE**, log ends with `DONE — {'track_metadata': N, 'set_track_slots': M, ...}` and `track_metadata` row count > 0.

---

## Do NOT do without operator sign-off

- `reconcile_orphans.py --apply` (already run 2026-05-30; 52 deletes, 42 registers)
- `reconcile_orphans.py --apply-promotions` (blocked on `5uzdn35` — ref 360s > 120s cap)
- Re-run migrations on canonical DB (already applied)
- Revert pi DB/disk from backup without explicit request

---

## Manual follow-ups (after materialize completes)

### 1. Five REVIEW disk orphans

| track_id | Notes |
|----------|--------|
| `21wfxm45` | orphan 683s vs ref 346s — listen / `replace_track_audio` |
| `29c2lftf` | orphan ffprobe failed — check file |
| `5uzdn35` | acappella-shaped, ref 360s — **manual only**, not auto-promote |
| `x25swf` | orphan 697s vs ref 241s |
| `xf5gs8x` | orphan 479s vs ref 189s |

```bash
ssh pi-storage 'cd ~/tracklist_engine && venvs/audio/bin/python scripts/reconcile_orphans.py'
```

### 2. Seven tracks need re-download (empty folders)

`4dtxu75`, `19bg4m9p`, `1fw35fxp`, `1mlz2hg5`, `1uz8820p`, `1wws7mtf`, `hm0pvnp` — use `scripts/redownload_via_ytmusic.py` when yt-dlp/cookies healthy.

### 3. Mac labeling / analysis

- Stem re-source (acappella/instrumental quality): [stem_discovery_playbook.md](stem_discovery_playbook.md)
- Pull sets: `labeling/pull_set_for_alignment.py` — manifest now has `version`, `stem`, `variant`, `axes_key`
- GT YAML: use **`claimed_stem`** (legacy `version_tag:` still parses)
- Write-back: `python -m labeling.write_back_ground_truth --db ... --yaml ...`
- Mac analyze loop for new/refreshed `track_audio` rows (BB job filter caveat)
- **Preserve** `9hp84x` acappella ref (`taid=758`) unless intentional

---

## Verification commands

```bash
# Code version on pi
ssh pi-storage 'cd ~/tracklist_engine && git rev-parse --short HEAD'   # expect f07b9aa+

# Identity columns exist
ssh pi-storage 'sqlite3 /mnt/storage/data/db/music_database.db \
  "PRAGMA table_info(track_audio);" | grep -E "stem|variant|recording"'

# Materialize progress
ssh pi-storage 'sqlite3 /mnt/storage/data/db/music_database.db \
  "SELECT COUNT(*) FROM set_track_slots; SELECT COUNT(*) FROM track_metadata;"'

# Reconcile should show 5 REVIEW
ssh pi-storage 'cd ~/tracklist_engine && venvs/audio/bin/python scripts/reconcile_orphans.py 2>&1 | tail -8'

# Sample axes on audio
ssh pi-storage 'sqlite3 /mnt/storage/data/db/music_database.db \
  "SELECT track_id, stem, variant FROM track_audio LIMIT 5;"'
```

---

## Mac repo (local)

- **Pushed to `origin/main`** — no pending identity rollout commits from this session unless you have new local edits.
- **Untracked:** `.claude/skills/` (not part of rollout).
- **Tests (local):** `venvs/audio/bin/python -m pytest tests/test_identity_axes.py tests/test_recording_axes.py tests/test_tokenizer_lossless.py tests/test_audio_pipeline_db.py -q`

---

## Conflict avoidance

| Area | Guidance |
|------|----------|
| `tokenizer/materialize.py` | Pi may be running; don’t change flush/DDL semantics mid-run without coordinating restart |
| `scripts/reconcile_orphans.py` | ASCII-only prints for pi SSH; identity column names (`stem`, not `variant_tag`) |
| `web_crawler/database/schema.sql` | Source of truth for **fresh** installs; canonical DB already migrated via SQL scripts |
| UVR / `requirements-audio.txt` | Deployed in `0104034`; default separator still **demucs** |

---

## Key paths

| Path | Role |
|------|------|
| `core/identity.py` | Axes + `RecordingAxes.key()` |
| `tokenizer/identity_axes.py` | Scrape → claims |
| `scripts/migrate_identity_axes.sql` | Column renames (already run on pi) |
| `scripts/migrate_phase4_recording.sql` | work/recording/slots/GT (already run on pi) |
| `docs/identity_and_inventory_plan.md` | Phase checklist |
| `docs/agent_handoff_reconcile_20260530.md` | Orphan reconcile (completed) |

---

*Archive or delete this file after materialize + manual follow-ups are done.*
