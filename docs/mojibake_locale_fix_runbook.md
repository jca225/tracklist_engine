# Runbook — pi UTF-8 locale fix + `track_audio.path` mojibake repair (issue #74)

**Status:** staged. The code-side defenses (below, "Already shipped") are merged
on branch `fix/mojibake-path-issue74`. The two **pi-live** steps here are
deliberately NOT executed by an agent — they mutate canonical pi state and must
ride a **coordinated deploy** (bundle with the Phase B `make deploy`, not a
second uncoordinated pi touch).

## Why the order is load-bearing

pi-storage's non-interactive locale is `LANG=en_US` → CPython
`sys.getfilesystemencoding()='iso8859-1'`. A pi process moving a non-ASCII
filename across the disk↔DB boundary under that locale double-encodes it: the
file on disk stays correct UTF-8, but the stored `track_audio.path` becomes
mojibake (e.g. en-dash `U+2013` → `U+00E2 U+0080 U+0093`, ``â€"``). It resolves
on pi *by accident* (iso8859-1 re-encodes the bad string back to the real bytes)
but a UTF-8 client (Mac `rsync` in `pull_set_for_alignment`) asks for a path that
does not exist → GT pulls fail.

**Repairing the DB rows BEFORE fixing the locale breaks pi-local access** — a
repaired path holds `U+2013`, which iso8859-1 cannot encode to the filesystem.
So: **locale first, repair second.** `scripts/repair_mojibake_paths.py` enforces
this — it refuses `--apply` unless the running process is under a UTF-8 FS
encoding.

## Already shipped (code, this branch — no pi touch)

- `core/mojibake.py` — `is_double_encoded_utf8` / `repair_double_encoded_utf8`,
  the single tested primitive (replaces the scattered `.encode('latin-1')
  .decode('utf-8')` band-aids).
- `core/db.py::insert_audio` — **normalizes** a mojibake path to correct UTF-8
  before storing (normalize, not reject: `insert_audio_or_reap` would unlink the
  real file otherwise). Recurrence guard for the write path.
- `scripts/corpus_integrity.py` — new ERROR-severity `mojibake_path` scan
  (`make check-corpus`), so the class is caught corpus-wide going forward.
- `scripts/repair_mojibake_paths.py` — the dry-run-default repair tool below.

## Step 1 — Fix the pi locale to UTF-8 (the linchpin; prevents recurrence)

Audit **every** context that runs the pipeline, not just the login shell (the
corrupting writer is a *service*):

1. Ensure a UTF-8 locale exists: `C.UTF-8` is always present on Debian/Ubuntu
   (no `locale-gen` needed); otherwise `sudo locale-gen en_US.UTF-8`.
2. Systemd units that run any ingest/analysis Python (scraper, jobqueue, retry
   drain, any download/acquire service). For each, add a drop-in:
   ```
   sudo systemctl edit <unit>
   # [Service]
   # Environment=LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONUTF8=1
   sudo systemctl daemon-reload && sudo systemctl restart <unit>
   ```
   Verify: `systemctl show <unit> -p Environment` shows the UTF-8 values, and
   `sudo cat /proc/$(pgrep -f <unit>)/environ | tr '\0' '\n' | grep -E 'LANG|PYTHONUTF8'`.
3. Login / cron: append `export LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONUTF8=1` to the
   pipeline user's shell profile and any crontab preamble.
4. Belt-and-suspenders: `PYTHONUTF8=1` (PEP 540) forces UTF-8 FS/IO encoding
   regardless of `LANG` — set it everywhere in #2/#3.

**Acceptance for step 1:** on pi,
`python3 -c "import sys;print(sys.getfilesystemencoding())"` prints a UTF-8
value in *service* context (not just interactive).

## Step 2 — Repair the mojibake rows (ONLY after step 1)

```bash
# dry-run first (safe; shows old→new + whether the repaired path exists on disk)
ssh pi-storage 'cd /path/to/tracklist_engine && \
  LANG=C.UTF-8 PYTHONUTF8=1 venvs/audio/bin/python \
  scripts/repair_mojibake_paths.py --db /mnt/storage/data/db/music_database.db'

# apply (refuses unless FS encoding is UTF-8; skips any row whose repaired path
# is not present on disk rather than pointing the DB at a missing file)
ssh pi-storage 'cd /path/to/tracklist_engine && \
  LANG=C.UTF-8 PYTHONUTF8=1 venvs/audio/bin/python \
  scripts/repair_mojibake_paths.py --db /mnt/storage/data/db/music_database.db --apply'
```

Back up the DB first (`cp music_database.db music_database.db.bak_<ts>`). The
tool never touches files — disk is already correct. It audit-logs each change to
stdout (this is an *encoding* repair, not an identity correction, so it is
intentionally NOT written to `track_audio_correction`).

## Step 3 — Verify

- `make check-corpus` → `mojibake_path` count is **0**.
- Each previously-broken row now rsyncs from a UTF-8 client (Mac
  `pull_set_for_alignment` of the affected sets succeeds).
- The 2 BB12 slots that abstained during the Crush-exit re-measure resolve.

## Affected rows at time of writing (2026-07-22)

6 rows, mostly acappella stem-candidates:
`track_audio_id` 20855, 20870, 20895, 20896, 23114, 23669. (Re-scan with the
dry-run before applying — the set may have grown.)
