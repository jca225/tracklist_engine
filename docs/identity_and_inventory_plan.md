# Identity + inventory plan

Consolidated sequencing for disk↔DB orphans, identity axis cleanup, labeling
workflow fixes, and the work/recording layer. **Canonical vocabulary and pi
rollout commands** also live in the root [CLAUDE.md](../CLAUDE.md) ("Track
identity" section) — update both when the design changes.

Status tracked inline below.

## Phase 0 — Stop bleeding ✅

- `core.db.insert_audio_or_reap()` — unlink file on insert `Err`
- All ingest/replace paths use the wrapper
- `spotdl_adapter` rename failure reaps staged file

## Phase 1 — Identity axes ✅ (code); run SQL on pi-storage

- `core/identity.py` — `RecordingAxes`, `version__stem__variant` key
- DB columns: `stem`, `variant`, `version` (see `migrate_identity_axes.sql`)
- Stem values: **regular** (not "full" / "original" on stem axis)

## Phase 2 — Reconcile orphans ✅

- `scripts/reconcile_orphans.py` — dry-run / `--apply` / `--apply-promotions`

## Phase 3 — Replace + pull workflow ✅

- `replace_track_audio.py` — `--promote-reference`, `--purge-siblings`
- `pull_set_for_alignment.py` — `manual` platform wins; synthetic track ids in SQL

## Phase 4 — work + recording ✅ (schema + migration)

- Tables: `work`, `recording`, view `identity_mismatch`
- `scripts/migrate_phase4_recording.sql` on pi-storage after Phase 1 SQL
- `recording_id` on `track_audio` / `set_track_slots` (= legacy `track_id`)

## Phase 5 — GT write-back ✅ (v1)

- Table: `set_ground_truth`
- `labeling/write_back_ground_truth.py` — YAML → DB
- Algorithmic aligner still deferred (`workspaces/`)

## Three axes (vocabulary)

| Axis | Values | Source |
|------|--------|--------|
| **version** | original, remix, rework, altversion, edit, bootleg, mashup | Scrape / `track_metadata.version` |
| **stem** | regular, acappella, instrumental | Scrape qualifier / `track_audio.stem` |
| **variant** | regular, extended | Edit length / `track_audio.variant` |

Concatenated lookup key: `version__stem__variant` (e.g. `remix__acappella__extended`)
via `RecordingAxes.key()` in `core/identity.py`. Remix **artist** stays on
`recording.version_artist`, not in the key.

## pi-storage rollout (not run yet)

```bash
# backup, deploy code, then:
sqlite3 /mnt/storage/data/db/music_database.db < scripts/migrate_identity_axes.sql
sqlite3 /mnt/storage/data/db/music_database.db < scripts/migrate_phase4_recording.sql
venvs/web_crawler/bin/python -m tokenizer.materialize   # refresh slots + claims
venvs/audio/bin/python scripts/reconcile_orphans.py --dry-run
```

Ground-truth YAML: prefer `claimed_stem:` (legacy `version_tag:` still parses).

## Baby rule

One full file in `tracks/`; use `stems/vocals` or `stems/instrumental` when the
tracklist says acappella/instrumental.
