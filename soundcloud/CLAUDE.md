# soundcloud — SoundCloud data-lake substrate

Reusable anonymous-`client_id` fetch layer + normalized graph store. **Off the
alignment DAG, off `music_database.db`.** The general SoundCloud ingestion
primitive that `personalization/` and `lab/` consume; NOT a specific consumer.

- **Auth:** anon `client_id` only (`client.py`, generalized from
  `personalization/soundcloud_client.py`, which is now a re-export shim). Public
  data of any user; no OAuth, no `/me`, no private items/reposts stream.
- **Storage (pi-storage, canonical):** `SC_LAKE_ROOT` (default
  `/mnt/storage/data/soundcloud`) → `sc_lake.db` + `raw/{entity}/{id}/*.jsonl`.
  The package is code-only; Mac queries over SSH.
- **Layers:** `config` → `records` → `schema.sql`+`store` → `client` → `fetch`
  → `rawlake` → `crawl` (depth-1 frontier) → `main` (CLI) → `analysis/first_look`.

## Commands

```bash
venvs/audio/bin/python -m soundcloud.main sync-user https://soundcloud.com/user-327506308
venvs/audio/bin/python -m soundcloud.main crawl --seed 327506308 --depth 1
venvs/audio/bin/python -m soundcloud.main stats
venvs/audio/bin/python -m soundcloud.analysis.first_look
```

## Deferred consumers (own specs later)

Audio-seed corpus (populates `sc_recording_map`), discovery backbone, MERT
embeddings, systemd crawler service, migrating personalization off the shim,
crawl depth > 1, OAuth.
