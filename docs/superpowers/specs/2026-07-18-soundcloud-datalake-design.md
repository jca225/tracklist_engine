# SoundCloud Data-Lake Substrate — Design (sub-project #1)

**Date:** 2026-07-18
**Status:** Approved design, pending implementation plan
**Author:** John Abrahams (+ Claude)

## 1. Purpose & scope

Build a **reusable SoundCloud data-lake ingestion primitive** (auth + any-user /
any-track public fetch) as shared project infrastructure, and use it to pull
John's own SoundCloud library (`https://soundcloud.com/user-327506308`,
`sc_user_id` to be resolved) **metadata-first** as a taste/graph research
substrate.

This is **sub-project #1** of a larger vision. The following are named consumers
of this layer and are **explicitly out of scope here** (each gets its own
spec → plan later):

- **Audio-seed corpus** — downloading library audio into `work`/`recording`/`track_audio` for alignment + mashups + DJ-set creation.
- **Discovery backbone** — using follows/likes/reposts to prioritize corpus-wide acquisition.
- **Research program** — the `lab/audience_prior/` "why we like music" work.

Off the alignment DAG. Off the canonical `music_database.db`. Mirrors how
`personalization/` already isolates its `taste_warehouse.db`.

### Decisions locked during brainstorming

| Axis | Decision |
|------|----------|
| Near-term purpose | Taste/graph research substrate **+** general SC data lake (metadata-first) |
| Auth | **Anonymous `client_id` scraping only** — reaches any user's *public* data. No private items, no reposts *stream* (public reposts on-profile still reachable). No OAuth. |
| Module home | **New top-level `soundcloud/`** — the general primitive; `personalization/` and `lab/` become consumers. |
| First payload | Profile `user-327506308` |
| Crawl scope | **Frontier primitive, depth-1 default** (approach B) — general crawler whose depth is a parameter; first run = depth-1 from John's profile |
| Storage | Dedicated off-canonical **`sc_lake.db`** (SQLite) + append-only raw JSONL, hosted **on pi-storage** under `/mnt/storage/data/soundcloud/` (durable + capacity), queried from Mac over SSH — mirrors how the corpus keeps canonical state off the Mac |
| Run location | Crawler runs **pi-side** (network + cheap CPU; long-running-capable), writing to the canonical pi-storage store. Mac is dev/query only. Systemd service still deferred. |

## 2. Architecture

New top-level module `soundcloud/`, layered bottom-up so each layer is testable
in isolation.

```
soundcloud/
  client.py        # auth + rate-limited transport (generalized from personalization)
  fetch.py         # typed per-endpoint primitives — the data-lake API
  store.py         # sc_lake.db schema + upsert
  crawl.py         # frontier driver (seed → queue → dedup → checkpoint)
  main.py          # CLI: sync-user / crawl / stats
  analysis/
    first_look.py  # thin validation analysis over the pulled library
  schema.sql       # sc_lake.db DDL
  deploy/          # pi-storage deploy notes / (future) systemd unit — deferred
```

**Physical data location (pi-storage, canonical):**

```
/mnt/storage/data/soundcloud/
  sc_lake.db                          # normalized nodes + edges + crawl checkpoints
  raw/{entity}/{id}/{fetched_at}.jsonl  # append-only API snapshots
```

The Python package `soundcloud/` is code only — it holds **no** local `data/`
dir. All persistence targets the pi-storage paths above (config-driven root, e.g.
`SC_LAKE_ROOT`, defaulting to `/mnt/storage/data/soundcloud`). Mac runs read via
SSH, matching the `pi-storage-query` pattern used across the corpus.

### 2.1 `client.py` — auth + transport

Generalized from `personalization/soundcloud_client.py`. Provides: `RateLimiter`,
`sc_client()`, `rl_get()`, `extract_client_id()`, `resolve(url)` (generalized from
`resolve_track`, no `kind=='track'` assertion — returns any resolved entity),
`next_url()`, and the `SC_API` / `SKIP_STATUS_CODES` constants.

**Migration handling:** `personalization/soundcloud_client.py` becomes a thin
re-export shim (`from soundcloud.client import *` plus the specific names it
currently exposes: `RateLimiter`, `sc_client`, `rl_get`, `extract_client_id`,
`resolve_track`, `next_url`). This keeps the running `tracklist-taste-scrape`
service on pi-worker working unchanged. `resolve_track` stays in the shim as a
track-asserting wrapper over the generalized `resolve`. Full migration of
`personalization` off the shim is **deferred** (future cleanup, not this
sub-project).

**What/how/depends:** turns a URL/endpoint into JSON, respecting SC rate limits;
callers pass a `client` + `RateLimiter` + `client_id`; depends only on `httpx`.

### 2.2 `fetch.py` — endpoint primitives (the data-lake API)

One typed function per API-v2 endpoint, each yielding raw JSON pages (paginated
via `next_url`). This is the reusable "any user / any track" surface every
consumer calls:

- `user(client, rl, cid, uid)` → user object
- `user_likes(...)`, `user_reposts(...)`, `user_playlists(...)`, `user_tracks(...)`, `user_followings(...)`, `user_followers(...)` → paged collections
- `track(client, rl, cid, tid)`, `playlist(client, rl, cid, pid)` → single objects

Functions are pure over their inputs and return raw dicts — **no DB writes, no
parsing decisions baked in** (parsing lives in `store.py`). A resource that
returns a `SKIP_STATUS_CODES` status yields nothing rather than aborting.

### 2.3 Raw lake — `{SC_LAKE_ROOT}/raw/{entity}/{id}/{fetched_at}.jsonl`

Append-only snapshots of every API response before parsing. Enables reprocessing
without refetching and gives an audit trail. `{fetched_at}` is an ISO timestamp
passed in by the caller (scripts stamp time at the edge; core stays clock-free
per repo style). Lives on pi-storage (`SC_LAKE_ROOT` default
`/mnt/storage/data/soundcloud`), never committed.

### 2.4 `store.py` + `schema.sql` — `sc_lake.db`

Dedicated SQLite at `{SC_LAKE_ROOT}/sc_lake.db` on pi-storage, off-canonical
(not `music_database.db`), off-DAG. Normalized nodes + edges, keyed on SC ids,
every row stamped `fetched_at` and `raw_ref` (path into the raw lake).

**Nodes:**
- `sc_users` (sc_user_id PK, permalink, username, followers_count, followings_count, verified, city, country, description, ...)
- `sc_tracks` (sc_track_id PK, title, sc_user_id, genre, tag_list, duration_ms, playback_count, likes_count, release/created dates, permalink, ...)
- `sc_playlists` (sc_playlist_id PK, title, sc_user_id, track_count, is_album, ...)

**Edges:**
- `sc_likes` (sc_user_id, sc_track_id, created_at?) — who likes what
- `sc_reposts` (sc_user_id, sc_track_id | sc_playlist_id, created_at?)
- `sc_follows` (follower_sc_user_id, followee_sc_user_id)
- `sc_playlist_tracks` (sc_playlist_id, sc_track_id, position)

**Identity seam (stub only):** `sc_recording_map (sc_track_id, recording_id,
method, confidence)` created **empty**, with a documented interface. Populated by
the future audio-seed / discovery-backbone consumers — not here.

`store.py` provides idempotent upserts (re-sync overwrites node attrs, edges are
insert-or-ignore) so re-runs are safe.

### 2.5 `crawl.py` — frontier driver

Resumable BFS frontier reusing `personalization`'s checkpoint idea (persist
queue + visited set + cursor in a `crawl_checkpoints` table in `sc_lake.db`). A
`CrawlPolicy` dataclass controls:

- `seed_user_ids: list[int]`
- `depth: int` (default **1**)
- `entity_types: frozenset` (which edges to expand: likes, reposts, playlists, uploads, followings, followers)
- rate (RPM) passed to `RateLimiter`

**Depth-1 semantics (first run):** fetch the seed user's own likes / reposts /
playlists / uploads and the *lists* of followings/followers (as node rows +
follow edges). Do **not** recurse into each neighbor's library. Bumping `depth`
later expands the taste graph without code changes.

### 2.6 `main.py` — CLI

- `sync-user <sc_user_id | profile_url>` — depth-1 convenience wrapper (resolves a profile URL to id via `client.resolve`).
- `crawl --seed <id> [--depth N] [--entities ...] [--rpm N]` — general entry.
- `stats` — coverage report over `sc_lake.db` (node/edge counts, per-entity totals).

### 2.7 `analysis/first_look.py` — research validation

One thin descriptive pass over the pulled library to prove the substrate is
usable and seed research questions: top liked artists, genre distribution over
likes, followings↔likes artist overlap, playlist sizes. Prints/writes a small
report. Heavy research stays in `lab/audience_prior/` (deferred).

## 3. Data flow

```
main sync-user 327506308
  → resolve profile URL → sc_user_id
  → crawl.py (CrawlPolicy depth=1)
      → fetch.py primitives (paged)
      → raw lake write (jsonl)
      → store.py parse + upsert → sc_lake.db
  → stats  (coverage)
  → analysis/first_look.py  (validation report)
```

## 4. Error handling

- **Transport errors:** `rl_get` retries with exponential backoff (existing behavior).
- **Per-resource skips:** `SKIP_STATUS_CODES` (401/403/404/429/500/502/503) skip that resource, don't abort the crawl.
- **client_id rotation:** if a request 401s mid-crawl, re-run `extract_client_id` once and retry; abort with a clear error if that fails too.
- **Resumability:** crawl checkpoints let an interrupted sync resume without re-fetching completed frontier nodes.
- **Core vs edge:** `client`/`fetch`/`store`/`crawl` core returns/raises values; `main.py` is the fail-fast edge (`sys.exit` on fatal). Matches repo "errors as values in core, fail-fast at the edge" style.

## 5. Testing

- `client.py`: unit-test `extract_client_id` regex + `next_url` param joining against fixture HTML/JS and URLs. `resolve` against a captured JSON fixture.
- `fetch.py`: mock `httpx` responses (respx or a fake client); assert pagination follows `next_url` and stops on empty/`SKIP_STATUS_CODES`.
- `store.py`: in-memory SQLite; assert upsert idempotency (double-insert of same node/edge is stable) and raw_ref/fetched_at stamping.
- `crawl.py`: depth-1 policy over a fixtured fake `fetch` — assert exactly the seed's own collections are pulled and neighbors are NOT recursed; assert checkpoint resume skips completed nodes.
- `first_look.py`: run over a small seeded `sc_lake.db`; assert the report fields exist.
- No live network in tests.

## 6. Explicitly deferred (YAGNI now)

Audio download; MERT embeddings for SC tracks; `work`/`recording` identity
mapping (seam is stubbed only); a **systemd unit** for the crawler (first runs are
invoked manually pi-side via `make deploy` + SSH — the always-on service comes
later); migrating `personalization` off the shim; any crawl past depth-1; OAuth /
private items /
reposts stream.

## 7. Open questions

None blocking. `sc_user_id` for `user-327506308` is resolved at runtime by
`client.resolve` on the profile URL.
