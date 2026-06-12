# Agent handoff — murph Space Miami taste-prior cohort (2026-06-10)

## Objective

Replicate the BB11/BB12 listener-cohort work in `personalization/` for
**it's murph @ Club Space Miami 2024** — i.e. collect the SoundCloud listener
cohort, enrich with likes + playlists, bot-score, cluster, and write findings.
`prior-mert` stays **deferred** (blocked on aligner pretrain, same as BB).

## Identifiers

| Thing | Value |
|---|---|
| 1001tracklists set_id | `pwgrrb1` |
| SC track id | `1801876378` |
| SC URL | https://soundcloud.com/its-murph-987074444/club-its-murph-live-space-miami-2024 |
| SC account | murph's own (confirmed — NOT Club Space's) |
| Upload stats at collect time | 90,556 plays · 2,941 likes · 72 reposts · 129 comments · 120.3 min |
| Mix registry | `personalization/config/mixes.yaml` (entry added) |
| Warehouse | `data/taste/taste_warehouse.db` (Mac-local, gitignored) |

## State at handoff

- **collect: DONE.** 2,845 unique listeners, 119 playhead-timestamped comments.
- **enrich (likes): IN PROGRESS** — ~644/2,845 users complete, ~270k likes.
  Checkpoint: `scrape_checkpoints` row (`mix_id='pwgrrb1', phase='enrich_likes'`).
- **enrich-playlists: NOT STARTED** (phase 2 of the driver loop).
- **score-bots / cluster / findings: NOT STARTED.**

Progress query:

```sql
SELECT json_array_length(checkpoint_json,'$.completed_sc_user_ids')
FROM scrape_checkpoints WHERE mix_id='pwgrrb1' AND phase='enrich_likes';
SELECT COUNT(*) FROM sc_likes WHERE mix_id='pwgrrb1';
```

## The connection-reset saga (why the code changed)

Batch-50 enrichment stalled at 600/2,845: SC's edge resets long-lived
keep-alive sockets, a reset anywhere in a ~150-request tick crashed the whole
tick, and the checkpoint only saved at tick end → zero forward progress.
Hardened in `soundcloud_client.py` + both enrich modules:

- `sc_client()` — no keep-alive (`max_keepalive_connections=0`), one-shot
  connections survive SC's edge behavior.
- `rl_get()` — 3 transport retries with exponential backoff.
- `TransportError` after retries **defers** the user (cursor kept in
  `in_progress`, resumed next tick) instead of crashing the tick.
- Checkpoint saved **per-user**, not per-tick.
- `SKIP_STATUS_CODES` (401/403/404/429/5xx) skip the user permanently.

"0 likes inserted" ticks right after a restart are **benign dedupe** — crashed
ticks had committed likes for users the checkpoint hadn't recorded yet.

## Restart the enrichment driver (background tasks die with the session)

Phase 1 (likes) until "enrich complete", then phase 2 (playlists) until the
checkpoint covers all listeners:

```bash
export TASTE_SC_RPM=30   # 45 default was fine too; resets were keepalive, not rate
while :; do
  out=$(venvs/audio/bin/python -m personalization.main enrich --mix pwgrrb1 --batch 10 2>&1)
  echo "$out" | tail -1
  echo "$out" | grep -q "enrich complete" && break
  echo "$out" | grep -q "likes inserted" || sleep 60
done
total=$(sqlite3 data/taste/taste_warehouse.db "SELECT COUNT(DISTINCT sc_user_id) FROM listeners WHERE mix_id='pwgrrb1'")
while :; do
  venvs/audio/bin/python -m personalization.main enrich-playlists --mix pwgrrb1 --batch 10 2>&1 | tail -1
  done_n=$(sqlite3 data/taste/taste_warehouse.db "SELECT json_array_length(checkpoint_json,'\$.completed_sc_user_ids') FROM scrape_checkpoints WHERE mix_id='pwgrrb1' AND phase='enrich_playlists'")
  echo "playlists: ${done_n:-0}/$total"
  [ "${done_n:-0}" -ge "$total" ] && break
done
```

Idempotent — checkpoints make re-runs safe. Check first whether a driver is
already running: `pgrep -fl "taste_prior.main enrich"`.

## Remaining steps (in order)

1. Finish likes + playlists enrichment (driver above).
2. `score-bots --mix pwgrrb1`.
3. `cluster --mix pwgrrb1` — **caveat:** k=12 / top-3k-track vocab was tuned on
   BB-scale cohorts; 2,845 users is BB11-scale (2,822 clustered) so likely OK,
   but sanity-check cluster sizes before trusting them.
4. `comment-heatmap` is **blocked**: no ground-truth labeling for `pwgrrb1`.
   Comments are already in `sc_mix_comments`; run the heatmap if/when the set
   gets labeled in Ableton.
5. **Cross-cohort overlap vs BB11/BB12** — the most interesting new analysis:
   shared `sc_user_id`s and shared liked tracks across `mix_id` values
   (`2nvzlh2k`, `1fsnxchk`, `pwgrrb1`). Different scene (house/tech vs mashup).
6. Append to `personalization/findings.md`: cohort size, bot rate,
   clusters, overlap.
7. Commit. **Deploy caveat:** once committed + `make deploy`, pi-worker's
   `loop --all-mixes` (`tracklist-taste-scrape.service`) will pick up
   `pwgrrb1` from `mixes.yaml`. Harmless (checkpoints) but it means pi-worker
   continues any unfinished enrichment — fine, just expected.

## User decisions on record

- Cohort wanted regardless of account ownership (resolved: murph's own).
- Enrichment runs **Mac one-shot**, not on pi-worker (until deploy, see above).
- `prior-mert` deferred.
