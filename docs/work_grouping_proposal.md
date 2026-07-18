# Work-grouping: link sibling recordings under one `work` (proposal)

**Status:** **APPLIED to canonical DB 2026-07-18** (1922 clusters, 3150 recordings
re-parented; works 18810→15662). Revert map:
`pi:/mnt/storage/data/db/recording_work_id_pre_grouping.csv` (single-column change).
The scorer follow-up (§Applying step 5) is also done on this branch.
**Tool:** [scripts/propose_work_grouping.py](../scripts/propose_work_grouping.py)
**Origin:** [eda/alignment/failure_analysis/IDENTITY_MISS_DECOMPOSITION.md](../eda/alignment/failure_analysis/IDENTITY_MISS_DECOMPOSITION.md)
(the `work` layer being unpopulated surfaced there).

## The gap

Every recording is currently its own singleton work — `work_id == recording_id`
for all **18,810**, and **0** works group more than one recording. Nothing links
version/stem/variant siblings: "Roses" ↔ "Roses (Acappella)", "Emily" ↔ "Emily
(Remix)", etc. This is a data-model hole (the `work`/`recording` two-layer design
is unrealised) and it makes the aligner scorer count "right song, wrong version" as
a full identity miss instead of a near-miss.

## Proposal (conservative, exact-key)

Two recordings are proposed as siblings iff their `work` rows share an EXACT
normalized `(title, artist-set)` key, where `work.title` / `work.artists_json` are
already the clean song title + artist (version qualifiers live on the
`version`/`stem`/`variant` axes, not the title). Exact-key matching **under-merges**
(a different title spelling stays split) — the safe failure mode — rather than
fusing two genuinely different songs.

## Dry-run result (canonical DB, read-only, 2026-07-18)

```
n_recordings:            18810
n_clusters_merging:       1922
n_recordings_affected:    5072   (27% of the corpus has an unlinked sibling)
n_works_after:           15660   (18810 -> 15660; 3150 works absorbed)
cluster-size distribution: {2:1280, 3:368, 4:133, 5:67, 6:33, 7:15, 8:11, 9:8, 10:2, 11:3, 12:1, 13:1}
```

Review sample (every cluster is a clean sibling group — same song+artist, differing
by version; no false merges observed):

```
"dna" [kendrick lamar]              remix, original
"rockstar" [post malone]            altversion, original, remix
"there for you" [garrix|troye]      original + 8 remixes
"electricity" [silk city]           original + 3 remixes
"symphonica" [nicky romero]         original + 2 (distinct) remixes
```

Two same-`version` recordings in a cluster (e.g. two "remix" rows) are typically
**distinct remixes** (different `version_artist`) — correctly the same *work*, still
distinct *recordings*. Grouping them is right.

## Applying (GATED — do not run without review)

Canonical mutation, so treat like the identity reconcile ops (dry-run first, no
blind `--apply`):

1. Regenerate the proposal on a fresh DB read:
   `python scripts/propose_work_grouping.py --db "file:/mnt/storage/data/db/music_database.db?immutable=1" --out /tmp/wg.json --emit-apply-sql > /tmp/wg_apply.sql`
2. **Review** `/tmp/wg_apply.sql` (it only *prints* the `UPDATE recording SET work_id=…`
   statements; nothing runs). Spot-check a sample of clusters against the source rows.
3. Back up `music_database.db`, then apply the reviewed SQL on pi-storage.
4. Orphaned `work` rows (the absorbed singletons) can be left (harmless) or GC'd in a
   follow-up; FK cascade behaviour must be checked before deleting.
5. **Scorer follow-up — DONE (this branch):** `score_timeline_vs_gt.py` now reads
   `--work-map` (default `labeling/fixtures/work_map.json`, a snapshot exported from
   canonical) and prints a second `identity (version-aware)` number crediting a strict
   miss that shares a `work` with an overlapping GT recording. Strict identity is
   unchanged (additive). Effect is small by design — **BB11 +1, BB12 +2** rescued (e.g.
   Chainsmokers "Roses" ↔ "Roses (Acappella)"), matching the finding that version-
   linking touches only ~2 identity misses. Snapshot drifts as the DB changes;
   regenerate via `ssh pi-storage 'sqlite3 <DB> "SELECT json_group_object(recording_id, work_id) FROM recording;"'`.

## Caveats

- Exact-key under-merges: siblings with differently-spelled titles stay split. A
  fuzzy pass (token-sort ratio) could recover more but risks false merges — defer
  until the conservative pass is applied and reviewed.
- `version_artist` is not yet used in the key; two different remixes of one song
  correctly land in the same work, which is intended.
