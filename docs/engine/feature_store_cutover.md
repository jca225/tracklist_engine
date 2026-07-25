# Feature-store cutover — ratified decisions

Companion to [feature_store_and_registry.md](feature_store_and_registry.md) (the
registry + producer API). That doc built and proved the canonical producer
**path**; this doc ratifies **how the live pipeline migrates onto it** and records
the topology the cutover uses. Migration is **additive and reversible first** —
no existing canonical table is mutated and no running service is repointed until
a forward-only slice is proven in production.

Pi-storage state at ratification (2026-07-25, read-only recon): 13T volume, 12T
free (8% used); canonical `music_database.db` = 33.9 GB; 18,878 track dirs under
`/mnt/storage/objects`; `/mnt/storage/provenance` did not exist.

## Decision 1 — physical topology

The canonical provenance store is **additive and separately namespaced**, never
inside the 33.9 GB `music_database.db` and never inside the location-keyed
`/mnt/storage/objects`:

- **Content-addressed object store:** `/mnt/storage/provenance/objects/` (sha256
  addressing, the existing `ArtifactStore` layout). Holds feature blobs, model
  checkpoints, and registered copies of source audio.
- **Central provenance DB:** a single `/mnt/storage/provenance/provenance.db`
  (SQLite) for the whole corpus — one DB, not per-set. Millions of
  observation/artifact rows are well within SQLite's range; the bulk lives in the
  object store, not the DB.
- **Why separate from `music_database.db`:** the canonical DB is the live
  scraper/analysis system of record; the provenance DB *references* its entities
  by id but is its own append-only store. Keeping them apart means the cutover
  cannot corrupt live state, and `rm -rf /mnt/storage/provenance` fully reverts.

## Decision 2 — backfill strategy: forward-first, staged backfill

1. **Prove a forward slice.** Register a small set of real tracks' audio +
   existing cached features (MERT, fingerprint) into the central store via the
   registry, deposited additively to pi. Run the law-checker on it. (This turn.)
2. **Forward-only for new analysis** (next): route *new* feature production
   through `REGISTRY.run` so anything computed from here on is canonical by
   construction. The legacy tables keep being written in parallel until parity is
   demonstrated — no big-bang.
3. **Staged corpus backfill** (later): register the existing 16k-track features in
   batches on-cluster. Never deletes or mutates the legacy tables; a separate,
   explicitly-confirmed step retires them only after parity + a promotion gate.

The invariant: **legacy storage is read, never destroyed, during migration.** Any
destructive retirement is its own gated, confirmed operation.

## Decision 3 — the MERT-head checkpoint gap

Brick 7 surfaced that `train.py --train-mert` trains the identity head **in
memory and never persists it**, so no MERT-head `FittedModel` can exist (Brick 7
registered the trajectory decoder instead, honestly labeled `axis=STRUCTURE`).

**Ratified fix:** add an additive checkpoint-save to the MERT-head training path
so the fitted head is written as a content-addressed `MODEL_CHECKPOINT` and
registered as a `FittedModel(axis=IDENTITY)` with its `TrainingSnapshot`. This is
purely additive (it does not change training behavior or the decode path) and is
realized when the head is next trained. Until then, law 13/14 grounding stands on
the real trajectory-decoder registration.

**Realized (Brick 9, 2026-07-25):** `train.py --save-head-checkpoint PATH`
(opt-in, default None — training behavior unchanged when absent) persists the
fitted head via `external/checkpoint.py`;
`workspaces/alignment_prototype/register_identity_head.py` registers it into the
Brick-7 producers store as a content-addressed `MODEL_CHECKPOINT` +
`TrainingSnapshot` (BB12 GT fixture + the trained-on MERT cache bytes) +
training `ProcessSpec` + `FittedModel(axis=IDENTITY)`. Laws 13/14 now PASS on
that store with a real identity model. Substrate-store scope only — the shipped
law_audit verdicts stand.

## What "touching infra" means here (safety contract)

- Only **additive** writes to pi under `/mnt/storage/provenance/**`.
- **No** writes to `music_database.db`, `/mnt/storage/objects`, or `/mnt/storage/stems`.
- **No** systemd/service changes, restarts, or `make deploy`.
- Every pi write is auditable and reversible by removing `/mnt/storage/provenance`.

## Forward-routing dual-write — status + enable-blockers (audit of 79271d2)

The dual-write (`analysis/persistence.py::_maybe_dual_write_provenance`, calling
`core.provenance.register_computed_feature`) is **landed flag-off** and adversarially
audited. Confirmed safe as landed: additive (one appended statement, Ok-path only,
after the legacy commit), flag-off is a literal no-op (does not even import
`core.provenance`), and every failure is swallowed.

**Fixed after the audit** (commit hardening): git-commit lookup is cached +
`timeout=5` + cwd-pinned (no hang); the provenance connection is closed (no fd
leak); `model_refs` is taken dynamically from `analyzer_versions` (no hardcoded
model/layer → no false provenance); object writes are atomic (temp + `os.replace`,
so a crash can't leave a truncated object at a content address); `store.connect`
uses WAL + a 30 s busy timeout.

**MUST be resolved before anyone sets `PROVENANCE_DUAL_WRITE=1` in production:**

1. ~~**Crash-window run poisoning.**~~ **RESOLVED** (PR #100, `core/provenance/reconcile.py`).
   `register_computed_feature` commits several times (param set / env / spec /
   run-begin / artifact / output / succeed); a kill mid-sequence (Vast **spot
   preemption is routine**) left a run stuck `status='running'` or an artifact with
   no output edge — permanent Law-1 offenders. `reconcile_store(conn, prune_orphans=…)`
   is the repair tool: it marks stale `running` runs `failed` (always safe, on by
   default) and reports/prunes orphan artifacts (`--prune`, opt-in — safe only
   because they were never a completed output). Run it at worker startup:
   `python -m core.provenance.reconcile --db <store-root> [--prune]`. Idempotent.
2. **Store-root location + worker ship-back.** The dual-write fires at the workers'
   **scratch-DB** persist (`analysis/vast_worker.py`, `scripts/mac_analyze_loop.py`),
   not the canonical commit. Enable **pi-local analysis first** (store root on local
   disk — WAL/sqlite over NFS/sshfs is unsupported and can hang). Decide the
   ship-back-to-central story before enabling on mac/Vast workers, or accept
   per-host local stores that never reach `/mnt/storage/provenance`.

**Follow-ups (not enable-blockers):** the dual-write passes `audio=None`, so a
feature links to its recording only by `track_audio_id` in metadata, not a
derivation edge (add audio linkage for full lineage); coverage is `persist_analysis`
only (not `persist_mert_measures` or the set-side writers). ~~no ordering test~~
**DONE** — `tests/test_audio_pipeline_analysis.py` now drives a real MERT feature
through the actual `persist_analysis` and asserts the `mert_features` artifact lands
(flag-on) / does not (flag-off), laws-clean + reconcile-clean.

## Turnkey enablement (the "make it real" flip)

With blocker #1 resolved, enabling the dual-write on the **Mac analysis worker**
(where MERT is computed — pi CPU runs no MERT, so the Vast/Mac GPU workers are the
only hosts that produce `mert_features`) is:

```sh
# 1. one-time: reconcile any crash residue in the target store (safe, idempotent)
venvs/audio/bin/python -m core.provenance.reconcile --db "$PROVENANCE_STORE_ROOT"
# 2. enable + run the loop (store root MUST be local disk, not NFS/sshfs)
export PROVENANCE_DUAL_WRITE=1
export PROVENANCE_STORE_ROOT="$HOME/Desktop/tracklist_engine/_mac_scratch/provenance_live"
caffeinate -i venvs/audio/bin/python scripts/mac_analyze_loop.py \
    --separator roformer --max-tracks <N>          # bound the batch
# 3. ship the additive store to pi (rsync into /mnt/storage/provenance/**, never
#    into music_database.db or /mnt/storage/objects) — REQUIRES pi online.
```

Worker-half validated live (2026-07-25, pi-offline): one real local BB13 track
(`Clean Bandit – Symphony (Ryos Remix)`) run through the shipped
`analyze_track → persist_analysis` with the flag on produced a real 152-measure
MERT stack that content-addressed into a durable store (`mert_features`
`ad58877a…`, 7.4 MB object), laws-clean + reconcile-clean. Permanent regression
guard: the `persist_analysis` E2E tests in `tests/test_audio_pipeline_analysis.py`.
