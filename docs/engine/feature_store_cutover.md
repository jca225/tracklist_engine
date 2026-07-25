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
