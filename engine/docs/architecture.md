# Architecture — goal-step spine (canonical SoR)

Behavioral contract: [`../dj_engine_pseudocode.md`](../dj_engine_pseudocode.md).  
Engine map: [`../../README.md`](../../README.md).

This tree is the **system of record**. Python sensors propose; Rust decides, promotes, and acts.
Former lab peer: [`../../archive/python_kernel/`](../../archive/python_kernel/).

## Spine

```text
STORE   ArtifactStore / provenance SQLite     content-addressed bytes + lineage
PROPOSE python/sensors LFs                 vote or abstain → EvidenceEmission JSON
DECIDE  majority_vote_fit (dj_kernel::pws)    emissions → AxisBelief
PROMOTE gate_canaries                         held-out gold vs independent LF
ACT     cotraining (on promote only)          opposite-view PseudoLabels
```

| Port | Lives here |
|------|------------|
| Propose | `python/sensors/lfs/` → `write_vertical_slice` → `staging/vertical_lf_bundle.json` |
| Decide / Promote / Act | `crates/dj_kernel` (`pws`, `cotraining`) via `dj_migrate pws-fit` / `run-round` |

`EvidenceEmission` is the common Propose encoding today — not the only future ontology. Locators (`track_id`, paths) never mint `RecordingId`.

## Fixtures-only (place-in DB)

Canary smoke and debug do **not** need live `music_database.db`. Gold fixtures under `fixtures/gold/` plus JSON under `staging/` are the place-in database. Do not mutate the pi DB from this repo.

## Smoke path

```bash
make canary-smoke   # vertical LF bundle → run-round (fit → gate → promote/reject)
```

Held-out gold: `fixtures/gold/canary_stem_gold.json`. Canary source: `title_stem_heuristic_lf` (not `claimed_stem_lf`).

## Related

- Code walkthrough (`make canary-smoke`): [canary_walkthrough.md](canary_walkthrough.md)
- Open-vocab OD Propose plug: [vision_od.md](vision_od.md)
- Storage / FeatureBlob: [storage.md](storage.md)
- Agent ops: [../AGENTS.md](../AGENTS.md)
- Debug: [../../.vscode/README.md](../../.vscode/README.md) or parent workspace `.vscode/README.md`
