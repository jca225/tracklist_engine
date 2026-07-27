# engine/ — provenance-first DJ alignment kernel (incubating)

Rust SoR (`dj_kernel` + `dj_migrate`) + Python sensors/LFs. **Incubating** —
fixtures-only canary today; does **not** replace the live Python DAG
(`alignment/`, `analysis/`, …). Sourced from `~/workspace/alignment_algorithm`.

Behavioral contract: [`../docs/engine/dj_engine_pseudocode.md`](../docs/engine/dj_engine_pseudocode.md)
(keep in sync with the workspace copy when either changes).

```bash
cd engine
make check
make canary-smoke
```

Spine: Propose (Python sensors) → Decide / Promote / Act (Rust). See
[`docs/architecture.md`](docs/architecture.md) and
[`docs/canary_walkthrough.md`](docs/canary_walkthrough.md).

## Layout

```
engine/
  crates/dj_kernel/   # opaque IDs, ArtifactStore, ProvenanceRepository, PWS
  crates/dj_migrate/  # inventory / map / stage / verify / gated apply / pws-fit
  python/sensors/     # Propose LFs / plugs
  schemas/            # JSON Schema at the Rust↔Python boundary
  fixtures/gold/      # BB11/BB12 dry-run samples (no live DB)
  staging/            # gitignored; migrate + vertical-lf output
```

## Explicit non-goals (this incubate)

- Replacing `alignment/` or cutting over the live DB
- Bulk copy of pi-storage audio
- Folding personalization / SoundCloud into Promote
