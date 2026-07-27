# Feature Store & Producer Registry (Brick 7)

> **Scope honesty:** this brick builds and *proves* the canonical feature-producer
> path on the provenance substrate — it does **not** migrate the live pipeline.
> pi-storage services, `analysis/persistence.py`, `track_fingerprints`, and the
> MERT blob tables are untouched; live cutover is a later, staged step. Law
> verdicts live in [law_audit.md](law_audit.md) (shipped pipeline) and
> `python -m core.provenance.laws` (substrate DBs) — not here.

## The canonical feature-store model

A feature is not a row in a bespoke table keyed by a mutable path — it is a
**content-addressed artifact** with full provenance:

```
audio artifact (sha256 = identity)
   │  run  ── ProcessSpec ──┬── ParameterSet   (canonical_hash of sorted-JSON values)
   ▼                        ├── EnvironmentSpec (os/arch/runtime + dependency-lock hash)
feature artifact (sha256)   └── model_refs      (e.g. "m-a-p/MERT-v1-330M", "layer=6")
   ▲ derivation edge (relation = transformation name)
```

- **ProcessSpec is deterministic**: its id is a sha256 over (name, version,
  code_commit, parameter canonical-hash, environment lock-hash, model_refs), so
  the identical process always has the identical id — recording it twice is a
  no-op (`INSERT OR IGNORE`).
- **Runs begun via `begin_run_from_spec`** stamp process/version/code/params
  *from* the spec and carry `process_spec_id`, which is what law 14 checks.
- **Models** get the same treatment (§8): checkpoint bytes → `MODEL_CHECKPOINT`
  artifact; the training set → a frozen `TrainingSnapshot` (sha over the sorted
  member artifact shas — law 13); the training process → a ProcessSpec; the
  whole thing → a `FittedModel` row pinned to one axis.

Substrate types/writers: `core/provenance/{types,versioning,store,repository}.py`.

## The registry — adding a new data type is two decorators

`core/provenance/registry.py` holds a module singleton `REGISTRY`. A producer
declares itself; `Registry.run` does *all* the provenance plumbing:

```python
from core.provenance import artifact_type, transformation, REGISTRY

@artifact_type(kind="my_feature", media_type="application/json", version=1)
class MyFeature: ...

@transformation(
    "analyze.my_feature", "1",
    inputs={"audio": "audio"},            # role -> artifact kind
    outputs={"features": "my_feature"},   # role -> artifact kind
    stage="analysis",
    params={"knob": 3},                   # becomes the ParameterSet
    model_refs=(),                        # models this producer stands on
)
def produce(inputs: Mapping[str, bytes]) -> Mapping[str, bytes]:
    return {"features": compute(inputs["audio"])}     # PURE bytes -> bytes

REGISTRY.run("analyze.my_feature", {"audio": audio_artifact},
             store=store, repo=repo, code_commit=sha)
```

`run` verifies declared input kinds, persists ParameterSet + live-captured
EnvironmentSpec + deterministic ProcessSpec, begins a versioned run, feeds the
inputs' bytes to the pure fn, stores each output content-addressed under its
declared kind, and records input/output/derivation edges (relation = the
transformation name). Failures are recorded (`fail`) and re-raised. The
declared kinds also give a queryable dependency graph (`REGISTRY.producers_of`,
`REGISTRY.graph()`). Proof that zero extra plumbing is needed for a brand-new
type: `tests/provenance/test_registry.py::test_new_dummy_type_is_content_addressed_and_fully_provenanced`.

## The two real producers (workspaces, not core)

`alignment/producers.py` (core stays stdlib-only; concrete
producers import aligner modules sideways):

- **`analyze.landmark_fp`** — fresh compute through `Registry.run`: audio bytes
  → the `landmark_fp` serialized fingerprint blob, registered as a
  `landmark_fingerprint` artifact derived from the audio artifact.
- **`migrate.mert_features`** — migration, not recompute: an existing cached
  MERT measure stack (`mert_store` `.cache/mert/<set>_mert.npz`) is
  re-serialized byte-stably and registered as a `mert_features` artifact
  derived from the audio artifact, with `model_refs=("m-a-p/MERT-v1-330M",
  "layer=6")`. This is the template for walking existing feature storage onto
  the canonical model without recomputation.
- **`register_trained_decoder`** — the real trained checkpoint on disk
  (`.cache/trajectory/decoder_slotsplit_seed0.pt`, the exact
  `trajectory/train.py:601` payload the law-14 audit row cites) becomes
  checkpoint-artifact + TrainingSnapshot (GT fixtures) + training ProcessSpec +
  `FittedModel`. Registration is flagged retrospective in the artifact
  metadata (the checkpoint predates the registry). Note: no MERT *identity
  head* is persisted anywhere in the repo (`train.py --train-mert` trains in
  memory), so the trajectory decoder is the persisted model that grounds
  laws 13/14.

CLI (pi access read-only; falls back to `--audio-file`):

```bash
venvs/audio/bin/python -m alignment.producers \
    --track 12m8zb3x --set-id 1fsnxchk
venvs/audio/bin/python -m core.provenance.laws \
    --db alignment/out/provenance/producers
```

## Migration stance

This brick = the **canonical path, proved on one real track + one real model**.
Deliberately *not* done here: cutting analysis services / `track_fingerprints`
/ MERT blob storage over to this store, backfilling the corpus, or touching
pi-storage. That cutover is a later staged step with its own coordination
(deploy + backfill + dual-read window); until then the shipped pipeline's
law_audit verdicts stand unchanged.
