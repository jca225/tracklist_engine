# AGENTS.md — how to work in `core/`

**This tree is canonical SoR** for the alignment engine (Rust +
`python/sensors`). The former pure-Python lab peer is archived at
[`../archive/python_kernel/`](../archive/python_kernel/) — do not treat it as SoR.

Short operating agreement. Product/architecture: [README.md](README.md),
[docs/architecture.md](docs/architecture.md),
[docs/canary_walkthrough.md](docs/canary_walkthrough.md) (intent → file → function),
repo-root [`../dj_engine_pseudocode.md`](../dj_engine_pseudocode.md),
[docs/storage.md](docs/storage.md).

**Fixtures are the place-in DB.** Canary smoke and debug use `fixtures/gold/` plus
`staging/*.json` — no live `music_database.db` required. Do not connect or mutate
the pi DB from this tree.

**TEMP learning comments:** many `TEMP:` / expanded `//!` notes in
`crates/dj_kernel` and `crates/dj_migrate` are temporary pedagogy. Grep `TEMP:`
to find or strip them once the architecture is familiar. Durable map:
[docs/canary_walkthrough.md](docs/canary_walkthrough.md).

## Quality gate (always)

```bash
make check              # rust + python agent self-correction loop
make migrate-dry-run    # transfer path without copying bytes
make vertical-lf        # Snorkel vertical slice
make cotrain-round      # co-training promote/reject + pseudo-labels
```

Do not claim done if `make check` is red.

## Near-term (do not skip)

1. ~~Independent canary~~ — `canary_stem_gold.json` + `title_stem_heuristic_lf`
2. Stem/MERT FeatureBlob lineage with ProcessSpec (not path identity)
3. Identity Propose that can disagree with tokenizer claims

## Branch hygiene (braid)

Use **braid** (`~/workspace/braid`) the same way as `tracklist_engine`:

```bash
make collide
make ready
make land-budget
make land-verify
```

Details: [docs/branch_hygiene.md](docs/branch_hygiene.md).

**Prerequisite:** git repo required for braid. `braid ready` ≠ green tests —
still run `make check`.

## Identity / data safety

- Never mint `RecordingId` from a path or legacy `track_id`.
- Never run `dj_migrate apply` without `--i-know-this-copies-bytes`.
- Do not mutate the live pi `music_database.db` from this repo (fixtures/staging only).
