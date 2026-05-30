---
name: refactor-safety
description: Safe module renames, directory splits, and bulk path updates across the Tracklist Engine repo. Inventory stale references, fix Path(__file__).parents[N], update deploy entrypoints, run make check, and write handoff docs when pi-storage ops are involved. Use when renaming or moving packages (e.g. audio_pipeline split), refactoring folder layout, updating adapter paths, or doing bulk import/path migrations. Triggers on "rename module", "move folder", "split package", "refactor X to Y", "bulk path update", "update imports after rename".
---

# Refactor Safety Checklist

Use this skill for **mechanical** renames and directory moves. Never combine mechanical rename + behavior change + pi-storage rollout in one agent session — split into separate phases.

## Phase split rule

| Phase | Scope | Stop when |
|-------|-------|-----------|
| 1. Inventory + mechanical rename | paths, imports, `parents[N]`, Makefile | `make check` green, grep clean |
| 2. Behavior change | logic, schema semantics | tests pass |
| 3. Cluster rollout | deploy, migrations, long jobs | handoff doc written, operator sign-off |

---

## 1. Inventory

Search for the old name everywhere:

```bash
rg 'OLD_NAME' --glob '*.py' --glob '*.md' --glob 'Makefile'
```

Common stale names after past refactors:

| Old | New |
|-----|-----|
| `audio_pipeline` | `ingest` + `analysis` |
| `data_analysis` | `eda` |
| `variant_tag` | `stem` (identity axis) |
| `edit_tag` | `variant` (identity axis) |

Also check:

- `python -m …` entrypoints in module `CLAUDE.md` deploy caveats
- systemd unit names documented in [ingest/CLAUDE.md](ingest/CLAUDE.md) and [analysis/CLAUDE.md](analysis/CLAUDE.md)
- [Makefile](Makefile) SSH deploy paths (usually unchanged)

---

## 2. Mechanical rename

### Path depth after moves

| File location | Repo root |
|---------------|-----------|
| `ingest/adapters/*.py`, `analysis/adapters/*.py` | `Path(__file__).resolve().parents[2]` |
| `scripts/*.py` | `parents[1]` |
| `tests/*.py`, top-level modules | `parents[1]` from repo root file |

**Never leave `parents[3]` in adapter modules** — that was the pre-split layout bug.

### Imports

Update all `import` / `from` lines. Keep intentional legacy strings:

- `audio_pipeline_v1` — DB `source` label in [analysis/persistence.py](analysis/persistence.py) (do not rename)

### Deploy entrypoints

If module entrypoints changed, update deploy caveats in root [CLAUDE.md](CLAUDE.md) and affected module guides. Example:

- `python -m audio_pipeline.main` → `python -m ingest.main`
- `python -m audio_pipeline.vast_worker` → `python -m analysis.vast_worker`

---

## 3. Verify

```bash
make check
```

This runs:

1. [scripts/guardrails.py](scripts/guardrails.py) — stale names, legacy columns, wrong adapter depth
2. Fast pytest: `test_repo_root_paths`, `test_identity_axes`, `test_recording_axes`, `test_essentia_adapter`

For broader coverage after large refactors:

```bash
venvs/audio/bin/python -m pytest tests/test_identity_axes.py tests/test_recording_axes.py tests/test_tokenizer_lossless.py tests/test_audio_pipeline_db.py -q
```

---

## 4. Docs

Update:

- Root [CLAUDE.md](CLAUDE.md) if chain topology or deploy paths changed
- Per-module [CLAUDE.md](ingest/CLAUDE.md) / [analysis/CLAUDE.md](analysis/CLAUDE.md) / etc.
- [.cursor/rules/](.cursor/rules/) if new invariants apply

Cursor rules for identity and repo paths: `identity-axes.mdc`, `repo-paths.mdc`.

---

## 5. Cluster (if pi-storage affected)

Only after code is committed and pushed:

```bash
git push origin main
make deploy
make restart-jobqueue   # if FastAPI / shared libs changed
```

Verify code version on pi:

```bash
ssh pi-storage 'cd ~/tracklist_engine && git rev-parse --short HEAD'
```

For DB migrations, follow [docs/identity_and_inventory_plan.md](docs/identity_and_inventory_plan.md):

1. Backup canonical DB
2. Apply migration SQL
3. Run materializer / reconcile **dry-run only** unless operator approved

Reuse verification patterns from recent handoffs:

- [docs/agent_handoff_identity_rollout_20260530.md](docs/agent_handoff_identity_rollout_20260530.md)
- [docs/agent_handoff_reconcile_20260530.md](docs/agent_handoff_reconcile_20260530.md)

---

## 6. Handoff doc

If the refactor triggers long-running pi jobs or destructive scripts, write:

`docs/agent_handoff_<topic>_<YYYYMMDD>.md`

Include:

- Commit pin (`git rev-parse --short HEAD`)
- **Do NOT** list (re-run migrations, `--apply` without dry-run, second materialize, etc.)
- Verification commands (copy-pasteable over SSH)
- Manual follow-ups

Archive or delete the handoff when work is complete.

---

## Anti-patterns

- Combining rename + new feature + pi migration in one session
- Editing files directly on pi-storage instead of commit → push → deploy
- Skipping `make check` because "it's just a rename"
- Renaming `audio_pipeline_v1` DB source strings (breaks existing analysis rows)
- Re-running `reconcile_orphans.py --apply` without a fresh dry-run

## Related skills

- **cluster-deploy** — deploy, restart, logs
- **pi-storage-query** — canonical DB state over SSH
