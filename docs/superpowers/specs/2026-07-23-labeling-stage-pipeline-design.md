# Labeling package → stage-pipeline refactor

**Date:** 2026-07-23
**Branch:** `refactor/labeling-stage-pipeline` (worktree off `origin/main`)
**Status:** design approved; implementation pending plan

## Problem

`labeling/` reads as a pile of ~35 scripts with no expressed pipeline. A prior
in-flight pass (uncommitted on `chore/repo-housekeeping`) sorted the flat pile
into sub-packages by **file kind** (`aligning/`, `als/`, `content/`,
`gt_export/`, `ground_truth/`). That did not produce coherence: it split
tools by what they *are*, not by where they sit in the actual process. This
refactor supersedes that split and reorganizes by **pipeline stage**, and pays
down the structural debt the data-flow map surfaced.

Base is `origin/main` (the flat layout, pre-split). The throwaway sub-package
split is not built upon.

## The pipeline (agreed mental model)

Labeling is a linear, human-in-the-middle process. Five stages bracket the
human's Ableton alignment, plus two crosscutting concerns (the `.als` codec and
the GT schema) and a set of verify checkpoints run by hand between stages.

```
STAGE 1  ACQUIRE   replicate a set locally
   pull → ~/aligning/<set>/ : mix, tracks/, stems/, manifest.json, content_catalog.json
STAGE 2  PREP      make it annotatable by a human
   tag M4As → rename [NNNbpm KK] → relink .als → fill clip names
   ── HUMAN aligns in Ableton → <set> align.als (the identity oracle) ──
STAGE 3  EXTRACT   .als → ground_truth.yaml
STAGE 4  IDENTITY  bind each clip to a recording (content-addressed + manual overrides)
STAGE 5  COMMIT    ground_truth.yaml → set_ground_truth rows on pi-storage

CROSSCUTTING:  als/ (the .als codec) · schema · verify/ (gate, anchor, audit, review-UI)
```

## Design decision: legible toolbox, not orchestrator

Chosen shape: each stage is a clean, single-responsibility module with **one
clear entrypoint**, in stage-named folders, with a documented linear order and
the verify gates as **manual** checkpoints. No orchestrator, no per-set state
machine, no resumability. This fits small-N golden-set labeling done by hand;
the aligner exists so labeling never scales to a size that would justify a
state machine.

## Target structure

```
labeling/
  als/          the .als codec (UNCHANGED; crosscutting library)
  schema.py     the GT schema (moved from ground_truth/schema.py — it is one file)

  acquire/      STAGE 1
                  pull_set          (was pull_set_for_alignment.py)
                  inventory_check
                  reconcile         (was reconcile_aligning_manifest.py)  [repair]
                  quarantine        (was quarantine_aligning_orphans.py)  [repair]
                  add_candidates    (was add_separated_to_candidates.py)  [repair]
  prep/         STAGE 2
                  tag_features      (was tag_aligning_folder.py)
                  rename_inline     (was inline_tag_aligning_folder.py)
                  relink_als        (was relink_als_after_tag.py)
                  fill_clip_names   (was fill_als_clip_tags.py)
  extract/      STAGE 3
                  export            (was export_als_to_gt.py)
                  _shared           (constants + helpers hoisted out of the god-hub)
  identity/     STAGE 4  (absorbs content/ + identity_overrides/)
                  content_hash
                  content_resolver
                  build_catalog     (was build_content_catalog.py)
                  overrides/<set>.yaml  (was identity_overrides/)
  commit/       STAGE 5
                  write_back        (was write_back_ground_truth.py)

  verify/       crosscutting checkpoints
                  gt_als_gate · anchor_check · als_path_audit · review_ui
```

Two headline moves: **`content/` dissolves into `identity/`** (Operation Crush
becomes stage 4, not a grafted-on subsystem, with `overrides/` beside it), and
**the `gt_export/` bucket splits** into its three real jobs (extract / commit /
verify) plus identity.

## Debts and fixes

1. **Identity smeared across 4 mechanisms** (`content_resolver`, the name-match
   ladder in `enrich_gt_track_ids`, `remap_gt_slot_labels`, `identity_overrides`)
   → one `identity/` stage. Primary = content-addressed (`content_resolver`,
   already wired into export). Fallback = `overrides/`. **The legacy name-match
   guess-ladder inside `enrich_gt_track_ids` is retired** — it is what Operation
   Crush was built to replace and never finished retiring.
   ⚠️ **Only behavior-affecting change.** Gated hard (see Verification).
2. **`export_als_to_gt` god-hub** — a CLI that is secretly the shared library
   for all of `gt_export/` (everyone imports its `DEFAULT_ALS`,
   `collect_kept_clip_rows`, …) → hoist those into `extract/_shared.py`; the CLI
   becomes a thin caller.
3. **Two `.als`-editing fidelities** — the principled codec (`als/write`) vs raw
   gzip+regex string-substitution in `relink`/`fill` → route `relink`/`fill`
   through the `als/` codec.
4. **`ssh_sqlite` scattered/re-implemented across 3 files** → move to `core/` as
   the one SSH-sqlite primitive; all call sites import it.
5. **`content/audio_index.py` dead** — superseded by `content_catalog` +
   `content_resolver`; nothing in the export path reads it → delete (grep-verify
   first).

## Delete / keep / verify-then-decide

- **Delete outright:** the retired one-off scripts present on `main`
  (`bb12_gt_spotcheck.py`, `extract_winners.py`, `migrate_aligning_naming.py`)
  and the deprecated `als_io.py` re-export shim.
- **Delete after grep-verify:** `audio_index.py`; the name-match ladder body
  inside `enrich_gt_track_ids`.
- **Keep, absorb:** `identity_overrides/` → `identity/overrides/` (live BB12
  data — never deleted).
- **Verify then decide:** `remap_gt_slot_labels` — if it is genuine slot-label
  *normalization* it survives as an `extract/` step; if it is more
  identity-guessing it is retired with the ladder. Determined during
  implementation by reading what it actually does.

## Blast radius

- **Imports:** every `labeling.<oldpkg>.<mod>` / `labeling.<mod>` reference
  repointed to `labeling.<stage>.<mod>`. Absolute imports only.
- **Makefile:** `audit-gt`, `check-inventory`, and any `python -m labeling.*`
  targets repointed.
- **`labeling/CLAUDE.md`:** rewritten around the 5 stages.
- **Guardrails:** `scripts/guardrails.py` stale-name/path checks + the
  `als_core_boundary` fence updated for new paths.
- **CI gate:** `gt_als_gate` fixture/path references.
- **Root `CLAUDE.md` / memory:** the labeling layout note updated.

Mechanical moves driven with the **refactor-safety** skill. `make check` is the
gate for drift.

## Non-goals (YAGNI)

No `make label SET=x` orchestrator, no per-set state tracking, no resumability,
no new abstraction over the stages. Stage-named folders + one entrypoint each +
a documented linear order in CLAUDE.md.

## Verification

- **Structural moves (debts 2–5, all renames):** `make check` green; full
  labeling test subset green; `python -m` entrypoints import and run `--help`.
- **Identity-ladder retirement (debt 1) — hard gate:** re-run
  `export_als_to_gt` for **BB11 (`2nvzlh2k`) and BB12 (`1fsnxchk`)**, diff the
  resulting `*_ground_truth.yaml` against the committed fixtures — **must be
  byte-identical** (content-resolution + overrides fully cover what the ladder
  produced). `gt_als_gate` must stay green. If the diff is non-empty, the ladder
  is doing something the resolver does not; stop and re-scope rather than ship a
  GT change.

## Sequencing

Land as reviewable commits: (1) structural moves + import/Makefile/guardrail
updates + CLAUDE.md, verified by `make check`; (2) `ssh_sqlite → core`;
(3) codec-route `relink`/`fill`; (4) the identity-ladder retirement, behind its
own hard gate. If step 4's diff is non-empty, ship 1–3 and defer 4.
