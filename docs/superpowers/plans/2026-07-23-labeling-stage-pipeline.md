# Labeling Stage-Pipeline Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Drive all mechanical moves with the **refactor-safety** skill.

**Goal:** Reorganize `labeling/` by pipeline stage (acquire → prep → extract → identity → commit, plus the `als/` codec and a `verify/` checkpoint set) instead of by file-kind, and pay down the 5 structural debts the data-flow map surfaced — without changing GT output except the deliberately-gated identity-ladder retirement.

**Architecture:** Pure structural moves for debts 2–5 (renames + import/Makefile/guardrail/CLAUDE.md updates), verified by the targeted gate. One behavior-affecting change (debt 1, identity-ladder retirement) behind a hard byte-identical-GT gate, sequenced last so it can be deferred if the gate is not met.

**Tech Stack:** Python (no pyproject; `venvs/audio/bin/python`), SQLite over SSH to pi-storage, lxml for `.als`, pytest.

## Global Constraints

- Work only in the worktree `../tl-labeling-refactor` on branch `refactor/labeling-stage-pipeline` (off `origin/main`, flat layout). The `chore/repo-housekeeping` tree is off-limits.
- **Do NOT use the full pre-commit hook.** Its `pytest tests/` step is flaky (test-isolation bug in `tests/scripts/test_pi_autopull.py`) and pollutes the tree with junk files (`app.py`, `scripts/migrations/migrate_base.sql`). Commit with `git commit --no-verify` and run the **targeted gate** instead (defined below).
- **Targeted gate** (the meaningful validation, run from worktree root with `venvs/audio/bin/python`):
  ```
  venvs/audio/bin/python scripts/guardrails.py
  venvs/audio/bin/python -m labeling.<verify-pkg>.gt_als_gate    # path updates as it moves
  bash scripts/typecheck.sh
  venvs/audio/bin/python -m pytest tests/labeling -q               # labeling subset only
  ```
  After every task, all four must be green. After any commit, run `git status --short` and delete stray `app.py` / `scripts/migrations/migrate_base.sql` if a test created them.
- Imports are **absolute** (`labeling.<stage>.<module>`). No relative imports.
- Basenames are **unchanged** in this plan (only package paths move); `python -m` invocations change only their package segment. Basename-shortening is an explicit non-goal here (optional future polish).
- Baseline gate on the base commit is green: `guardrails` OK, `gt_als_gate` BB11=141 / BB12=152 labels OK, `typecheck` clean. Any task that turns one red must fix it before commit.

---

## Target layout (old → new)

```
labeling/als/*                         UNCHANGED (codec)
labeling/ground_truth/schema.py     → labeling/schema.py         (flatten; drop the package)

labeling/pull_set_for_alignment.py  → labeling/acquire/pull_set_for_alignment.py
labeling/inventory_check.py         → labeling/acquire/inventory_check.py
labeling/reconcile_aligning_manifest.py → labeling/acquire/reconcile_aligning_manifest.py
labeling/quarantine_aligning_orphans.py → labeling/acquire/quarantine_aligning_orphans.py
labeling/add_separated_to_candidates.py → labeling/acquire/add_separated_to_candidates.py

labeling/tag_aligning_folder.py     → labeling/prep/tag_aligning_folder.py
labeling/inline_tag_aligning_folder.py → labeling/prep/inline_tag_aligning_folder.py
labeling/relink_als_after_tag.py    → labeling/prep/relink_als_after_tag.py
labeling/fill_als_clip_tags.py      → labeling/prep/fill_als_clip_tags.py

labeling/export_als_to_gt.py        → labeling/extract/export_als_to_gt.py
                                      (+ labeling/extract/_shared.py — hoisted constants/helpers)

labeling/content_hash.py            → labeling/identity/content_hash.py
labeling/content_resolver.py        → labeling/identity/content_resolver.py
labeling/build_content_catalog.py   → labeling/identity/build_content_catalog.py
labeling/identity_overrides/*.yaml  → labeling/identity/overrides/*.yaml
labeling/audio_index.py             → DELETE (after dead-code proof)
labeling/enrich_gt_track_ids.py     → SPLIT (path-match+overrides → identity/; title-overlap ladder → retire)

labeling/write_back_ground_truth.py → labeling/commit/write_back_ground_truth.py

labeling/gt_als_gate.py             → labeling/verify/gt_als_gate.py
labeling/anchor_check.py            → labeling/verify/anchor_check.py
labeling/als_path_audit.py          → labeling/verify/als_path_audit.py
labeling/gt_review_ui.py            → labeling/verify/review_ui.py
labeling/remap_gt_slot_labels.py    → labeling/extract/remap_gt_slot_labels.py  (pending Task 6 verdict)

DELETE (dead one-offs / shim present on main):
  labeling/bb12_gt_spotcheck.py · labeling/extract_winners.py
  labeling/migrate_aligning_naming.py · labeling/als_io.py
```

---

### Task 1: Delete dead one-offs and the `als_io` shim

**Files:**
- Delete: `labeling/bb12_gt_spotcheck.py`, `labeling/extract_winners.py`, `labeling/migrate_aligning_naming.py`, `labeling/als_io.py`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing (removal only).

- [ ] **Step 1: Prove each is unreferenced**

Run (fixed glob — note the quotes):
```bash
cd ../tl-labeling-refactor
for m in bb12_gt_spotcheck extract_winners migrate_aligning_naming als_io; do
  echo "== $m =="; grep -rn "$m" --include='*.py' . | grep -v "labeling/$m.py" || echo "  (no refs)"
done
grep -rn "als_io" Makefile docs .claude 2>/dev/null || echo "  (no non-py als_io refs)"
```
Expected: `als_io` may appear as a deprecated re-export shim referenced by docstrings only; the three one-offs show no importers. If any real importer appears, STOP and reassess.

- [ ] **Step 2: Delete**
```bash
git rm labeling/bb12_gt_spotcheck.py labeling/extract_winners.py labeling/migrate_aligning_naming.py labeling/als_io.py
```

- [ ] **Step 3: Targeted gate green**

Run the Global-Constraints targeted gate. Expected: all four green (these files had no live consumers).

- [ ] **Step 4: Commit**
```bash
git commit --no-verify -m "refactor(labeling): drop dead one-offs + deprecated als_io shim"
```

---

### Task 2: Move Stage-1 (`acquire/`) and Stage-2 (`prep/`) modules

**Files:**
- Create: `labeling/acquire/__init__.py`, `labeling/prep/__init__.py`
- Move (git mv): the 5 acquire modules and 4 prep modules per the layout table.
- Modify: every importer of the moved modules; `Makefile`; `scripts/guardrails.py`; `.claude/skills/alignment-pull/SKILL.md` and any skill referencing these paths.

**Interfaces:**
- Consumes: nothing new.
- Produces: `labeling.acquire.<mod>`, `labeling.prep.<mod>` import paths. Note `labeling.acquire.pull_set_for_alignment.ssh_sqlite` is imported by `remap_gt_slot_labels` (handled in Task 5) — keep the symbol name `ssh_sqlite`.

- [ ] **Step 1: Create packages and move**
```bash
cd ../tl-labeling-refactor
mkdir -p labeling/acquire labeling/prep
: > labeling/acquire/__init__.py ; : > labeling/prep/__init__.py
git add labeling/acquire/__init__.py labeling/prep/__init__.py
for f in pull_set_for_alignment inventory_check reconcile_aligning_manifest quarantine_aligning_orphans add_separated_to_candidates; do
  git mv labeling/$f.py labeling/acquire/$f.py; done
for f in tag_aligning_folder inline_tag_aligning_folder relink_als_after_tag fill_als_clip_tags; do
  git mv labeling/$f.py labeling/prep/$f.py; done
```

- [ ] **Step 2: Rewrite intra-`labeling` imports of moved modules**

Find every reference and repoint. Run to enumerate first:
```bash
grep -rn -E "labeling\.(pull_set_for_alignment|inventory_check|reconcile_aligning_manifest|quarantine_aligning_orphans|add_separated_to_candidates|tag_aligning_folder|inline_tag_aligning_folder|relink_als_after_tag|fill_als_clip_tags)" --include='*.py' .
```
Repoint each hit: `labeling.pull_set_for_alignment` → `labeling.acquire.pull_set_for_alignment`, etc. Known cross-package importers: `reconcile_aligning_manifest`↔`inventory_check`, `quarantine_aligning_orphans`→`reconcile`, `remap_gt_slot_labels`→`pull_set_for_alignment.ssh_sqlite`. Also fix each moved file's own `Path(__file__).parents[N]` repo-root computation — the depth increased by 1 (`parents[2]` → `parents[3]`). Grep: `grep -rn "parents\[" labeling/acquire labeling/prep`.

- [ ] **Step 3: Update `Makefile` and skills**
```bash
grep -n -E "labeling\.(pull_set_for_alignment|inventory_check|tag_aligning_folder|inline_tag_aligning_folder|relink_als_after_tag|fill_als_clip_tags|reconcile_aligning_manifest|quarantine_aligning_orphans|add_separated_to_candidates)" Makefile
grep -rn -E "labeling/(pull_set_for_alignment|tag_aligning_folder|...)" .claude/skills
```
Repoint `python -m labeling.<mod>` → `python -m labeling.<stage>.<mod>` and `labeling/<mod>.py` → `labeling/<stage>/<mod>.py` in `Makefile` (`check-inventory`, any tag/pull targets) and in `.claude/skills/alignment-pull/SKILL.md`.

- [ ] **Step 4: Update guardrails stale-path map**

Open `scripts/guardrails.py`; update any hardcoded `labeling/<mod>.py` paths and stale-name entries to the new stage paths. Run `venvs/audio/bin/python scripts/guardrails.py` and fix until green.

- [ ] **Step 5: Targeted gate + import smoke**
```bash
for m in labeling.acquire.pull_set_for_alignment labeling.acquire.inventory_check labeling.prep.tag_aligning_folder labeling.prep.inline_tag_aligning_folder labeling.prep.relink_als_after_tag labeling.prep.fill_als_clip_tags; do
  venvs/audio/bin/python -c "import importlib,sys; importlib.import_module('$m'); print('ok $m')" || exit 1; done
```
Then the full targeted gate. Expected: all green.

- [ ] **Step 6: Commit**
```bash
git commit --no-verify -m "refactor(labeling): stage-1 acquire/ + stage-2 prep/ folders"
```

---

### Task 3: Move `verify/` (gate, anchor, audit, review UI) and flatten `schema.py`

**Files:**
- Create: `labeling/verify/__init__.py`
- Move: `gt_als_gate.py`, `anchor_check.py`, `als_path_audit.py`, `gt_review_ui.py` → `labeling/verify/` (review_ui keeps basename `gt_review_ui.py` this pass). Move `ground_truth/schema.py` → `labeling/schema.py`; `git rm -r labeling/ground_truth`.
- Modify: every `labeling.ground_truth.schema` importer → `labeling.schema`; every `labeling.gt_als_gate` / `anchor_check` / `als_path_audit` / `gt_review_ui` importer and CLI ref; `.githooks/pre-commit` (its `-m labeling.gt_als_gate` line → `labeling.verify.gt_als_gate`); `Makefile` `audit-gt`; `.github/workflows/guardrails.yml` if it names the gate; `scripts/guardrails.py`.

**Interfaces:**
- Consumes: `labeling.schema` (moved) by every YAML producer/consumer.
- Produces: `labeling.verify.gt_als_gate` (CLI), `labeling.schema.{GroundTruthSet,GroundTruthTrack,RefSegment,load,save,dump}`.

- [ ] **Step 1: Move gate/verify modules + flatten schema**
```bash
cd ../tl-labeling-refactor
mkdir -p labeling/verify ; : > labeling/verify/__init__.py ; git add labeling/verify/__init__.py
for f in gt_als_gate anchor_check als_path_audit gt_review_ui; do git mv labeling/$f.py labeling/verify/$f.py; done
git mv labeling/ground_truth/schema.py labeling/schema.py
git rm labeling/ground_truth/__init__.py
```

- [ ] **Step 2: Repoint `labeling.ground_truth.schema` → `labeling.schema` everywhere**
```bash
grep -rln "labeling\.ground_truth\.schema" --include='*.py' . | xargs sed -i '' 's/labeling\.ground_truth\.schema/labeling.schema/g'
grep -rn "labeling\.ground_truth" --include='*.py' . || echo "  (clean)"
```

- [ ] **Step 3: Repoint verify-module imports + CLI refs + `.githooks`**

Repoint `labeling.gt_als_gate`→`labeling.verify.gt_als_gate` (and the other three) in all `.py`, `Makefile` (`audit-gt`), `.githooks/pre-commit` line 18, `.github/workflows/guardrails.yml`, and `scripts/guardrails.py`. Fix `parents[N]` depth in the moved files (`grep -rn "parents\[" labeling/verify labeling/schema.py`).

- [ ] **Step 4: Targeted gate (note: gate path changed)**

Run gate via new path: `venvs/audio/bin/python -m labeling.verify.gt_als_gate` — expect BB11 141 / BB12 152 OK. Then guardrails, typecheck, `pytest tests/labeling -q`.

- [ ] **Step 5: Commit**
```bash
git commit --no-verify -m "refactor(labeling): verify/ checkpoints + flatten schema.py"
```

---

### Task 4: Extract stage — move `export_als_to_gt`, hoist the shared hub into `extract/_shared.py`

**Files:**
- Create: `labeling/extract/__init__.py`, `labeling/extract/_shared.py`
- Move: `export_als_to_gt.py` → `labeling/extract/export_als_to_gt.py`
- Modify: importers of `DEFAULT_ALS`, `DEFAULT_SET_DIR`, `ClipRow`, `collect_kept_clip_rows` (currently imported FROM `export_als_to_gt` by `enrich_gt_track_ids`, `anchor_check`, `als_path_audit`, `gt_review_ui`).

**Interfaces:**
- Consumes: `labeling.als.*`, `labeling.identity.content_hash`, `labeling.identity.content_resolver` (Task 5 lands identity; if Task 5 not yet done, these still resolve at the old `labeling.content.*` path — sequence Task 5 before this OR keep a temporary import. **Sequence: do Task 5 before Task 4.**)
- Produces: `labeling.extract._shared.{DEFAULT_ALS, DEFAULT_SET_DIR, ClipRow, collect_kept_clip_rows}` and `labeling.extract.export_als_to_gt` (the CLI `export_gt`).

- [ ] **Step 1: Move export into extract/**
```bash
mkdir -p labeling/extract ; : > labeling/extract/__init__.py
git mv labeling/export_als_to_gt.py labeling/extract/export_als_to_gt.py
```

- [ ] **Step 2: Identify the shared surface**

List what the four downstream tools import from export:
```bash
grep -rn "from labeling\.\(extract\.\)\?export_als_to_gt import" --include='*.py' .
```
Expected symbols: `DEFAULT_ALS`, `DEFAULT_SET_DIR`, `ClipRow`, `collect_kept_clip_rows`.

- [ ] **Step 3: Move those definitions into `extract/_shared.py`**

Cut the definitions of `DEFAULT_ALS`, `DEFAULT_SET_DIR`, `ClipRow`, `collect_kept_clip_rows` (and any private helper they alone need) out of `export_als_to_gt.py` into `extract/_shared.py`. In `export_als_to_gt.py`, `from labeling.extract._shared import DEFAULT_ALS, DEFAULT_SET_DIR, ClipRow, collect_kept_clip_rows`. Repoint the four downstream importers to `labeling.extract._shared`.

- [ ] **Step 4: Repoint remaining `labeling.export_als_to_gt` refs**
```bash
grep -rn "labeling\.export_als_to_gt" --include='*.py' . && echo "repoint each ^ to labeling.extract.export_als_to_gt"
```
Update `parents[N]` depth in the moved file.

- [ ] **Step 5: Targeted gate + export smoke on BB12**
```bash
venvs/audio/bin/python -m labeling.extract.export_als_to_gt --help >/dev/null && echo "cli ok"
```
Then the targeted gate. Expected green.

- [ ] **Step 6: Commit**
```bash
git commit --no-verify -m "refactor(labeling): extract/ stage + hoist shared hub out of export god-module"
```

---

### Task 5: Identity stage — dissolve `content/` into `identity/`, absorb overrides, move `ssh_sqlite` to core

**Files:**
- Create: `labeling/identity/__init__.py`, `labeling/identity/overrides/` (dir)
- Move: `content_hash.py`, `content_resolver.py`, `build_content_catalog.py` → `labeling/identity/`; `identity_overrides/*.yaml` → `labeling/identity/overrides/`.
- Create: `core/ssh_sqlite.py` — the one SSH-sqlite primitive.
- Modify: `acquire/pull_set_for_alignment.py` (its `ssh_sqlite` + its `python3 -m labeling.content.build_content_catalog` shell-out → `labeling.identity.build_content_catalog`), `extract/export_als_to_gt.py` (`labeling.content.*` → `labeling.identity.*`, and its `_load_content_catalog`), `enrich_gt_track_ids.py` (`load_identity_overrides` path → `labeling/identity/overrides/`), and every `labeling.content.*` importer.

**Interfaces:**
- Consumes: `core.ssh_sqlite.ssh_sqlite(query, *, host, db) -> list[dict]` (new home).
- Produces: `labeling.identity.{content_hash, content_resolver, build_content_catalog}`; `labeling/identity/overrides/<set_id>.yaml`.

- [ ] **Step 1: Move content modules + overrides**
```bash
mkdir -p labeling/identity/overrides ; : > labeling/identity/__init__.py
for f in content_hash content_resolver build_content_catalog; do git mv labeling/$f.py labeling/identity/$f.py; done
git mv labeling/identity_overrides/*.yaml labeling/identity/overrides/
git rm -r --ignore-unmatch labeling/identity_overrides
```

- [ ] **Step 2: Repoint `labeling.content.*` → `labeling.identity.*`**
```bash
grep -rln "labeling\.content\." --include='*.py' . | xargs sed -i '' 's/labeling\.content\./labeling.identity./g'
grep -rn "labeling\.content\b" --include='*.py' . || echo "  (clean)"
```
Also update the shell-out string in `pull_set_for_alignment.py`: `python3 -m labeling.content.build_content_catalog` → `labeling.identity.build_content_catalog`. Update `OVERRIDES_DIR` in `enrich_gt_track_ids.py` (`labeling/identity_overrides` → `labeling/identity/overrides`). Fix `parents[N]` depths. **Note:** `build_content_catalog` runs on pi under bare `python3` — keep it stdlib-only (no new imports).

- [ ] **Step 3: Create `core/ssh_sqlite.py` and repoint the three call sites**

Extract the canonical SSH-sqlite helper (from `pull_set_for_alignment.ssh_sqlite`) into `core/ssh_sqlite.py` as `ssh_sqlite(query, *, host, db)`. Repoint `acquire/pull_set_for_alignment.py` to import it, `extract`/`enrich`'s `_ssh_sql`/`remap`'s usage to import from `core.ssh_sqlite`. Delete the duplicate `_ssh_sql` body in `enrich_gt_track_ids.py`.

- [ ] **Step 4: Targeted gate**

Guardrails (update its `labeling/content` path entries), typecheck, gate, `pytest tests/labeling -q`. Verify `build_content_catalog` still stdlib-only: `venvs/audio/bin/python -c "import ast,sys; ast.parse(open('labeling/identity/build_content_catalog.py').read())"` and eyeball its imports.

- [ ] **Step 5: Commit**
```bash
git commit --no-verify -m "refactor(labeling): identity/ stage (absorb content/ + overrides), ssh_sqlite -> core"
```

---

### Task 6: Move `commit/` (write_back) + decide `remap_gt_slot_labels`'s home + prove `audio_index` dead

**Files:**
- Create: `labeling/commit/__init__.py`
- Move: `write_back_ground_truth.py` → `labeling/commit/`; `remap_gt_slot_labels.py` → `labeling/extract/` (verdict below); delete `audio_index.py` if proven dead.
- Modify: `Makefile` write-back target; importers.

**Interfaces:**
- Consumes: `labeling.schema`, `core.ssh_sqlite`.
- Produces: `labeling.commit.write_back_ground_truth` (CLI).

- [ ] **Step 1: Prove `audio_index` dead, then delete**
```bash
grep -rn "audio_index" --include='*.py' . | grep -v "labeling/audio_index.py" || echo "  DEAD — no importers"
```
If dead: `git rm labeling/audio_index.py`. If any importer appears, move it to `identity/` instead and note why in the commit.

- [ ] **Step 2: Read `remap_gt_slot_labels` — normalization or guessing?**

Read the file. If it canonicalizes `slot_label` against pi `set_track_slots` labels (normalization) → it survives; `git mv labeling/remap_gt_slot_labels.py labeling/extract/remap_gt_slot_labels.py`. If it *assigns identity by guessing* → leave it in place, mark for Task 7 retirement, and note the verdict in this step. Record the verdict in the commit message.

- [ ] **Step 3: Move write_back**
```bash
mkdir -p labeling/commit ; : > labeling/commit/__init__.py
git mv labeling/write_back_ground_truth.py labeling/commit/write_back_ground_truth.py
```

- [ ] **Step 4: Repoint imports + Makefile + guardrails + parents[N]**
```bash
grep -rn "labeling\.write_back_ground_truth\|labeling\.remap_gt_slot_labels\|labeling\.audio_index" --include='*.py' . Makefile
```
Repoint all; fix depths; update guardrails path map.

- [ ] **Step 5: Targeted gate**

All four green.

- [ ] **Step 6: Commit**
```bash
git commit --no-verify -m "refactor(labeling): commit/ stage; remap verdict=<normalization|retire>; drop dead audio_index"
```

---

### Task 7: Rewrite `labeling/CLAUDE.md` + root CLAUDE.md pointer + memory note around the 5 stages

**Files:**
- Modify: `labeling/CLAUDE.md` (rewrite "Layout" around acquire/prep/extract/identity/commit + als + verify), root `CLAUDE.md` (labeling layout mentions), any `.cursor/rules` path refs.

**Interfaces:** docs only.

- [ ] **Step 1: Rewrite `labeling/CLAUDE.md` "Layout" section** to the 5-stage model with the new paths; keep the seeded-vs-hand provenance and consistency-model sections intact (update only paths).

- [ ] **Step 2: Grep root CLAUDE.md + cursor rules for stale `labeling/<oldpath>` and fix**
```bash
grep -rn -E "labeling/(export_als_to_gt|write_back_ground_truth|content_|build_content_catalog|gt_als_gate|pull_set_for_alignment|ground_truth/schema)" CLAUDE.md .cursor 2>/dev/null
```

- [ ] **Step 3: Full targeted gate + `make check`**

Run `make check` (guardrails + entropy audit + fast pytest subset) — note if `make check` itself invokes the polluting suite; if so, run its guardrails+pytest-subset components directly. Expected green.

- [ ] **Step 4: Commit**
```bash
git commit --no-verify -m "docs(labeling): rewrite CLAUDE.md around the 5-stage pipeline"
```

---

### Task 8 (GATED, may be deferred): Retire the identity-ladder in `enrich_gt_track_ids`

> ⚠️ Only behavior-affecting task. If its gate is not met, STOP, ship Tasks 1–7, and open a follow-up. Do not ship a GT change on a non-empty diff.

**Files:**
- Modify/split: `labeling/enrich_gt_track_ids.py` (final home: `labeling/identity/enrich_gt_track_ids.py` — move it as part of this task). Keep: exact-path-match + overrides application. Remove: the unique-`set_track_slots`-title-overlap tier (`_overlap_score`, `lookup_db_label` title path).

**Interfaces:**
- Consumes: `labeling.identity.content_resolver`, `labeling/identity/overrides/*.yaml`, `core.ssh_sqlite`.
- Produces: `labeling.identity.enrich_gt_track_ids` (CLI) producing GT `track_id`s **only** via content-resolution + overrides + exact path match.

- [ ] **Step 1: Capture golden GT before any change**
```bash
cp labeling/fixtures/bb11_ground_truth.yaml /tmp/bb11_gold.yaml
cp labeling/fixtures/bb12_ground_truth.yaml /tmp/bb12_gold.yaml   # adjust names to actual fixtures
```
Confirm fixture filenames with `ls labeling/fixtures/*ground_truth*`.

- [ ] **Step 2: Establish whether any GT row resolves ONLY via the title-overlap tier**

Instrument or read: for BB11 and BB12, determine if every `track_id` currently produced by the title-overlap tier is *also* obtainable via content-resolution or an override. If some rows depend solely on title-overlap, those need explicit overrides authored first (add to `labeling/identity/overrides/<set>.yaml`). Document findings inline.

- [ ] **Step 3: Author any needed overrides, then remove the title-overlap tier**

Delete `_overlap_score` and the title-matching path in `lookup_db_label`; keep exact-path-match + overrides. Move the file to `labeling/identity/enrich_gt_track_ids.py`; repoint refs.

- [ ] **Step 4: Re-export BB11 + BB12 and diff — HARD GATE**
```bash
venvs/audio/bin/python -m labeling.extract.export_als_to_gt --als <BB11 als> --set-dir <BB11 dir> --out /tmp/bb11_new.yaml
venvs/audio/bin/python -m labeling.identity.enrich_gt_track_ids ... /tmp/bb11_new.yaml   # same pipeline as before
diff /tmp/bb11_gold.yaml /tmp/bb11_new.yaml && echo "BB11 BYTE-IDENTICAL"
# repeat BB12
```
Expected: **byte-identical** for both. If not, STOP — the ladder was doing something the resolver+overrides do not; revert Task 8 and open a follow-up issue.

- [ ] **Step 5: Gate + commit**
```bash
venvs/audio/bin/python -m labeling.verify.gt_als_gate   # BB11 141 / BB12 152 OK
git commit --no-verify -m "refactor(labeling): retire identity-ladder — content-resolver + overrides only (GT byte-identical)"
```

---

## Self-Review

- **Spec coverage:** Debt 1 → Task 8 (gated). Debt 2 (god-hub) → Task 4. Debt 3 (dup `.als` editing via codec) — **GAP**: the plan does not yet route `relink`/`fill` through the `als/` codec. Add as **Task 9** (below) or fold into a follow-up; it is pure-internal, non-GT, low-risk, and was listed in the spec sequencing as step 3. Debt 4 (ssh_sqlite→core) → Task 5. Debt 5 (audio_index dead) → Task 6. Structure/deletes → Tasks 1–7.
- **Placeholders:** none (angle-bracket `<BB11 als>` in Task 8 is a real path filled from `ls labeling/fixtures` / the alignment-pull skill; acceptable as it is environment-specific).
- **Type consistency:** `ssh_sqlite(query, *, host, db)` used consistently in Tasks 2/5/6/8; `_shared` symbols consistent Task 4↔downstream.

### Task 9: Route `relink_als_after_tag` / `fill_als_clip_tags` through the `als/` codec

**Files:** Modify `labeling/prep/relink_als_after_tag.py`, `labeling/prep/fill_als_clip_tags.py`.

- [ ] **Step 1:** Read `als/write.py` + `als/cst.py` to confirm the codec exposes in-place `Path`/`RelativePath`/`Name` mutation with gzip round-trip and backup.
- [ ] **Step 2:** If the codec covers it, replace the raw gunzip→regex→gzip in `relink`/`fill` with codec calls; keep the `*.bak` behavior. If the codec lacks a needed primitive, add it to `als/write.py` with a roundtrip test in `tests/labeling/`. If the gap is large, defer to a follow-up and note why.
- [ ] **Step 3:** Round-trip test on a committed `.als` fixture: relink→fill→open-check, assert clips resolve. Targeted gate green.
- [ ] **Step 4:** Commit `refactor(labeling): route relink/fill .als edits through the als codec`.

---

## Execution order

1 → (2, 3 independent) → 5 → 4 → 6 → 7 → 9 → 8 (gated, last). Task 5 **before** Task 4 (export imports identity). Task 8 last and independently revertible.
