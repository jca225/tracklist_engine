# HANDOFF: Repo declutter — Tracklist Engine

> For: an agent executing a low-risk repo cleanup
> Author context: produced from a live audit on 2026-07-26 (branch
> `chore/merge-train-deploy-routing`, +0 vs main)
> Goal: reduce visual/structural clutter without touching the alignment DAG or
> breaking any of the ~9–30 in-flight branches.

## Core findings (don't re-derive — these were measured)

- **First-party surface:** 972 tracked `.py`, ~174k LOC. The DAG chain
  (`core` · `analysis` · `ingest` · `labeling` · `tokenizer` · `web_crawler`) is
  only ~15% of it and is lean and correct — leave it alone.
- **`workspaces/`** is 41% of tracked Python, but 340 of its 403 files are LIVE
  (`alignment_prototype` 276 files committed today; `pws_aligner` 64 files,
  cotrain active). Only ~63 files are graveyard.
- **The provenance engine is wired into the shipped hot path**
  (`analysis/persistence.py`, and related contracts/ingest surface). It is
  **KEEP**. Do not remove or "park" it. (An older memory note calls it
  un-wired — that note is stale.)
- **The real clutter** is ~880M of untracked scratch + generated/vendored dirs
  polluting the VSCode explorer, not the code structure.

## HARD RULES — read before doing anything

1. **DO NOT physically nest the DAG** (no `pipeline/{core,analysis,…}`). It
   would rewrite 5,166 import lines across 2,540 files, break
   `python -m ingest.main` systemd units on pi-storage, and force every
   in-flight branch to rebase. Explicitly rejected.
2. **DO NOT rename or move any tracked code directory in this pass.** braid
   reports 176-file overlap with `chore/repo-housekeeping` + 8 more branches.
   Renames wait for a quiet tree (a later "Wave 2").
3. **DO NOT touch:** `data/` (46G; but 113 tracked `data/djs/*.json` job files
   must stay), `venvs/`, the DAG modules, `alignment`,
   `pws_aligner`, `core/provenance`, `tests/`.
4. **Tracked vs untracked matters:** tracked files → `git rm` (git history is
   the "just in case" backup, fully recoverable). Untracked files → physically
   move (git won't save them; deleting is permanent).
5. **Confirm the attic path with the human before moving anything:** default
   `~/tracklist_attic/`.
6. **Show every destructive command's output; verify counts before and after.**
   Nothing here is urgent — correctness over speed.

---

## TASK 1 — VSCode view fix (SAFE on any branch, do first)

Create `.vscode/settings.json` (shared explorer excludes). Commit:

`chore(vscode): hide generated+vendored dirs from explorer.`

```json
{
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "venvs": true,
    "data": true,
    "logs": true,
    "profiles": true,
    "reports": true,
    "deploy": true,
    "cue-detr": true,
    ".claude/worktrees": true
  },
  "search.exclude": { "venvs": true, "data": true, "archive": true }
}
```

---

## TASK 2 — Write the plan doc + fix stale memory (SAFE, do first)

2a. Save this handoff to `docs/repo_declutter_plan.md`, then add a one-line
pointer to `docs/design_docs_index.md`.

2b. Update the provenance memory note at
`~/.claude/projects/-Users-johnnycabrahams-Desktop-tracklist-engine/memory/project_provenance_engine_phase0.md`:
the claim "shipped infer.py/analysis/labeling NOT rewired" is now false —
provenance is partially wired into the hot path as of 2026-07-26 (notably
`analysis/persistence.py` dual-write). Do not delete the rest of the note.

---

## TASK 3 — Wave 1a: evict untracked scratch (needs attic-path confirmation)

These dirs are untracked (0 files in git) — verify with
`git ls-files <dir> | wc -l` returns 0 before moving. Physically move to the
attic; do **NOT** `git rm` (they aren't tracked).

```bash
cd /Users/johnnycabrahams/Desktop/tracklist_engine
ATTIC=~/tracklist_attic            # CONFIRM with human first
mkdir -p "$ATTIC"
for d in archive _mac_scratch profiles logs scratchpad; do
  test -z "$(git ls-files "$d")" || { echo "ABORT: $d has tracked files"; break; }
  mv "$d" "$ATTIC/" && echo "moved $d"
done
rm -rf __pycache__      # regenerated, safe
```

Reclaims ~880M. `archive/` (48M) is not in git history — this is the only copy,
hence move-not-delete.

Leave `data/`, `papers/`, `reports/`, `deploy/` in place (`papers`/`reports`/
`deploy` are tracked; `data` is the DB copy).

---

## TASK 4 — Wave 1b: retire dead workspace forks (new branch)

Branch off main:

```bash
git switch main && git pull --ff-only && git switch -c chore/repo-declutter
```

### Delete-or-park ledger for `workspaces/`

| Fork | tracked py | last commit | verdict | action |
|---|---|---|---|---|
| `msst_webui` | 0 | none | empty | **PARK — live RoFormer root** (see execution notes) |
| `streaming_mir` | 5 | 07-15 | already promoted into `analysis/pipeline.py` | **RETIRED 2026-07-27** |
| `separation_qa` | 7 | 07-10 | isolated? | **PARK — setup script dependency** (see execution notes) |
| `mashup_compat` | 7 | 07-23 | recent, isolated | PARK — leave for now |
| `source_detection` | 10 | 07-19 | 7 refs, verify first | PARK — leave for now |
| `section_hsmm` | 34 | 07-02 | cooling | PARK — leave for now |
| `alignment_prototype` | 276 | today | LIVE | DO NOT TOUCH |
| `pws_aligner` | 64 | 07-23 | LIVE (cotrain) | DO NOT TOUCH |

### Pre-delete isolation check

```bash
for f in msst_webui streaming_mir separation_qa; do
  echo "== $f =="; grep -rl "workspaces/$f" --include='*.py' . \
    --exclude-dir=venvs --exclude-dir=__pycache__ | grep -v "workspaces/$f/"
done
```

If a grep shows a **real code importer** outside the fork, stop and report —
don't delete it.

Then: `make check` must pass before commit. Commit as
`chore(workspaces): retire promoted/dead forks (…)`. Open a PR against main.

---

## VERIFICATION CHECKLIST

- [ ] `git ls-files <moved-dir>` was empty before every `mv` in Task 3
- [ ] `make check` green after Task 4
- [ ] no import of a deleted fork remains
- [ ] DAG modules, `core/provenance`, `tests/`, `data/djs/*.json` untouched
- [ ] VSCode explorer now shows ~11 meaningful folders

---

## OUT OF SCOPE (do NOT attempt without fresh human sign-off)

- Nesting/renaming the DAG or any tracked dir (Rules 1 & 2)
- Parking `mashup_compat` / `source_detection` / `section_hsmm` (needs
  verification pass — Wave 2)
- Consolidating off-DAG dirs (`eda`/`personalization`/`soundcloud`) under a
  parent — legitimate but a rename, so it waits for the branch backlog to clear
- Any `docs/` consolidation (high branch-overlap)

---

## Execution notes (2026-07-27)

Isolation check before any `git rm` found live external dependencies the
original ledger understated:

1. **`workspaces/msst_webui` — DO NOT REMOVE.** Zero tracked files (gitignored
   vendored tree, ~2.4G), but it is the **live MSST RoFormer root**:
   `analysis/roformer_config.py` (`DEFAULT_MSST_ROOT`),
   `analysis/roformer_chain.yaml` (`msst_root:`), and
   `scripts/setup_roformer_separation.sh`. `.gitignore` already documents it as
   the vendored separation trainer. Hide from explorer if desired; never `rm`.
2. **`workspaces/separation_qa` — DO NOT REMOVE in this pass.**
   `scripts/setup_roformer_separation.sh` invokes
   `workspaces/separation_qa/download_msst_models.py` and writes smoke paths
   under that tree. Comment/doc refs elsewhere are fine; the setup script is a
   real external consumer.
3. **`workspaces/streaming_mir` — RETIRED 2026-07-27** on `chore/repo-declutter`.
   External hits were comment/doc only (`scripts/render_set_stems.py`, tests,
   CLAUDE.md). No `from workspaces.streaming_mir` outside the fork. Promoted
   findings live in `analysis/` + `render_set_stems.py`. Untracked
   `ws2_snippets/` moved to `~/tracklist_attic/streaming_mir/`.
