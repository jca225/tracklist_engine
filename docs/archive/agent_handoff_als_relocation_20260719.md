# Agent handoff — BB12 `.als` relocation + fixture-id reconciliation (2026-07-19)

## Status: BLOCKED on a human (John) — resume when he's awake.

> **⚠ UPDATE 2026-07-21 (see [operation_crush_master_plan.md](operation_crush_master_plan.md) §4):**
> **Step A (relocation) is DONE** — BB12 is canonical at
> `~/aligning/_labeling/1fsnxchk/BB12 align Project/bb12_align.als` (final name
> `bb12_align.als`, not "BB12 align.als"), relinked **613/613**. Ref counts here
> ("574 refs") are superseded by **318 external + 295 local**. **Steps B (export
> search paths) and C (manifest↔`.als` reconcile) remain live inputs to Phase 1.**

John approved the approach but deferred execution to tomorrow. **Do not start
without confirming the open decisions below with him.**

## Context (why this exists)

The alignment scorer's "same song" pool is keyed on `work_id` via
`labeling/fixtures/id_maps/<set>_work.json` (built by `scripts/build_work_map.py`
from the GT fixture's `track_id`s). The committed `bb12_ground_truth.yaml`
carries **stale `track_id`s** — 14 slots resolve to a *different song* in the
canonical DB (e.g. slot 003 "Two Friends - Emily (Remix)" has `track_id=2uq9800f`
which is "Pacific Coast Highway (Acappella)"). The scorer is now robust to this
(canonical pool, no name-token guessing — stale ids surface as honest misses,
not silent wrong-song credits), but the fixture itself must be fixed to get
correct scores.

**Root cause:** `export_als_to_gt.py` copies the pull manifest's `recording_id`
into the fixture's `track_id`. The BB12 manifest
(`~/aligning/1fsnxchk__Two Friends - Big Bootie Mix Volume 12/manifest.json`)
had `recording_id=None` for all 165 tracks, AND its slot content diverged from
the `.als` (manifest slot 003 = "Calvin Harris - Outside", `.als` slot 003 =
Emily). So `export_als_to_gt`'s path-match misses and the `slot_id_map` fallback
carries the stale ids through.

## What's already DONE (committed on `trm-ablation-framework`, NOT pushed — branch is behind origin by 28, so a push needs a rebase first)

Three commits this session:
1. `d700e94` — scorer + display + capture-fidelity fence (canonical work-sibling
   pool, audible-truth scoring, `_detect_loops` track_name fix, `is_success`
   relaxation for cross-recording stem-compatible matches, invariant tests).
2. `baa9f36` — `scripts/build_work_map.py`, `scripts/audit_gt_recording_ids.py`,
   `scripts/reconcile_works.py` + work-map fixtures + guardrails ratchet bumps
   (justified in commit body).
3. `29b49e1` — `scripts/resolve_manifest_recording_ids.py` + ratchet bump.

Key findings (do NOT re-litigate):
- Canonical DB `work` table is **healthy**: 0 same-song-different-work gaps
  corpus-wide. No DB reconciliation needed. The Emily premise was wrong.
- Only DB integrity nits: 2 dangling recording→work FKs (`106dumq5`,
  `10d2kzhp` — empty names, referenced by 2 `track_audio` + 2 `set_track_slots`
  rows), and 956 empty-name placeholder recordings. Neither feeds the scorer.
  Leave for a separate DB-hygiene pass; do not block on them.

## What's already DONE (local state, NOT committed — John authorized, reversible)

- **BB12 manifest resolved:** `scripts/resolve_manifest_recording_ids.py --set-id
  1fsnxchk --apply` wrote 152 fresh `recording_id`s (derived from `pi_path`,
  DB-verified). Backup at
  `~/aligning/1fsnxchk__…/manifest.json.bak_resolve_20260720T002144Z`.
  7 mismatches + 1 not-in-DB were flagged (not written) — need human review.
- **Committed fixture is INTACT** (a re-export was attempted, found to NOT fix
  ids — `slot_id_map` fallback reproduced stale ids — and it shifted slot 111's
  span by 74s, so the fixture was restored from
  `labeling/fixtures/bb12_ground_truth.yaml.bak_pre_resolve_20260720T002150Z`.
  Verified byte-identical. Do NOT re-export until the manifest/`.als` path
  divergence is resolved.)

## THE OPEN TASK — resume here

### Step A — Relocate the BB12 `.als` to a canonical path (copy-only, no relink)

John picked the **"BB align"** naming convention (`BB11 align.als` / `BB12 align.als`).

**Constraint (critical):** the BB12 `.als` uses **depth-3 relative sample refs**
(`../../../1fsnxchk__Two Friends - Big Bootie Mix Volume 12/tracks/…`, 574 refs).
A copy with NO relink only works at the same depth (3 under `~/aligning/`).
Putting it directly in the set dir (depth 1) would orphan all 574 samples.

Source: `~/aligning/_backups/20260616_150150/big bootie 12 labeling Project/big bootie 12 labeling_fast.als` (493k; mtime July 12 21:46 — newest).

BB11 `.als` is **already canonical** at
`~/aligning/2nvzlh2k__Two Friends - Big Bootie Mix Episode 11/BB11 align Project/BB11 align.als`
(depth 2, depth-2 refs). Leave BB11 alone — copying it to depth 3 would break
its refs without a relink.

**Plan John was choosing between (confirm which):**
- **"go"** — copy just the `.als` (493k) to
  `~/aligning/_labeling/1fsnxchk/big bootie 12 labeling Project/BB12 align.als`
  (depth 3, refs preserved, renamed). Lightest; sufficient for export.
- **"go full"** — same, but also copy the whole 5 G BB12 project folder
  (`Samples/`, `Backup/`, `Ableton Project Info/`) so Ableton can open the copy
  with its own samples. Disk is fine (267 Gi free; folder is 5 G).

Disk: 267 Gi free on `/System/Volumes/Data`. BB12 project dir = 5.0 G, BB11 = 2.8 M. Space is a non-issue.

**Verify after copy (no fixture overwrite):**
```
ALS=~/aligning/_labeling/1fsnxchk/big\ bootie\ 12\ labeling\ Project/BB12\ align.als
SETDIR=~/aligning/1fsnxchk__Two\ Friends\ -\ Big\ Bootie\ Mix\ Volume\ 12
venvs/audio/bin/python -m labeling.export_als_to_gt --als "$ALS" --set-dir "$SETDIR" --out /tmp/bb12_reexport_check.yaml
# expect 171 slot rows; compare structure to committed fixture (slot, set_start_s, set_end_s, claimed_stem)
```
Also: have John open the copied `.als` in Ableton to confirm samples resolve
(proves the depth math). The committed fixture must NOT be overwritten in this
step.

### Step B — Update code to search canonical locations

`labeling/export_als_to_gt.py` `DEFAULT_ALS` is stale
(`~/Desktop/big bootie 12 labeling Project/…`, gone). Replace with a
`find_default_als(set_dir)` that searches, in order:
1. `<set_dir>/*.als` (non-SEEDED) — works after a future Save-As into the set dir.
2. `~/aligning/_labeling/<set_id>/*/BB12 align.als`-style (the Step A copy).
3. The `_backups/*/big bootie 12 labeling Project/*.als` fallback (today's actual).
4. The old Desktop default (backcompat).

Update `tests/labeling/test_export_capture_fidelity.py` to use the same search
(it currently imports `DEFAULT_ALS` directly and skips when absent). Separate
commit; guardrails + tests must pass. Note: adding argparse/pathlib patterns
may trip the `kernel_flags`/`parents_depth`/`raw_manifest_read` ratchets — bump
with justification in the commit body if so (precedent: commits `baa9f36`,
`29b49e1`).

### Step C — Reconcile the manifest↔`.als` path divergence, then re-export

**MAJOR FINDING (2026-07-19, committed as `19bf25f`):** the divergence is
**near-total, not a handful of slots.** `scripts/diagnose_manifest_als_paths.py
--set-id 1fsnxchk --als <backups .als>` reports **290 of 296 non-silent clips
path-MISS** (only 6 match). The cause is a path-string convention drift
between `.als` clip paths (e.g. `4-154__…`, `cand1__…`) and manifest
`local_path`s — `resolve_identity` does an exact path match, so the prefix
difference breaks it for almost everything, and the `slot_id_map` fallback
carries the stale ids through. **This is why the attempted re-export
reproduced essentially all stale ids.** The diagnostic's candidate-matcher
pinpoints the right manifest track per missed clip — that list IS the
reconciliation work.

The re-export will STILL reproduce stale ids via the `slot_id_map` fallback
until the manifest's `local_path`s path-match the `.als` clip paths again.
This is John's live WIP (manifest edited today, many `.bak_reconcile_*`
backups). **Coordinate with John** — do not mutate his manifest further
without sign-off. The path reconciliation is likely a path-normalization
fix (strip the Ableton track-number prefix from clip paths, or add it to
manifest local_paths) — but confirm the canonical form with John before
applying en masse. Once reconciled:
```
venvs/audio/bin/python -m labeling.export_als_to_gt --als <canonical> --set-dir <setdir>
venvs/audio/bin/python scripts/audit_gt_recording_ids.py --set-id 1fsnxchk   # expect 0 real mismatches
venvs/audio/bin/python scripts/build_work_map.py --set-id 1fsnxchk
# then re-score + re-render the gallery (website was running as of 2026-07-19)
```
The audit's name-matcher has **false positives** on same-song-different-format
names (e.g. "Weak (Filtered Vocals)" vs "AJR - Weak", "Ke$ha" vs "Kesha",
"Galantis - You" exact-match). The REAL mismatches to watch are the
cross-artist ones (slot 003 Emily→PCH, slot 028 Beatles→Garrix, slot 031
CCR→Killers) and the `tlp*` ids not in DB (slots 024w1, 026w1, 028w1).

### Step D — Repeat for BB11 (`2nvzlh2k`) after BB12 is green.

## Hard constraints (AGENTS.md)

- **Do not `mv`/`rm` the originals** — copy only. Keep the `_backups` BB12
  `.als` as the safety backup.
- **Do not touch the canonical DB** (pi-storage). 0 work gaps; nothing to merge.
- **Do not overwrite the committed fixture** until Step C's audit is clean.
- **Do not mutate John's live manifest** without sign-off (it's WIP).
- `make check` must pass before any push. Branch `trm-ablation-framework` is
  behind origin by 28 — a push needs a rebase (coordinate; don't force-push).
- Work in a worktree if doing substantive work (AGENTS.md §1); the shared tree
  is dirty with other agents' WIP (vast/gpubox/contrast/flywheel/agentic).

## Files to know

- `scripts/resolve_manifest_recording_ids.py` — manifest rid resolver (dry-run + --apply).
- `scripts/diagnose_manifest_als_paths.py` — **the path-mismatch diagnostic (THE unblock for Step C)**. Run: `PYTHONPATH=. venvs/audio/bin/python scripts/diagnose_manifest_als_paths.py --set-id 1fsnxchk --als <backups .als>`. Finding: 290/296 BB12 clips path-miss.
- `scripts/audit_gt_recording_ids.py` — fixture↔DB name audit (operator-run, needs SSH).
- `scripts/build_work_map.py` — builds `labeling/fixtures/id_maps/<set>_work.json`.
- `scripts/reconcile_works.py` — DB work-merge candidate report (found 0; dry-run).
- `labeling/export_als_to_gt.py` — the exporter (DEFAULT_ALS is stale; Step B).
- `tests/labeling/test_export_capture_fidelity.py` — the capture-fidelity fence.
- `alignment/score_timeline_vs_gt.py` — the scorer.
- `eda/alignment/spectrogram_review/` — the review website (was running).

## Quick orientation commands for the next agent

```
git log --oneline -4   # see the 3 commits this session
venvs/audio/bin/python scripts/audit_gt_recording_ids.py --set-id 1fsnxchk   # current mismatch count
venvs/audio/bin/python scripts/resolve_manifest_recording_ids.py --set-id 1fsnxchk   # dry-run
ls ~/aligning/_backups/20260616_150150/big\ bootie\ 12\ labeling\ Project/   # the .als source
ls ~/aligning/2nvzlh2k__Two\ Friends\ -\ Big\ Bootie\ Mix\ Episode\ 11/BB11\ align\ Project/   # BB11 (already canonical)
```
