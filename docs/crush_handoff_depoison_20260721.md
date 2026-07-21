# Operation Crush — de-poisoning handoff (2026-07-21)

**For the next agent.** This is the Crush critical path: kill the GT id-poison so
BB11+BB12 can pass L0–L4 and clear the Rolling Thunder starting gate. The
branch-consolidation phase is done; the resolver foundation is landed. What
remains is wiring it and ripping out the guess-ladder. **A prior session's
correction is authoritative over the older Crush docs** — verify state yourself
before trusting any number below.

Fast pull picks this up: `main` @ `3cbfa4a` (or later). Everything referenced is
on `main` unless noted.

---

## Where this sits (one breath)

North star = SOTA aligner over ~40k sets. Sequence: **Crush (data integrity) →
Rolling Thunder (science on clean GT) → Scale**. Rolling Thunder is *blocked*
until Crush exits ([docs/operation_rolling_thunder_proposed.md](operation_rolling_thunder_proposed.md)).
**Crush exit = BB11+BB12 pass L0–L4 + pi read-back exact + one clean SSOT
re-measurement.** The de-poisoning below is the remaining gate.

---

## What already landed (do NOT redo)

- **Capture-fidelity fence** (#61/#62): GT derives from the `.als`; `gt_als_gate.py`
  is CI-enforced. yaml can't silently drift from Ableton anymore.
- **Content-identity foundation** (#56, MERGED): `labeling/content_resolver.py` —
  `resolve_clip_identity()`, `ContentCatalog.from_entries()`, `ClipIdentity`,
  `ResolveDiagnostic`. Identity comes from the active FileRef's
  `OriginalFileSize`/`OriginalCrc` (see `labeling/als/read.py::clip_content_identity`).
  **This is the tool that replaces the guess-ladder — it is landed but NOT yet wired.**
- **Branch divergence retired**: the ~124-file `trm-ablation-framework` backlog is
  reconciled — landed (#56/#65/#66/#67), folded into de-poisoning (PR-A, see #47
  comment), or dropped (vast).
- **spectrogram_review audit UI** (#67): an RT2 asset, pre-positioned.

## What is STILL poisoned (the job)

Verified on `main` @ `3cbfa4a`:

- `labeling/fixtures/id_maps/{1fsnxchk,2nvzlh2k}_slots.json` — **the `slot_id_map`
  poison carriers — still present.**
- **No GT row carries `id_source`** (`grep -r id_source labeling/` → empty).
- **The guess-ladder is still live and still the export fallback:**
  - `labeling/als/identity.py:85 match_manifest_for_path` (weak path tiers).
  - `labeling/export_als_to_gt.py:168-181` — when `recording_id is None`, falls back
    to `slot_id_map.get(slot_label)`. `_load_slot_id_map` @ `:483`, used @ `:512/:542`.
    **This is the exact line that carries a different song's id onto a GT row.**
- **3 poison ids still armed** (per prior-session correction): slot 028 Beatles→Garrix,
  031 CCR→Killers, 144 Snakehips→PCH. Cross-song id `2p25k23p` = issue **#63**.
- **Audio round-trip law uncarried**: #37 was closed as broken (m4a decode false-fails
  on py3.14); its denotational `.als→mix` validation is unreplaced.
- No renumber-metamorphic (C1b) green gate yet (`tests/test_alignment_metamorphic.py`
  exists — extend, don't recreate).

---

## The work — 4 steps, in order

Tracking issues: **#40** (robust GT capture), **#47** (BB12 stale ids / gated
re-export), **#50** (`.als` integrity protocol), **#51** (GT rollback snapshot),
**#63** (the `2p25k23p` collision).

### Step 1 — Wire the resolver → bind ids by content, add `id_source`
- Consume `resolve_clip_identity()` in the export path
  (`export_als_to_gt.py::_clip_row`), so each GT row's `recording_id` is bound from
  the clip's content identity (`OriginalFileSize`/`OriginalCrc`) against a
  `ContentCatalog` built from `track_audio` — **not** from slot/path.
- Stamp every row `id_source: content | abstain` (add to
  `labeling/ground_truth/schema.py` `RefSegment`). Content-bound → `content`;
  unresolvable → `abstain` (id left null, row flagged — do NOT guess).
- Acceptance: the 3 poison slots resolve to the *correct* song or `abstain`; none
  silently carry a cross-song id.

### Step 2 — Delete `slot_id_map` + weak tiers
- Remove the `slot_id_map` fallback from `export_als_to_gt.py` (:168-181, `_load_slot_id_map`,
  :512/:542) and delete `labeling/fixtures/id_maps/*_slots.json`.
- Remove / fail-close the weak tiers of `match_manifest_for_path`
  (`als/identity.py:85`). Keep only exact/content matches; everything else abstains.
- Acceptance: `grep -rn 'slot_id_map' --include=*.py . | grep -v attic/` is empty;
  callers (`enrich_gt_track_ids.py`, `als_path_audit.py`) still import cleanly.

### Step 3 — Re-carry the audio round-trip law
- Fix the m4a decode first (the reason #37 false-failed on py3.14), then re-add the
  denotational `.als → mix` round-trip check as a gate (sha256 stamp). It proves the
  exported GT reconstructs the mix the annotator heard.
- Acceptance: round-trip green on BB11+BB12; wired into `guardrails.py` like the
  `gt_als_gate`.

### Step 4 — C1b renumber-metamorphic green → clean SSOT re-measure
- Extend `tests/test_alignment_metamorphic.py`: renumbering slots must not change
  the content-bound identity (the property `slot_id_map` violated by construction).
- Then regenerate `docs/alignment_status.md` via `/align-checkpoint` on the
  **de-poisoned** GT — the first honest post-Crush numbers. **This is Crush exit.**

---

## Traps (read before touching code)

- **Take main's bb12, never trm's.** `trm-ablation-framework` carries an OLD-scheme
  `bb12_ground_truth.yaml` + a pre-#62 slot scheme in `als/identity.py`. Main's
  #62-derived GT + slot scheme is canonical. The prior PR-A merge attempt proved
  trm's als toolkit is entangled with the old scheme and **breaks `gt_als_gate`** —
  see the deferral + **salvage inventory** on
  [#47](https://github.com/jca225/tracklist_engine/issues/47). Reuse trm's *new*
  pieces (`gap_classify.py`, `_dedup_core`, the manifest-reconciliation scripts) by
  **re-basing them on the new scheme**, not copying.
- **Never hand-edit `docs/alignment_status.md`.** It's the regenerated SSOT — only
  `/align-checkpoint` writes numbers there. Do not type alignment numbers into other
  docs or memory; cite the canonical doc.
- **pi-storage is canonical.** `data/db/music_database.db` is a stale dev copy. Query
  `ssh pi-storage 'sqlite3 /mnt/storage/data/db/music_database.db "..."'` for real
  state; `track_audio.path` may have UTF-8/Latin-1 mojibake.
- **Snapshot before write-back** (#51): back up `set_ground_truth` + the yaml before
  any re-export; the user's prior bb12 hand-WIP is preserved at tag
  `wip/bb12-enrichment-backup`.
- **Abstain, don't guess.** The whole point: a null/abstained id is correct; a
  confidently-wrong cross-song id is the poison.

## Guardrails / how to verify

- `make check` (guardrails + fast pytest). Pre-commit runs `scripts/guardrails.py`
  (incl. `gt_als_gate`) + `typecheck.sh` (core) + full `pytest tests/`.
- Build worktrees off `main`; the pre-commit hook resolves `venvs/audio/bin/python`
  relative to the worktree root, so `ln -sfn <repo>/venvs venvs` in the worktree or
  the hook falls back to system python (missing lxml → false gate-import failure).
- Ratchets in `scripts/guardrails_ratchet.json` — deleting `slot_id_map` / weak tiers
  should *lower* counts; that's fine (baselines are ceilings).

## Pointers

- Plans: [operation_crush_assault_plan.md](operation_crush_assault_plan.md) (discrepancy
  register D1–D10), [operation_rolling_thunder_proposed.md](operation_rolling_thunder_proposed.md).
- Identity model: [core/identity.py](../core/identity.py), CLAUDE.md "Track identity (three axes)".
- Memory: `project_path_identity_root_cause` (2 diseases + DELETE slot_id_map + C1 test),
  `project_identity_by_string_bug_class`, `feedback_no_seeded_labeling`,
  `feedback_als_ref_parsing_unescape`.
- Session that produced this handoff: landed #56/#65/#66/#67, deferred PR-A/PR-C into
  this work, dropped vast. Branch/worktree state clean; `crush-content-identity`
  worktree is merged-and-removable.

## First concrete moves

1. `git pull`; confirm `main` tip; skim #47 (salvage inventory) + #40.
2. Reproduce the poison: export BB12 and show slot 028/031/144 carrying cross-song ids
   (the failing state you are fixing).
3. Start Step 1 in a worktree off `main` (`crush/depoison-content-binding`).
