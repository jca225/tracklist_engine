# Handoff plan: extract the Ableton `.als` ↔ structured-data layer into an `als/` codec sub-package

**Audience:** another agent picking this up fresh. **Status:** plan only — NOT started.
**Author context:** drafted 2026-07-02 while diagnosing the BB11 GT identity bug (below).

## Goal (one sentence)

Consolidate the scattered Ableton-`.als` parsing / identity-resolution / writing logic
(currently `labeling/als_io.py`, 927 LOC / 39 public symbols, + ~9 consumers) into a single
cohesive **`als/` codec sub-package** with a clean public API and a **round-trip law
(`parse ∘ print = id`)** as the verification pillar — treating `.als ↔ structured data` as a
bidirectional grammar (source ↔ AST), per the `project_als_grammar_roundtrip` memory
(Law A: BB12 152/152 clips round-trip).

## ⚠️ READ FIRST — coordination (this is the biggest risk)

**A parallel agent is actively working in the `.als` space** (BB11 micro-pitch detune,
`project_bb11_master_tempo_export` / `TempoArrangementMapper`, `seed_als_from_timeline`). This
refactor moves `als_io.py`, which they are likely editing. **Do NOT start a big move while
that workstream is live.** Before any file move:
1. `git pull`, scan `git log`/memory for recent `.als` work; confirm the als space is quiet.
2. Coordinate with the human — get an explicit "als space is clear" before moving files.
3. Prefer a **new package that re-exports** (additive) over an in-place rename, so the old
   import paths keep working during transition (`labeling/als_io.py` becomes a thin shim
   `from als import *` until consumers are migrated). This de-risks collision massively.
4. Use the **`refactor-safety` skill** (`.claude/skills/refactor-safety/`) — it's built for
   exactly this (inventory stale refs, fix `Path(__file__).parents[N]` depth, deploy
   entrypoints, `make check`, handoff docs).

## Current surface (inventory, 2026-07-02)

**Core:** `labeling/als_io.py` — 927 LOC, 39 defs/classes. Groups naturally into three concerns:

- **read** — `load_als_xml`, `parse_master_tempo`, `ArrangementMapper`,
  `TempoArrangementMapper`, `select_arrangement_mapper`, `parse_layer_clips`,
  `clip_original_path`, `track_display_name`, `audible_span`, `clip_gain_breakpoints`,
  `audible_from_curve`, `split_clip_at_mix_span_edges`, `_find_mix_splice_beat`
- **identity** — `build_manifest_index`, `match_manifest_for_path`, `resolve_identity`,
  `strip_user_tags`, `classify_path`, `display_from_path`, `slot_from_path`,
  `_normalize_path`, `_stem_folder_name`, `_filename_stem_marker`, `labels_overlap`
- **write** — `write_tempo_envelope`, `write_locators`, `build_vol_envelopes`,
  `volume_automation_id`, `envelope_value`
- **models** (dataclasses) — `WarpMarkers`, `MixClipSpan`, `AudibleSpan`, `ParsedClip`,
  `ManifestSlot`, `ManifestIndex`
- **utils** — `tempo_ratio`, `normalize_stem_value`

**Consumers** (must keep working — update imports):
`labeling/export_als_to_gt.py`, `labeling/enrich_gt_track_ids.py`, `labeling/als_path_audit.py`,
`labeling/relink_als_after_tag.py`, `alignment/seed_als_from_timeline.py`,
`alignment/transition_probe.py`, `.../seed_tempo_test.py`,
`scripts/attic/ingest_bb12_winners.py`, `scripts/fetch_candidate_stems.py`. Also the tag→relink→fill
trio (`inline_tag_aligning_folder.py`, `relink_als_after_tag.py`, `fill_als_clip_tags.py`).

## Target structure (proposed — the agent may refine)

```
als/                       # NEW sub-package. Location DECISION below.
  __init__.py              # curated public API (re-export the stable surface only)
  models.py                # ParsedClip, ManifestSlot/Index, WarpMarkers, MixClipSpan, AudibleSpan
  read.py                  # load_als_xml + clip/tempo/arrangement/audible/gain parsing
  identity.py              # manifest index, match_manifest_for_path, resolve_identity, path classify
  tags.py                  # strip_user_tags, _BRACKET_TAG, tag-aware path normalization
  write.py                 # tempo envelope, locators, volume envelopes (seeding primitives)
  roundtrip.py             # parse∘print=id law + anchor_check harness (the verification pillar)
```

**Location DECISION (leave to the agent + human):** `labeling/als/` (scoped under the
labeling stage — simplest, no top-level justification needed) vs. a top-level `als/`
(it's cross-cutting: consumed by `workspaces/` + `scripts/` too, which argues for top-level —
but the root CLAUDE.md requires explicit justification for new top-level folders). Recommend
**`labeling/als/`** unless the cross-cutting consumption is judged to warrant top-level.

## Fold in the BB11 robustness fix (do this AS PART of the extraction)

The extraction is the right moment to fix a real bug found 2026-07-02
(`project_bb11_identity_export_bug`): `match_manifest_for_path` does EXACT path matching, but
the annotator's `[NNNbpm KK]` rename tags make clip paths differ from manifest paths →
`track_id: None` on all rows (BB11 GT export: 0/127 matched). The fix: make the matcher
**tag-insensitive** — apply `strip_user_tags` to BOTH the clip filename and the manifest
filename before comparison (machinery already exists, just unused in the matcher). Add a
regression test: a tagged clip path must resolve to its un-tagged manifest row. This makes GT
production robust for **every** annotator-tagged set, not just BB11. (Part 2 of that bug — the
stale tlp-id manifest — is a data re-pull, NOT code; out of scope here but note it; check
whether `enrich_gt_track_ids.py` already backfills canonical ids.)

## Migration steps (for the agent)

1. **Coordinate + pull** (see ⚠ above). Confirm als space quiet. Branch.
2. **Create `als/` package**, move symbols by concern (read/identity/write/models/tags). Keep
   private helpers private. Curate `__init__.py` — export only the stable public surface.
3. **Make `labeling/als_io.py` a shim** (`from als import *  # noqa` + deprecation note) so no
   consumer breaks mid-migration. Migrate consumers' imports incrementally, then drop the shim.
4. **Fix `Path(__file__).parents[N]`** depths in moved files (refactor-safety skill catches these).
5. **Fold in the tag-insensitive matcher fix** + regression test.
6. **Verification pillar:** implement/relocate the round-trip law — `parse ∘ print = id` on
   BB12's `.als` (152/152 clips) + `labeling/anchor_check` (YAML vs fresh re-export). This is
   the acceptance gate: the refactor is correct iff the round-trip still holds bit-for-bit on
   the GT-relevant fields.
7. **`make check`** (guardrails + pytest) green before push. Update `labeling/CLAUDE.md` +
   root CLAUDE.md module index to point at `als/`.
8. **Handoff doc** if any pi-storage / deploy entrypoint or the parallel agent's files are touched.

## Acceptance criteria

- All 9 consumers import from `als/` and run unchanged in behavior.
- Round-trip law green on BB12 (152/152) + `anchor_check` passes.
- Tag-insensitive matcher: BB11 GT re-export populates `track_id` (0/127 → matches; note the
  *tlp-vs-canonical* join still needs Part 2's re-pull to fully validate identity downstream).
- `make check` green. No new top-level folder without justification recorded.

## Related

`project_als_grammar_roundtrip` (the parse∘print=id law), `project_bb11_identity_export_bug`
(the robustness fix to fold in), `project_bb11_master_tempo_export` (`TempoArrangementMapper` —
parallel-agent-active), `labeling/CLAUDE.md` (annotator `[NNNbpm KK]` rename convention),
`.claude/skills/refactor-safety/`.
