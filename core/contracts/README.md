# core/contracts — artifact inventory + laws (A0)

Typed records for every artifact that crosses a stage boundary. Rationale,
failure-class evidence, and phased rollout: [docs/entropy_reduction_plan.md](../../docs/entropy_reduction_plan.md)
(workstream A). Serialization: **msgspec** Structs (frozen; strictness ratchets
up as writers migrate).

## Laws

1. One `load()` per artifact — validates and **normalizes at the boundary**
   (`normalize_slot_label`, id branding). Consumers never re-normalize.
2. Every set-scoped artifact carries `set_id`; combining two goes through
   `join_guard` (the BB11-scored-against-BB12-GT class becomes a loud error).
3. No silent defaults; loaders raise with field-level detail.
4. Raw `json.load`/`yaml.safe_load` of these artifacts outside this package is
   a ratcheted guardrail class (`scripts/guardrails_ratchet.json`).

## Inventory (artifact × writer × readers × known drift)

| artifact | writer | readers | drift history |
|---|---|---|---|
| `out/<set>_predicted_timeline*.json` | `alignment_prototype/infer.py` (+`joint_ref_decode` mutates) | `score_timeline_vs_gt` ✅migrated, `failure_analysis/build_span_table`, `seed_als_from_timeline`, `render_review_snippets`, agentic loop | stale `claimed_stem` (pre-f678f3a); scored against wrong set's GT (43c24a6) |
| `labeling/fixtures/*_ground_truth.yaml` | `labeling/export_als_to_gt.py` | typed loader EXISTS: `labeling/ground_truth/schema.py` (Result-based). Scorer previously bypassed it with raw yaml | legacy `version_tag`; track_id=None export bug; loop-seconds correction |
| `~/aligning/<set>/manifest.json` | `labeling/pull_set_for_alignment.py` (sole sanctioned writer) | `infer._manifest_by_tid`, `joint_ref_decode`, `labeling/als` identity, `enrich_gt_track_ids`, scorer (fiber ref audio) | `slot_label` None-drift (43c24a6); bed rows silently dropped; stale after re-stems (0960565) |
| slot spine (`set_track_slots` via ssh) | tokenizer materialize | `infer.fetch_slot_rows`, pull `fetch_tracks`, inventory_check | claimed_stem row-text drop (f678f3a); tlp↔recording namespace |
| `labeling/fixtures/id_maps/<set>.json` | enrich/export tooling | scorer, `infer._manifest_by_tid` | to be subsumed by an `entity_ids` crosswalk table (plan A3) |

## Migration state

- **Done:** `PredictedTimeline` (+ `load_timeline`), id NewTypes +
  `normalize_slot_label`, `join_guard`; `score_timeline_vs_gt` loads through
  contracts and join-guards timeline×GT.
- **Next (A2):** `TimeMap` + typed time domains in `core/timebase.py`.
- **Next (A2/A3):** `Manifest` record (bed rows emitted audio-less), slot-spine
  record shared by infer+pull, GT loader relocation from `labeling/ground_truth`.
