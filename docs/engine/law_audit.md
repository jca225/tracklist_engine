# §16 System-Law Audit — current repo vs the provenance-first contract

> **As of 2026-07-25**, branch `provenance-engine-phase0`. Audits the current repo
> against the 21 executable system laws in
> [dj_engine_pseudocode.md](dj_engine_pseudocode.md) §16. Evidence gathered by five
> parallel audit passes; the load-bearing identity/coordinate claims
> (`dataset.py`/`infer.py`/`mert_model.py`) were re-verified by hand. This is the
> Phase 0 deliverable of the
> [convergence plan](../provenance_engine_convergence_plan.md) — the ranked,
> evidence-backed poison worklist.

> **Executable checker note (2026-07-25):** the checkable subset of these laws
> now runs as real predicates via `python -m core.provenance.laws --db <root>`
> ([core/provenance/laws.py](../../core/provenance/laws.py)). **Scope: it grades
> a provenance *substrate DB* (what the Brick 1–4 writers persisted), NOT the
> shipped pipeline — the verdicts below grade the shipped pipeline and still
> stand.** Checkable now: 1, 2, 3, 4, 5, 6, 7, **8, 9** (since Brick 6: §7
> human-label assertions imported from the GT fixtures via
> `python -m workspaces.alignment_prototype.provenance_gt`), 10, 11, **13, 14**
> (since Brick 7: §2/§8 versioning backbone + producer registry —
> [feature_store_and_registry.md](feature_store_and_registry.md); the grounded
> producers DB under `workspaces/alignment_prototype/out/provenance/producers/`
> passes both with real rows), 21; laws 12 and 15–20 report NOT_APPLICABLE
> until their entities exist in the substrate. The 8/9 VIOLATED rows below
> still grade the *shipped* GT writer (`write_back_ground_truth.py`), which
> remains delete-and-replace with bare point values; likewise the 13/14
> VIOLATED rows still grade the shipped training paths
> (`trajectory/train.py:601`, `scripts/train_identity_verifier.py`), which
> remain unversioned — Brick 7 builds the canonical path, it does not cut the
> live pipeline over.

## Tally

| Verdict | Count | Laws |
|---|---|---|
| **PASS** | 4 | 7, 18, 19, 20 |
| **PARTIAL** | 7 | 3, 4, 5, 6, 10, 15, 21 |
| **VIOLATED** | 9 | 1, 8, 9, 11, 12, 13, 14, 16, 17 |
| **N/A-NOT-BUILT** | 1 | 2 |

**Read:** the repo already honors the *discipline* laws it argued its way to
empirically — abstentions persist (7), eval is per-set-per-axis (18), n=2 is never
sold as corpus calibration (19), reconstruction never certifies identity (20). The
violations cluster in exactly the machinery that was never built: **provenance/artifact
lineage** (1,2,3,13,14), **append-only human history + uncertainty** (8,9), and
**calibrated posteriors + honest coordinates** (11,12). The identity poison (4,5) is
half-fixed — Crush cured the labeling side, the aligner side is still open.

## Summary table

| # | Law | Verdict | Closed by | One-line |
|---|---|---|---|---|
| 1 | provenance_is_complete | 🔴 VIOLATED | P1 | features/timelines/models written with no Run/Derivation record |
| 2 | provenance_is_acyclic | ⚪ N/A | P1→P3 | no derivation graph exists to be acyclic |
| 3 | all_artifacts_verify | 🟡 PARTIAL | P1 | sha columns exist but everything fetched/joined by mutable path |
| 4 | no_source_key_is_canonical_recording_id | 🟡 PARTIAL | P1 | `COALESCE(recording_id, track_id)` leaks the scrape key as identity |
| 5 | no_path_was_used_as_identity | 🟡 PARTIAL | P1 | labeling PASS (Crush); **aligner VIOLATED** (size-1 spine pool) |
| 6 | every_unknown_parser_row_has_diagnostic | 🟡 PARTIAL | P0→P1 | scrape keeps raw rows; tokenizer silently `continue`s unkeyable rows |
| 7 | all_abstentions_are_persisted | 🟢 PASS | — | abstains stored as NULL-`recording_id` rows in `set_ground_truth` |
| 8 | human_history_is_append_only | 🔴 VIOLATED | P1 | re-export `DELETE`s + `INSERT OR REPLACE`s GT in place |
| 9 | human_uncertainty_is_field_specific | 🔴 VIOLATED | P1 | GT stored as bare point values, zero uncertainty representation |
| 10 | identity_placement_structure_are_separate | 🟡 PARTIAL | P2 | computed/scored separately, but collapse into one span + one confidence |
| 11 | no_unknown_coordinate_equals_zero | 🔴 VIOLATED | P2 (+P1 schema) | unknown cue → real `0.0`; GT coords are `NOT NULL` |
| 12 | decoder_consumes_posteriors_not_raw_margins | 🔴 VIOLATED | P2 | Viterbi over raw cosine margins; calibration seam deferred |
| 13 | every_model_has_training_snapshot | 🔴 VIOLATED | P3 | heads train on live GT/MERT; no frozen TrainingSnapshot |
| 14 | every_model_has_code_and_environment | 🔴 VIOLATED | P3 (found. P1) | checkpoints save weights+args, no code sha / env / dep-lock |
| 15 | no_round_consumes_its_own_outputs | 🟡 PARTIAL | P3 | set-split proxy guards leakage; no round object / cycle guard |
| 16 | pseudo_label_graph_is_acyclic | 🔴 VIOLATED | P3 | pseudo-labels are a flat boolean flag, no id/parent/round |
| 17 | rejected_rounds_are_reproducible | 🔴 VIOLATED | P3 | no round entity; rejected attempts discarded |
| 18 | evaluation_is_per_set_and_per_axis | 🟢 PASS | — | scorer reports id/placement/structure per `--set-id`, per stem |
| 19 | not_claims_corpus_calibration_from_two_sets | 🟢 PASS | (gate P3) | docs+code label n=2 LOSO development-only throughout |
| 20 | reconstruction_does_not_certify_identity | 🟢 PASS | — | recon is a placement feature + stopping signal only |
| 21 | every_published_value_is_explainable | 🟡 PARTIAL | P3 | timelines carry input-provenance stamp; no per-axis `explain()` |

## What each phase turns green

- **Phase 0** (this): 6 (tokenizer parse diagnostics — begins here).
- **Phase 1 — identity/provenance slice = RT1:** 1, 3, 4, 5(aligner), 7(durable aligner store), 8, 9, 11(GT schema), 14(ProcessSpec foundation), 2(graph foundation). *This is where the poison dies.*
- **Phase 2 — placement/structure + decoder:** 10, 11(writer), 12; 18 already green.
- **Phase 3 — cotraining lineage + snapshots + gates:** 2(cycle guard), 13, 14, 15, 16, 17, 19(mechanical gate), 21.

## Per-law evidence

### 1 · provenance_is_complete — 🔴 VIOLATED → P1
- `web_crawler/database/schema.sql` — 40 tables, none is `run`/`artifact`/`derivation`; only `content_history` carries a sha.
- `analysis/persistence.py:52,103,127,145` — feature/analysis INSERTs with no `run_id`/input linkage.
- `core/db.py:548-597` `insert_audio` — rows keyed by `(recording_id, platform, player_id)` locators, no producing Run.
- `workspaces/alignment_prototype/infer.py:930` — timeline is `out_path.write_text(json.dumps(...))`, a bare file with no RunOutput edge.

### 2 · provenance_is_acyclic — ⚪ N/A-NOT-BUILT → P1→P3
- No `derivation`/`run_input`/`run_output` edges exist → no graph to be cyclic. Vacuous until the graph is built; hardens into §12 `validate_pseudo_label_for_next_round` in P3.

### 3 · all_artifacts_verify — 🟡 PARTIAL → P1
- `schema.sql:194,333` `track_audio.sha256`/`set_audio.sha256` exist (nullable, informational); `content_history:763-784` has content/payload shas.
- But addressing is by mutable path: `core/db.py:344,362,395,422` path loaders; `mert_store.py:55-60,116` MERT `.npz` by set-id path, no verify; `external/checkpoint.py:47` model `.pt` at a named path, no hash.

### 4 · no_source_key_is_canonical_recording_id — 🟡 PARTIAL → P1
- Model is correct in principle: `materialize.py:212` keeps `recording_id` and `track_id` separate; pseudocode `:442` forbids `Recording(recording_id=parsed["track_key"])`.
- **But** `infer.py:124` `SELECT ... COALESCE(recording_id, track_id) ...` (same fallback in `labeling/build_content_catalog.py`, `remap_gt_slot_labels.py`, `inventory_check.py`, `pull_set_for_alignment.py`) — an un-reconciled slot silently uses the 1001TL source key as canonical identity. **[verified by hand]**

### 5 · no_path_was_used_as_identity — 🟡 PARTIAL (labeling PASS · aligner VIOLATED) → P1
- **Labeling FIXED:** `labeling/content_resolver.py` binds by `(file_size,n)`/head-hash or abstains, "no filename/slot guessing"; `export_als_to_gt.py:180-190,283` ambiguity hard-abstain on `(recording_id, stem, variant)`. `slot_id_map` is DEAD in runtime (only in tests/docs/guardrails).
- **Aligner OPEN:** `dataset.py:41-52` `slot_candidates_from_targets` = one `(recording_id, stem)` per slot; `infer.py:204-218` `slot_pools_from_rows` = **size-1 pool** from `set_track_slots` keyed by `slot_label`; `mert_model.py:294` `pools = self.slot_pools or slot_candidates_from_targets(...)` → the model can only "select" the spine's claim. The real multi-candidate open-set pool lives in `evals/instance_separability.py` + `open_set_acappella_identity_findings.md` and is **not wired into `infer.py`**. **[verified by hand — matches decision #19 verbatim]**

### 6 · every_unknown_parser_row_has_diagnostic — 🟡 PARTIAL → P0→P1
- Scrape preserves every row: `web_crawler/scraper.py:262-273` writes all `tlTab` children to `dj_set_rows`; AJAX failures → `scrape_failures` (`:214-229`).
- **But** `tokenizer/materialize.py:280,288-289,296` bare `continue` past any unkeyable row — no diagnostic, no counter. Rvmor/sided partially rescued via synthetic `tlp{id}` (`:315`); residual `None` path emits nothing. No `observation`/`ParseDiagnostic` table.

### 7 · all_abstentions_are_persisted — 🟢 PASS → —
- `export_als_to_gt.py:241-269` content miss → `(None,…, "abstain")`, still emitted as a row; `write_back_ground_truth.py:99-107` writes NULL-`recording_id`/`id_source="abstain"` rows; `ground_truth/schema.py:104-110` makes `id_source: abstain` first-class; `content_resolver.py:104-141` abstains via explicit diagnostic. Minor aligner-side gap: `never_matched.py:22-23` `continue`s track_id-less rows (per-run JSON, not a durable table) → tighten in P1.

### 8 · human_history_is_append_only — 🔴 VIOLATED → P1
- `write_back_ground_truth.py:94-108` `BEGIN IMMEDIATE` → `DELETE FROM set_ground_truth WHERE set_id=?` → `INSERT OR REPLACE`; `schema.sql:649-669` PK `(set_id, label)`, no `supersedes`/`version`/`valid_from`. Append-only `content_history` exists for audio bytes but was never applied to GT.

### 9 · human_uncertainty_is_field_specific — 🔴 VIOLATED → P1
- `schema.sql:649-668` — `recording_id`/`set_start_s`/`tempo_ratio`/… carry no paired sigma; `write_back_ground_truth.py:66-82` row tuple has no uncertainty term; `rg uncertainty|student_t|sigma|posterior labeling/` = 0 hits. The only `confidence` is the *aligner's* (`harness/contract.py:54`), not the human's.

### 10 · identity_placement_structure_are_separate — 🟡 PARTIAL → P2
- Separated in sensor/routing (`harness/axes.py:22,41-59`) and scoring (`score_timeline_vs_gt.py:60-66` distinct `id_correct`/`place_err_s`/`ref_err_s`). But fused at the decision object: `infer.py:182-194` `SpanTarget` + span dict hold `recording_id`+`set_start_s`+`ref_segments` with one scalar `confidence` (`contract.py:49-56`). No per-axis `AxisBelief`.

### 11 · no_unknown_coordinate_equals_zero — 🔴 VIOLATED → P2 (+P1 schema)
- `infer.py:181` `start = r["cue_s"] if ... else 0.0`; `:189` `ref_start_s=0.0` seeded unconditionally; `schema.sql:654-656` coords are `NOT NULL` (GT can't represent unknown). Read-side scorer is correct (`score_timeline_vs_gt.py:60-66` `float | None`). **[verified by hand]**

### 12 · decoder_consumes_posteriors_not_raw_margins — 🔴 VIOLATED → P2
- `path_decode.py:209-212,255-261` decodes a piecewise-linear path over raw normalized matched-filter cosine curves; `joint_ref_decode.py` calls it directly. Fusion by axis priority not posterior (`harness/merge.py:29-36`). Calibration hook is an unbuilt squash (`harness/path_decode_probe.py:10-11,34`).

### 13 · every_model_has_training_snapshot — 🔴 VIOLATED → P3
- `cotrain.py:51-86` trains on live `load_set(yaml)` + `mert_store.load_bb12_mert(...)`; `trajectory/train.py:601` saves `{model, args}` only; `external/checkpoint.py:37-49` records counts, no snapshot id. A GT re-materialize silently invalidates a checkpoint.

### 14 · every_model_has_code_and_environment — 🔴 VIOLATED → P3 (foundation P1)
- `trajectory/train.py:601` payload `{model,args}` — no `code_commit`/`ParameterSet`/`EnvironmentSpec`; `scripts/train_identity_verifier.py:79` bare `pickle.dump`. No `ProcessSpec`/`EnvironmentSpec` anywhere. (The `infer.py:913` provenance stamp records `code_sha` for the *timeline*, not for any model.)

### 15 · no_round_consumes_its_own_outputs — 🟡 PARTIAL → P3
- `trajectory/train.py:84-88` refuses train==eval; `trajectory/pseudo_labels.py:34-35,70-72` only AUTO_COMMIT spans mint labels; `cotrain.py:126-128` LOSO excludes the held set. But no round object / round-level output-feedback guard; the §12 `ProvenanceCycle` check is unbuilt. Set-split is a coarse proxy, not the law.

### 16 · pseudo_label_graph_is_acyclic — 🔴 VIOLATED → P3
- `trajectory/pseudo_labels.py:137-141` provenance = two booleans (`pseudo_label`, `agentic_quality`), no id/parent/round; `pseudo_materialize.py` has drop counts, no lineage. No graph → acyclicity untrackable.

### 17 · rejected_rounds_are_reproducible — 🔴 VIOLATED → P3
- No `CoTrainingRound` entity/table; `cotrain.py:111-156` `run_loso` returns an in-memory dict, no seed/spec captured; rejected attempts discarded. Closest artifact is the accepted-timeline provenance stamp (`provenance.py:70-88`), not rejected rounds.

### 18 · evaluation_is_per_set_and_per_axis — 🟢 PASS → —
- `score_timeline_vs_gt.py:187-188,564-590` invoked per `--set-id`, prints identity/placement/ref-offset as separate blocks, ref-offset split per stem; `:592-627` structure per class+stem; `docs/alignment_status.md:44-46` "never a single scalar". Minor: `eval.py` `EvalReport` (LOSO path) collapses — but canonical `make scorecard` is fully per-set-per-axis.

### 19 · not_claims_corpus_calibration_from_two_sets — 🟢 PASS → (mechanical gate P3)
- `alignment_status.md:31-32`, `cotrain_loso_findings.md:66-72`, state-of-record #10/#6 all label n=2 LOSO development/directional; `cotrain.py:147` unseen-set decode = "honest floor". Enforced by convention; `gate_calibration_claim_is_honest` (§11) unbuilt but nothing makes the forbidden claim.

### 20 · reconstruction_does_not_certify_identity — 🟢 PASS → —
- `recon_probe.py:729-751,777,787` recon features perturb the ref offset (placement); `AUC(match→id-correct)` is an eval diagnostic only. Pseudocode `:1172-1177,991` codify recon as stopping/plausibility, never identity. `drivers/ml.py` consumes recon as a placement feature.

### 21 · every_published_value_is_explainable — 🟡 PARTIAL → P3
- `provenance.py:70-88` stamps code_sha + spine/idmap/gt hashes on each timeline; `:91-119` drift check; `make align-state` prints FRESH/STALE; `infer.py:474` per-span `start_source`. But no per-occurrence `explain()` returning axis posteriors / contributing emissions / decoder trace / pseudo-label ancestry (pseudocode `:1211-1252`). The stamp answers "what made this / is it current", not "why this value".
