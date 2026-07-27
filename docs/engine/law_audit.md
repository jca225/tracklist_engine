# §16 System-Law Audit — current repo vs the provenance-first contract

> **As of 2026-07-26 @ `c62e9f5` (Day 7 refresh).** Prior stamp was 2026-07-25
> on `provenance-engine-phase0`. This pass re-grades only the laws the 7-day
> plan asked to watch (1, 3, 4, 5-aligner, 7, 8, 9, 11-schema) plus the Phase 2
> cluster that actually moved (10, 11-writer, 12). Evidence below; hot-path vs
> off-path substrate are graded separately where they diverge.

> **Executable checker note (2026-07-25, still true):** the checkable subset of
> these laws runs as real predicates via `python -m core.provenance.laws --db
> <root>` ([core/provenance/laws.py](../../core/provenance/laws.py)). **Scope:
> it grades a provenance *substrate DB*, NOT the shipped pipeline** — the
> verdicts below grade the shipped / default-on path unless marked off-path.
> Checkable now: 1, 2, 3, 4, 5, 6, 7, **8, 9**, 10, 11, **13, 14**, 21; laws 12
> and 15–20 report NOT_APPLICABLE until their entities exist in the substrate.
> Fold-ins C+E (PR #112) keep the producer registry append-only and
> kind-consistency self-maintaining.

## Tally

| Verdict | Count | Laws |
|---|---|---|
| **PASS** | 4 | 7, 18, 19, 20 |
| **PARTIAL** | 9 | 3, 4, 5, 6, 10, **11**, **12**, 15, 21 |
| **VIOLATED** | 7 | 1, 8, 9, 13, 14, 16, 17 |
| **N/A-NOT-BUILT** | 1 | 2 |

**Day 7 flips vs 2026-07-25:** **11** and **12** move VIOLATED → PARTIAL
(off-path Phase 2 belief→timeline emits `None` / calibrated posteriors;
default-on `infer.py` / `path_decode` still violate). **1, 3, 4, 5-aligner, 7,
8, 9** do **not** flip this week — RT1/Crush already accounted in the prior
PARTIAL/PASS rows; Phase 1B stays default-off so 5-aligner remains PARTIAL
(pool exists, hot path still size-1). No law goes fully green from Phase 2
alone — that matches the plan's honesty clause.

**Read:** discipline laws still hold (7, 18, 19, 20). Violations remaining are
lineage/snapshots (1, 13, 14, 16, 17), append-only human GT (8, 9), and the
**hot-path** cutover gap (10–12 still PARTIAL until dual-read). Identity poison
(4, 5) is half-fixed — Crush labeling PASS; aligner pool wired but default-off.

## Summary table

| # | Law | Verdict | Closed by | One-line |
|---|---|---|---|---|
| 1 | provenance_is_complete | 🔴 VIOLATED | P1 | features/timelines/models written with no Run/Derivation on the hot path |
| 2 | provenance_is_acyclic | ⚪ N/A | P1→P3 | no derivation graph exists to be acyclic |
| 3 | all_artifacts_verify | 🟡 PARTIAL | P1 | sha columns + content-addressed provenance store; hot path still joins by mutable path |
| 4 | no_source_key_is_canonical_recording_id | 🟡 PARTIAL | P1 | `COALESCE(recording_id, track_id)` leaks the scrape key as identity |
| 5 | no_path_was_used_as_identity | 🟡 PARTIAL | P1 | labeling PASS (Crush); aligner pool built (1B) but default-off — hot path size-1 |
| 6 | every_unknown_parser_row_has_diagnostic | 🟡 PARTIAL | P0→P1 | scrape keeps raw rows; tokenizer silently `continue`s unkeyable rows |
| 7 | all_abstentions_are_persisted | 🟢 PASS | — | GT abstains + Phase 2 belief path persists `None` / empty posteriors |
| 8 | human_history_is_append_only | 🔴 VIOLATED | P1 | re-export `DELETE`s + `INSERT OR REPLACE`s GT in place |
| 9 | human_uncertainty_is_field_specific | 🔴 VIOLATED | P1 | GT stored as bare point values, zero uncertainty representation |
| 10 | identity_placement_structure_are_separate | 🟡 PARTIAL | P2 | off-path `AxisBelief` chains exist; hot `infer.py` still one span + one confidence |
| 11 | no_unknown_coordinate_equals_zero | 🟡 PARTIAL | P2 (+P1 schema) | off-path decoder emits `None`; GT schema + hot `infer.py` still seed `0.0` |
| 12 | decoder_consumes_posteriors_not_raw_margins | 🟡 PARTIAL | P2 | off-path belief timeline consumes calibrated posteriors; hot path still raw margins |
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
- **Phase 2 — placement/structure + decoder:** 10, 11(writer), 12; 18 already green. *Off-path substrate now PARTIAL on 10–12; hot-path cutover still open.*
- **Phase 3 — cotraining lineage + snapshots + gates:** 2(cycle guard), 13, 14, 15, 16, 17, 19(mechanical gate), 21.

## Per-law evidence

### 1 · provenance_is_complete — 🔴 VIOLATED → P1
- `web_crawler/database/schema.sql` — 40 tables, none is `run`/`artifact`/`derivation`; only `content_history` carries a sha.
- `analysis/persistence.py:52,103,127,145` — feature/analysis INSERTs with no `run_id`/input linkage.
- `core/db.py:548-597` `insert_audio` — rows keyed by `(recording_id, platform, player_id)` locators, no producing Run.
- `workspaces/alignment_prototype/infer.py:930` — timeline is `out_path.write_text(json.dumps(...))`, a bare file with no RunOutput edge.
- *Off-path:* `core/provenance` + producers DB under `out/provenance/` persist Run/Artifact rows for the Phase 2 shadow lane — does not clear the hot-path violation.

### 2 · provenance_is_acyclic — ⚪ N/A-NOT-BUILT → P1→P3
- No `derivation`/`run_input`/`run_output` edges exist → no graph to be cyclic. Vacuous until the graph is built; hardens into §12 `validate_pseudo_label_for_next_round` in P3.

### 3 · all_artifacts_verify — 🟡 PARTIAL → P1
- `schema.sql:194,333` `track_audio.sha256`/`set_audio.sha256` exist (nullable, informational); `content_history:763-784` has content/payload shas.
- But addressing is by mutable path: `core/db.py:344,362,395,422` path loaders; `mert_store.py:55-60,116` MERT `.npz` by set-id path, no verify; `external/checkpoint.py:47` model `.pt` at a named path, no hash.
- *Off-path:* content-addressed belief bundles / timeline lineage (PR #115) verify by digest when that seam is used.

### 4 · no_source_key_is_canonical_recording_id — 🟡 PARTIAL → P1
- Model is correct in principle: `materialize.py:212` keeps `recording_id` and `track_id` separate; pseudocode `:442` forbids `Recording(recording_id=parsed["track_key"])`.
- **But** `infer.py:124` `SELECT ... COALESCE(recording_id, track_id) ...` (same fallback in `labeling/build_content_catalog.py`, `remap_gt_slot_labels.py`, `inventory_check.py`, `pull_set_for_alignment.py`) — an un-reconciled slot silently uses the 1001TL source key as canonical identity. **[verified by hand]**

### 5 · no_path_was_used_as_identity — 🟡 PARTIAL (labeling PASS · aligner pool default-off) → P1
- **Labeling FIXED:** `labeling/content_resolver.py` binds by `(file_size,n)`/head-hash or abstains, "no filename/slot guessing"; `export_als_to_gt.py:180-190,283` ambiguity hard-abstain on `(recording_id, stem, variant)`. `slot_id_map` is DEAD in runtime (only in tests/docs/guardrails).
- **Aligner OPEN on hot path:** `dataset.py:41-52` / `infer.py` size-1 spine pool remains default. **Phase 1B** `candidate_pool.py` + `identity_override.py` exist but stay **default-off / fail-closed** (decision #23 defer) — not a PASS until the override is the live path.

### 6 · every_unknown_parser_row_has_diagnostic — 🟡 PARTIAL → P0→P1
- Scrape preserves every row: `web_crawler/scraper.py:262-273` writes all `tlTab` children to `dj_set_rows`; AJAX failures → `scrape_failures` (`:214-229`).
- **But** `tokenizer/materialize.py:280,288-289,296` bare `continue` past any unkeyable row — no diagnostic, no counter. Rvmor/sided partially rescued via synthetic `tlp{id}` (`:315`); residual `None` path emits nothing. No `observation`/`ParseDiagnostic` table.

### 7 · all_abstentions_are_persisted — 🟢 PASS → —
- `export_als_to_gt.py:241-269` content miss → `(None,…, "abstain")`, still emitted as a row; `write_back_ground_truth.py:99-107` writes NULL-`recording_id`/`id_source="abstain"` rows; `ground_truth/schema.py:104-110` makes `id_source: abstain` first-class; `content_resolver.py:104-141` abstains via explicit diagnostic.
- *Phase 2:* belief→timeline persists `offset_s=None`, empty posteriors, `chosen=None` on abstention (PR #114/#115) — strengthens the law on the shadow path.

### 8 · human_history_is_append_only — 🔴 VIOLATED → P1
- `write_back_ground_truth.py:94-108` `BEGIN IMMEDIATE` → `DELETE FROM set_ground_truth WHERE set_id=?` → `INSERT OR REPLACE`; `schema.sql:649-669` PK `(set_id, label)`, no `supersedes`/`version`/`valid_from`. Append-only `content_history` exists for audio bytes but was never applied to GT.

### 9 · human_uncertainty_is_field_specific — 🔴 VIOLATED → P1
- `schema.sql:649-668` — `recording_id`/`set_start_s`/`tempo_ratio`/… carry no paired sigma; `write_back_ground_truth.py:66-82` row tuple has no uncertainty term; `rg uncertainty|student_t|sigma|posterior labeling/` = 0 hits. The only `confidence` is the *aligner's* (`harness/contract.py:54`), not the human's.

### 10 · identity_placement_structure_are_separate — 🟡 PARTIAL → P2
- Separated in sensor/routing (`harness/axes.py:22,41-59`) and scoring (`score_timeline_vs_gt.py:60-66` distinct `id_correct`/`place_err_s`/`ref_err_s`).
- *Off-path PROGRESS:* `placement_structure_beliefs.py` + `belief_timeline.py` carry separate PLACEMENT/STRUCTURE `AxisBelief` chains (PR #114/#115).
- *Hot path still fused:* `infer.py` `SpanTarget` + span dict hold `recording_id`+`set_start_s`+`ref_segments` with one scalar `confidence`.

### 11 · no_unknown_coordinate_equals_zero — 🟡 PARTIAL → P2 (+P1 schema)
- *Off-path PROGRESS:* calibrated-belief decoder emits `None` coordinates on abstention; structure abstentions excluded from structure scoring (PR #115).
- *Still VIOLATED on hot path / GT schema:* `infer.py` seeds unknown cue / `ref_start_s` as `0.0`; `schema.sql` coords are `NOT NULL` (GT can't represent unknown). Read-side scorer already allows `float | None`.

### 12 · decoder_consumes_posteriors_not_raw_margins — 🟡 PARTIAL → P2
- *Off-path PROGRESS:* belief→timeline requires named calibration before probe confidence becomes a posterior; consumes `CalibratedAxisProbability`, not raw margins (PR #114/#115). LOSO producer is development-only (PR #116); structure from synthetic-only TRM closed (#26).
- *Hot path still VIOLATED:* `path_decode.py` / `joint_ref_decode.py` decode over raw matched-filter curves; fusion by axis priority not posterior.

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
- *Off-path:* AxisBelief records carry `calibration_id` / process stamps for the shadow lane — still not a full `explain()`.
