# Design & plan docs index

Standing design/plan/research docs that are **not** dated snapshots and not
superseded — kept live and browsable here so they stay reachable (the docs GC
treats anything linked from this index as LIVE). Dated handoffs/bearings/reviews
are *not* listed here; they age out to [archive/](archive/) via
`scripts/docs_gc.py`. Canonical live docs (state-of-record, status,
recharacterization, objective, architecture) are roots in their own right and
are intentionally omitted below.

Add a doc here when it's a durable design/plan worth keeping; remove + `git mv`
to `archive/` when it's fully executed, superseded, or refuted (log refutations
in `workspaces/alignment_prototype/attic/EXPERIMENTS.md`).

## Engine architecture — provenance-first rewrite

- [engine/dj_engine_pseudocode.md](engine/dj_engine_pseudocode.md) — **canonical architectural contract** for the Provenance-First DJ Alignment Engine (§0 fundamental rules, §16 executable system laws, §11 promotion gates). A behavioral contract the repo is checked *against*; not directly executable.
- [engine/dj_engine_design_notes.md](engine/dj_engine_design_notes.md) — the design's own framing notes (three DAGs, the honest fork at analyze, the anti-degeneration motive + typing/validation stack).
- [provenance_engine_convergence_plan.md](provenance_engine_convergence_plan.md) — Point A→B migration toward the contract: strangler-fig, not big-bang; Phase 1 identity/provenance slice = RT1; Crush B–E subsumed, nothing stranded.

## Alignment — placement & structure (the walls)

- [dj_set_alignment_math.md](dj_set_alignment_math.md) — the constituent-alignment math model: a set as two independently aligned lanes (mix↔constituent instrumental, mix↔constituent vocals); foundational formalism for placement/identity.
- [acappella_warp_decode_plan.md](acappella_warp_decode_plan.md) — why widening the warp grid *regresses*; lever is placement + learned model, not a bigger DP grid.
- [fine_placement_plan.md](fine_placement_plan.md) — sub-bar placement via chroma/fingerprint fusion + banded DTW over the coarse window; foundational placement research.
- [loop_tracing_research_brief.md](loop_tracing_research_brief.md) — acappella loop/jump tracing; the multiseg self-similarity wall (~34% of GT loss). Open problem.
- [reconstruction_supervision_plan.md](reconstruction_supervision_plan.md) — mix-reconstruction error as a label-free correctness signal; Step 2 lives as an ML training feature.

## Alignment — probes & agentic harness

- [auditory_probes_plan.md](auditory_probes_plan.md) — DSP placement probes + belief layer + the validation-gate template; design of the harness now built in `agentic/`.
- [pomdp_agentic_aligner_design.md](pomdp_agentic_aligner_design.md) — POMDP framing of the aligner (belief-update loop, graded autonomy); increments 1–3 built, 4–5 (LLM-as-policy) designed.
- [editable_reconstruction.md](editable_reconstruction.md) — fingerprint-banded NMF to recover placement / gain curves / EQ; validated method feeding reconstruction-supervision.

## Ingest / acquisition / analysis quality

- [acquisition_as_decision_model.md](acquisition_as_decision_model.md) — acquisition as `CaseClaim`/`Attempt`/`Resolution`; corrections-ledger → checks wiring for the pseudo-label flywheel.
- [genre_aware_key_plan.md](genre_aware_key_plan.md) — hierarchical genre → multi-profile KeyExtractor to de-bias key extraction on non-EDM; improves ML feature quality.
- [stem_quality_ranking_plan.md](stem_quality_ranking_plan.md) — stem-candidate quality ranking (Q-phases); Q1 verdict = no scorer beats the prior, Q2 human labels mandatory.
- [mojibake_locale_fix_runbook.md](mojibake_locale_fix_runbook.md) — issue #74 `track_audio.path` double-encoding: code defenses (shipped) + the ordered pi UTF-8-locale fix → row-repair runbook (staged for a coordinated deploy).

## Benchmark & paper

- [benchmark_certification_research.md](benchmark_certification_research.md) — certify the 48k-set corpus *without listening* (round-trip metamorphic + confident-learning + capture-recapture).
- [corpus_rigor_related_work.md](corpus_rigor_related_work.md) — living bibliography for the alignment paper (prior art, weak supervision, scale-without-GT).
- [unmixdb_benchmark_plan.md](unmixdb_benchmark_plan.md) — external eval harness + NMF baseline + quad-fingerprint adapter vs André-2024 on UnmixDB.

## Infra / operations

- [durable_compute_buffer_spec.md](durable_compute_buffer_spec.md) — S3 manifest buffer + integrity-gated reconciler decoupling GPU compute from pi uptime. DEFERRED (Phase 1 post-Aug 1).

## Audit / open fronts

- [opinion_audit.md](opinion_audit.md) — census of silent "guillotine" gates that discard strong evidence; priority-ordered fix ledger (race-board validated).

## North-north star — product, generation, taste (lab-side)

- [startup_strategy.md](startup_strategy.md) — product shape (workspace + instant app), copyright wedge, positioning.
- [tracklist_data_engine_plan.md](tracklist_data_engine_plan.md) — plain-English 4-phase data-engine framing (pantry → chef → kitchen loop → drop the tracklist).
- [dj_selection_model.md](dj_selection_model.md) — info-dynamics model of *why* a DJ picked a track; downstream of alignment.
- [dj_craft_rules.md](dj_craft_rules.md) — practitioner rules (harmonic Camelot moves, phrasing, levels, acappella craft) grounding the mashup decision model.
- [mashup_decision_model_plan.md](mashup_decision_model_plan.md) — three-stage mashup taste model: hand rules → pretrain → post-train with verb logging.
- [papers_to_levers.md](papers_to_levers.md) — maps 9 papers to the three mashup decision-model stages, with ranked experiments.
