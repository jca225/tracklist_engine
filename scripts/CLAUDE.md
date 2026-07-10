# scripts/ — operational entry points (cross-cutting)

Not a chain module — run-from-root CLI tools and shell launchers that drive the
pipeline stages. Each belongs *conceptually* to a stage (below); they live here
because they're operational entry points, not library code. Invoke from repo
root, usually with `venvs/audio/bin/python scripts/<x>.py`.

Layout: live tools stay **flat** (external raw URLs, systemd/deploy muscle
memory, and `from scripts.x import …` test imports all pin the flat paths);
the taxonomy lives in this index. Two subdirectories hold the rest:

- **[migrations/](migrations/)** — one-shot pi-storage DB migration SQL.
- **[attic/](attic/)** — completed one-off campaigns/rollouts, kept for the
  record (ledger table at the bottom). Do not re-run one without reading its
  row. Moved scripts use `parents[2]` for repo root.

> Corpus-empirics scripts that used to live here (`bb_popularity.py`,
> `aux_db_sync.py`, `bb_*.py`) moved to `eda/corpus_empirics/` (commit 4a2fe45).
> Don't recreate them here.

## Stage map (live, flat)

**Ingest — rescue / replace** (see [../ingest/CLAUDE.md](../ingest/CLAUDE.md)):
- `redownload_via_ytmusic.py` — re-source yt-dlp `track_audio` rows via YT Music search (the main rescue path; sends `full_name` so the remixer qualifier resolves the right release).
- `redownload_via_spotdl.py` — re-source yt-dlp rows via pooled spotdl.
- `rescue_common.py` — shared two-phase rescue library for the `redownload_via_*` pair (imported, not run).
- `replace_track_audio.py` — manually replace one track's audio (YouTube / YT Music /
  **SoundCloud** / Spotify URL, or local file). **Destructive** when replacing an
  existing row. SC-only scrape rows: `--url 'https://api.soundcloud.com/tracks/<id>'`.
  Backs the `replace-track-audio` skill.
- `replace_stem_audio.py` — replace a bad acappella/instrumental row by `--track-audio-id` + URL/file; logs `axis=stem`, runs fingerprint check.
- `mac_push_acquire.py` — Mac-side YT Music search + yt-dlp + scp + pi replace, for when pi yt-dlp hits bot detection but Mac works.
- `scan_wrong_versions.py` — corpus wrong-version scan (Topic original, live, wrong remix).
- `log_acquisition.py` — acquisition-case ledger writer (`core/acquisition_case.py` emit path).
- `backfill_bb12_cases.py` — rebuild acquisition-case records for BB12 (imported by `tests/test_acquisition_case.py`).

**Ingest — stems / variants / identity:**
- `acquire_variant.py` — acquire a vocal/instrumental variant (staging or canonical `track_audio` row).
- `ingest_stem_url.py` — **Mac URL-first driver**: SSH to pi (`acquire_variant` add or `replace_stem_audio` replace), optional `--pull`, `--fail-on`, `--file` scp fallback, `--skip-if-ingested` re-run guard (sha256 vs canonical). See [../docs/stem_discovery_playbook.md](../docs/stem_discovery_playbook.md).
- `fetch_candidate_stems.py` — pull candidate acappella/instrumental files into `~/aligning/.../candidates/` for audition.
- `candidate_vocal_gate.py` — HuBERT-L9 gate picking the best acappella candidate per slot (0.6 floor; can flip `claimed_stem`).
- `ingest_candidate_winners.py` — `stems/*/candidates/WINNER.txt` → canonical ingest (re-run-safe; `--force` re-ingests).
- `apply_stem_matches.py` — Discord `proposed_matches.csv` → `ingest_stem_url`; `--auto` applies metadata∧audio double-confirms unreviewed, `--review-out` = the human queue (re-run-safe; `--force` re-ingests). Adjudicate the queue by ear in Ableton via `workspaces/alignment_prototype/review/seed_stem_review_als.py` (seed CAND/CAT pairs, rename `ACC`/`REJ`, `--harvest` writes decisions back).
- `match_stem_library.py` — map staged stem-library files (Discord corpus) → recordings; `--verify` = HuBERT/chromaprint audio verify (GPU: `vast_stem_verify.sh`); decision bands `auto_accept`/`accept`/`review`/`abstain` (audio folds into the band).
- `discord_scrape.py` / `discord_grab.sh` — Discord stem-corpus retrieval (staging on pi; ToS-risk acknowledged in-file).
- `promote_identity_overrides.py` — `labeling/identity_overrides/<set>.yaml` → `set_track_slots.recording_id`.

**Labeling / GT loop** (see [../labeling/CLAUDE.md](../labeling/CLAUDE.md)):
- `aligning_refresh.py` — chain inline_tag + relink + fill_als after pull.
- `reconcile_gt_inventory.py` — GT YAML → inventory action CSV (dry-run); closes labeling→canonical loop.
- `correction_report.py` / `gt_ref_source_report.py` — ledger and GT ref_source analytics.
- `reconcile_orphans.py` — route disk orphans (no `track_audio.path`) into delete / register / promote; dry-run by default. Use **ASCII** punctuation in print paths (pi SSH locale). Do not re-run `--apply` after a completed pass without dry-run — see [../docs/agent_handoff_reconcile_20260530.md](../docs/agent_handoff_reconcile_20260530.md).

**Analysis** (MIR workers — see [../analysis/CLAUDE.md](../analysis/CLAUDE.md)):
- `mert_backfill_loop.py` — MERT-only 330M re-embed (no Demucs/beats); corpus-wide by default, optional `--set-ids`.
- `set_mert_backfill_loop.py` — set-side MERT measures backfill (requires `migrations/migrate_set_mert_measures.sql` on the target DB).
- `mac_analyze_loop.py` — Mac-MPS analysis loop (sibling of `vast_loop.py`). `--separator {demucs,uvr}`.
- `mac_analyze_sets.py` — one-shot beat_this + stem backend on full DJ-set mixes via Mac MPS. `--separator {demucs,uvr}`.
- `pi_analyze_set_beats.py` — CPU beat_this on set mixes (pi-storage side).
- `separate.py` — standalone single-file separation for QA / A-B (`uvr` | `demucs` | `both`), via the project adapters' Python API. Supersedes the old `sota_stems.py`. See [../analysis/CLAUDE.md](../analysis/CLAUDE.md) "Stem-separation backends".
- `render_set_stems.py` — render/export separated set stems for audition.
- `setup_separation.sh` / `setup_roformer_separation.sh` — host-specific separation-backend install (CPU vs CUDA; see `requirements-audio.txt`).
- `backfill_track_fingerprints.py` — corpus-wide landmark-fingerprint backfill (done; keep for new tracks).
- `cache_set_fingerprint_hits.py` — per-set fingerprint hit cache for the aligner.
- `cache_tracklist_boundaries.py` — cache scraped tracklist boundary times (info-dynamics + surprise probe input).
- `recognize_segment.py` / `recognize_sweep.py` — ACRCloud segment recognition (open-set identify + sweep driver).
- `info_dynamics_embed_set.py` / `info_dynamics_bb_batch.sh` — info-dynamics embedding per set + BB batch driver.

**Personalization** (see [../personalization/](../personalization/)):
- `build_taste_roster.py` / `merge_taste_roster.py` — SoundCloud listener-cohort roster build + merge (`personalization/config/mixes.yaml`).

**Dev / guardrails:**
- `guardrails.py` + `guardrails_ratchet.json` — stale-name/path/dead-flag checks + entropy ratchet baselines (`make check`, pre-commit, CI).
- `typecheck.sh` — mypy subset (`make check`, pre-commit, CI).

**Vast provisioning / GPU workers — ⚠️ DO NOT MOVE OR RENAME:**
- `vast_bootstrap.sh` — provisions an ephemeral Vast box.
- `vast_run.sh` — launches a Vast run (`vast_worker` + pi-storage sshfs).
- `vast_taste_embed.sh` — tail MERT embed (no pi-storage; label `taste-embed`).
- `vast_info_dynamics.sh` — info-dynamics sets: beats CPU + RoFormer/MERT CUDA (label `info-dynamics`; rent 4090 PyTorch template in UI first).
- `vast_stem_verify.sh` — GPU stem-library verify pass (`match_stem_library.py --verify`; label `stem-verify`).
- `vast_loop.py` — Vast-side analysis loop (drives `analysis.vast_worker`).

These are coupled to **external absolute paths** that a rename silently
breaks: `vast_run.sh` and the bootstrap are fetched by **GitHub raw URL**
(`https://raw.githubusercontent.com/jca225/tracklist_engine/main/scripts/...`),
and `vast_loop.py` self-references `/workspace/tracklist_engine/scripts/vast_loop.py`
on the deployed box. If you must relocate them, update the raw URLs and the
`/workspace` path in lockstep and re-test a fresh Vast bootstrap.

## migrations/ — one-shot pi-storage DB migration SQL

Applied over SSH per the rollout checklists (root [CLAUDE.md](../CLAUDE.md),
[../docs/identity_and_inventory_plan.md](../docs/identity_and_inventory_plan.md)).
All have been applied to the canonical DB unless a plan doc says otherwise;
re-running is not idempotent for the renames — check before applying.

`migrate_identity_axes.sql` · `migrate_phase4_recording.sql` ·
`migrate_layer_role.sql` · `migrate_set_ground_truth_p1.sql` ·
`migrate_set_mert_measures.sql` · `migrate_slot_satisfaction_view.sql`

## attic/ — completed one-offs (ledger)

| script | what it was | status |
|---|---|---|
| `fix_bb11_lmct_013w1.sh` | BB11 slot 013w1 acappella-candidate ingest + folder refresh | Done; single-slot fix. |
| `reconcile_pass1_manual.sh` | pass-1 manual delete list for dup clusters | **Executed 2026-05-30 — do NOT re-run** ([handoff](../docs/agent_handoff_reconcile_20260530.md)). |
| `deploy_inventory_coherence.sh` + `backfill_layer_role.py` | slot-inventory coherence rollout (layer_role column + backfill) | Rolled out; `make check-inventory` is the live gate. |
| `ingest_bb12_winners.py` | batch-ingest BB12 Ableton candidate winners (P1.5) | Campaign done (BB12 GT complete). |
| `upgrade_ytmusic_nonoriginal.py` | re-source ~239 unvalidated `hits[0]` YT Music refs with the version gate | Campaign done. The 106 wrong-version suspects thread (preview-clip) may resurrect it. |
| `mac_redownload_bb_remix.py` / `mac_redownload_tracklist.py` / `collect_redownload_failures.py` / `retry_redownload_track_ids.py` | Mac-side BB remix / tracklist redownload campaign + failure collection/retry | Campaign done; the live rescue path is `redownload_via_ytmusic.py`. |
| `murph_taste_enrich_driver.sh` / `murph_taste_post_enrich.sh` | Murph taste-roster enrichment drivers | Campaign done (roster merged). |
| `vast_synthetic_pretrain.sh` | Vast driver for synthetic MERT pretrain | Experiment **closed** (flat on BB12 — see `workspaces/alignment_prototype/attic/EXPERIMENTS.md`). |
