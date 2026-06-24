# scripts/ — operational entry points (cross-cutting)

Not a chain module — a flat bag of run-from-root CLI tools and shell launchers
that drive the pipeline stages. Each belongs *conceptually* to a stage (below);
they live here because they're operational entry points, not library code.
Invoke from repo root, usually with `venvs/audio/bin/python scripts/<x>.py`.

> Corpus-empirics scripts that used to live here (`bb_popularity.py`,
> `aux_db_sync.py`, `bb_*.py`) moved to `eda/corpus_empirics/` (commit 4a2fe45).
> Don't recreate them here.

## Stage map

**Ingest** (download / acquisition — see [../ingest/CLAUDE.md](../ingest/CLAUDE.md)):
- `redownload_via_ytmusic.py` — re-source yt-dlp `track_audio` rows via YT Music search (the main rescue path; sends `full_name` so the remixer qualifier resolves the right release).
- `redownload_via_spotdl.py` — re-source yt-dlp rows via pooled spotdl.
- `replace_track_audio.py` — manually replace one track's audio (YouTube / YT Music /
  **SoundCloud** / Spotify URL, or local file). **Destructive** when replacing an
  existing row. SC-only scrape rows: `--url 'https://api.soundcloud.com/tracks/<id>'`.
  Backs the `replace-track-audio` skill.
- `acquire_variant.py` — acquire a vocal/instrumental variant (staging or canonical `track_audio` row).
- `replace_stem_audio.py` — replace a bad acappella/instrumental row by `--track-audio-id` + URL/file; logs `axis=stem`, runs fingerprint check.
- `ingest_stem_url.py` — **Mac URL-first driver**: SSH to pi (`acquire_variant` add or `replace_stem_audio` replace), optional `--pull`, `--fail-on`, `--file` scp fallback. See [../docs/stem_discovery_playbook.md](../docs/stem_discovery_playbook.md).
- `reconcile_gt_inventory.py` — GT YAML → inventory action CSV (dry-run); closes labeling→canonical loop.
- `apply_stem_matches.py` — reviewed Discord `proposed_matches.csv` → `ingest_stem_url`.
- `ingest_candidate_winners.py` — `stems/*/candidates/WINNER.txt` → canonical ingest.
- `promote_identity_overrides.py` — `labeling/identity_overrides/<set>.yaml` → `set_track_slots.recording_id`.
- `scan_wrong_versions.py` — corpus wrong-version scan (Topic original, live, wrong remix).
- `aligning_refresh.py` — chain inline_tag + relink + fill_als after pull.
- `correction_report.py` / `gt_ref_source_report.py` — ledger and GT ref_source analytics.
- `reconcile_orphans.py` — route disk orphans (no `track_audio.path`) into delete / register / promote; dry-run by default. Use **ASCII** punctuation in print paths (pi SSH locale). Do not re-run `--apply` after a completed pass without dry-run — see [../docs/agent_handoff_reconcile_20260530.md](../docs/agent_handoff_reconcile_20260530.md).
- `reconcile_pass1_manual.sh` — pre-apply manual delete list for dup clusters (run before bulk `--apply` when needed).
- `migrate_identity_axes.sql` / `migrate_phase4_recording.sql` — pi-storage DB column renames + `work`/`recording`/`set_ground_truth` (run once after deploy; then `tokenizer.materialize`).

**Analysis** (MIR workers — see [../analysis/CLAUDE.md](../analysis/CLAUDE.md)):
- `mert_backfill_loop.py` — MERT-only 330M re-embed (no Demucs/beats); corpus-wide by default, optional `--set-ids`.
- `mac_analyze_loop.py` — Mac-MPS analysis loop (sibling of `vast_loop.py`). `--separator {demucs,uvr}`.
- `mac_analyze_sets.py` — one-shot beat_this + stem backend on full DJ-set mixes via Mac MPS. `--separator {demucs,uvr}`.
- `separate.py` — standalone single-file separation for QA / A-B (`uvr` | `demucs` | `both`), via the project adapters' Python API. Supersedes the old `sota_stems.py`. See [../analysis/CLAUDE.md](../analysis/CLAUDE.md) "Stem-separation backends".

**Vast provisioning / GPU workers — ⚠️ DO NOT MOVE OR RENAME:**
- `vast_bootstrap.sh` — provisions an ephemeral Vast box.
- `vast_run.sh` — launches a Vast run (`vast_worker` + pi-storage sshfs).
- `vast_taste_embed.sh` — tail MERT embed (no pi-storage; label `taste-embed`).
- `vast_info_dynamics.sh` — info-dynamics sets: beats CPU + RoFormer/MERT CUDA (label `info-dynamics`; rent 4090 PyTorch template in UI first).
- `vast_loop.py` — Vast-side analysis loop (drives `analysis.vast_worker`).

These three are coupled to **external absolute paths** that a rename silently
breaks: `vast_run.sh` and the bootstrap are fetched by **GitHub raw URL**
(`https://raw.githubusercontent.com/jca225/tracklist_engine/main/scripts/...`),
and `vast_loop.py` self-references `/workspace/tracklist_engine/scripts/vast_loop.py`
on the deployed box. If you must relocate them, update the raw URLs and the
`/workspace` path in lockstep and re-test a fresh Vast bootstrap.
