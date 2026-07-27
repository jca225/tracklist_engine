# scripts/ — DAG / cluster / gate entry points

Flat operational CLIs. Invoke from repo root with `venvs/audio/bin/python scripts/<x>.py`
(or bash for shell wrappers). Subdirs: `migrations/` (pi DB SQL), `attic/` (retired one-offs).

## Keep list (focused DAG)

**Gate / hygiene**
- `guardrails.py` + `guardrails_ratchet.json`
- `entropy_audit.py` + `entropy_ratchet.json`
- `typecheck.sh`, `docs_gc.py`, `merge_train.py`

**Analysis loops**
- `mac_analyze_loop.py`, `mac_analyze_sets.py`
- `render_set_stems.py`, `separate.py`
- `setup_separation.sh`, `setup_roformer_separation.sh`, `download_msst_models.py`
- `loop_hardening.py`, `loop_prefetch.py`

**Ingest surgery**
- `replace_track_audio.py`, `replace_stem_audio.py`, `acquire_variant.py`
- `redownload_via_ytmusic.py`, `rescue_common.py`

**Cluster**
- `pi_autopull.sh`, `gpubox_agentic_both.py`

**Aligner support**
- `aligning_refresh.py`, `align_state.py`, `build_bridge_id_map.py`
- `cache_set_fingerprint_hits.py`, `backfill_track_fingerprints.py`

Retired scripts live in git history (and optionally `~/tracklist_attic/`). Do not
reintroduce personalization / info-dynamics / discord / recognize_* drivers here.
