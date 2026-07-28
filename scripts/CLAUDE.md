# scripts/ — DAG / cluster / gate entry points

Flat operational CLIs. Invoke from repo root with `venvs/audio/bin/python scripts/<x>.py`
(or bash for shell wrappers). Subdir: `migrations/` (pi DB SQL). Retired one-offs
moved to [archive/scripts/](../archive/scripts/) — see [archive/README.md](../archive/README.md).

**Everything below has a live binding** — a Makefile target, CI step, git hook,
systemd unit, Python import, test, or Claude skill — with two exceptions noted
under "No static reference, still live". Audited 2026-07-28; nothing else here
is dead.

> When auditing this dir, grep **both** forms: `foo.py` *and* `scripts.foo`.
> Filename-only greps miss `from scripts.foo import ...` and will wrongly
> report tested modules (e.g. `loop_prefetch`) as unused.

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

## No static reference, still live (do NOT archive)

- `aligning_refresh.py` — chains the `labeling/prep` audition pipeline
  (`inline_tag` → `relink_als` → `fill_als_clip_tags`). Human-invoked, so no
  code references it.
- `build_bridge_id_map.py` — under active modification on in-flight branches.

Retired scripts live in [archive/scripts/](../archive/scripts/) and in git
history. Do not reintroduce personalization / info-dynamics / discord /
recognize_* drivers here.
