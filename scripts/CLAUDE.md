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
- `replace_track_audio.py` — manually replace one track's audio (URL / local file). **Destructive** (deletes old row + cascades). Backs the `replace-track-audio` skill.
- `acquire_variant.py` — acquire a vocal/instrumental variant from a URL into a staging folder (v1); v2 (planned) writes a canonical `track_audio` row reusing `replace_track_audio.py`'s write path.

**Analysis** (MIR workers — see [../analysis/CLAUDE.md](../analysis/CLAUDE.md)):
- `mac_analyze_loop.py` — Mac-MPS analysis loop (sibling of `vast_loop.py`).
- `mac_analyze_sets.py` — one-shot beat_this + Demucs on full DJ-set mixes via Mac MPS.
- `sota_stems.py` — SOTA 2-stem separation (Mel-Band RoFormer + MDX23C-InstVoc HQ) via the `audio-separator` CLI; beats `mix − vocals` subtraction.

**Vast provisioning / GPU workers — ⚠️ DO NOT MOVE OR RENAME:**
- `vast_bootstrap.sh` — provisions an ephemeral Vast box.
- `vast_run.sh` — launches a Vast run.
- `vast_loop.py` — Vast-side analysis loop (drives `analysis.vast_worker`).

These three are coupled to **external absolute paths** that a rename silently
breaks: `vast_run.sh` and the bootstrap are fetched by **GitHub raw URL**
(`https://raw.githubusercontent.com/jca225/tracklist_engine/main/scripts/...`),
and `vast_loop.py` self-references `/workspace/tracklist_engine/scripts/vast_loop.py`
on the deployed box. If you must relocate them, update the raw URLs and the
`/workspace` path in lockstep and re-test a fresh Vast bootstrap.
