# taste_prior — per-user subjective taste prior (workspaces)

Incubates the **personalization layer** — SoundCloud listener cohorts, timestamped
likes, mix comments with playhead position, playlist graphs, MERT user priors —
distinct from the aligner chain. Spec: [docs/taste_prior_plan.md](../../docs/taste_prior_plan.md).

**Runs on pi-worker** (`tracklist-taste-scrape.service`). Mac dev: `venvs/audio/bin/python`.
Data: `data/taste/` (gitignored).

## Commands

```bash
# Init / status
venvs/audio/bin/python -m workspaces.taste_prior.main init-db
venvs/audio/bin/python -m workspaces.taste_prior.main status

# Import legacy Archive (BB11)
venvs/audio/bin/python -m workspaces.taste_prior.main import-archive \
  --mix 2nvzlh2k --archive-dir "/path/to/.../data/raw/bb11"

# Live SC scrape (pi-worker loop or Mac one-shot)
venvs/audio/bin/python -m workspaces.taste_prior.main collect --mix 1fsnxchk
venvs/audio/bin/python -m workspaces.taste_prior.main enrich --mix 2nvzlh2k --batch 20
venvs/audio/bin/python -m workspaces.taste_prior.main enrich-playlists --mix 2nvzlh2k --batch 20
venvs/audio/bin/python -m workspaces.taste_prior.main loop --all-mixes --once

# Analysis
venvs/audio/bin/python -m workspaces.taste_prior.main score-bots --mix 2nvzlh2k
venvs/audio/bin/python -m workspaces.taste_prior.main cluster --mix 2nvzlh2k
venvs/audio/bin/python -m workspaces.taste_prior.main prior-mert \
  --mix 2nvzlh2k --max-tracks 150 --max-users 100 --device mps
venvs/audio/bin/python -m workspaces.taste_prior.main comment-heatmap \
  --mix 1fsnxchk --gt labeling/fixtures/bb12_ground_truth.yaml --set-id 1fsnxchk \
  --out data/analysis/bb12_comment_heatmap.json
venvs/audio/bin/python -m workspaces.taste_prior.main run-analysis --mix 2nvzlh2k

# pi-worker systemd
make install-taste-scrape && make restart-taste-scrape && make logs-taste-scrape
```

Findings: [findings.md](findings.md)

## Promote criteria

Move to top-level `personalization/` when: corpus join works, MERT priors stable, export API defined.
