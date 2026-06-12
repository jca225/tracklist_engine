# personalization — per-user subjective taste prior

The **personalization layer** — SoundCloud listener cohorts, timestamped likes,
mix comments with playhead position, playlist graphs, MERT user priors — distinct
from the aligner chain. **Promoted out of `workspaces/taste_prior/` (2026-06-12)**
as the producer side of the step-2 generation-pretrain boundary; the consumer
(HRM-Text training) becomes its own repo once the export schema freezes. Specs:
[docs/taste_prior_plan.md](../docs/taste_prior_plan.md),
[docs/personalization_export_contract.md](../docs/personalization_export_contract.md).

**Runs on pi-worker** (`tracklist-taste-scrape.service`). Mac dev: `venvs/audio/bin/python`.
Data: `data/taste/` (gitignored).

## Commands

```bash
# Init / status
venvs/audio/bin/python -m personalization.main init-db
venvs/audio/bin/python -m personalization.main status

# Import legacy Archive (BB11)
venvs/audio/bin/python -m personalization.main import-archive \
  --mix 2nvzlh2k --archive-dir "/path/to/.../data/raw/bb11"

# Live SC scrape (pi-worker loop or Mac one-shot)
venvs/audio/bin/python -m personalization.main collect --mix 1fsnxchk
venvs/audio/bin/python -m personalization.main enrich --mix 2nvzlh2k --batch 20
venvs/audio/bin/python -m personalization.main enrich-playlists --mix 2nvzlh2k --batch 20
venvs/audio/bin/python -m personalization.main loop --all-mixes --once

# Analysis
venvs/audio/bin/python -m personalization.main score-bots --mix 2nvzlh2k
venvs/audio/bin/python -m personalization.main cluster --mix 2nvzlh2k
venvs/audio/bin/python -m personalization.main prior-mert \
  --mix 2nvzlh2k --max-tracks 150 --max-users 100 --device mps
venvs/audio/bin/python -m personalization.main comment-heatmap \
  --mix 1fsnxchk --gt labeling/fixtures/bb12_ground_truth.yaml --set-id 1fsnxchk \
  --out data/analysis/bb12_comment_heatmap.json
venvs/audio/bin/python -m personalization.main run-analysis --mix 2nvzlh2k

# pi-worker systemd
make install-taste-scrape && make restart-taste-scrape && make logs-taste-scrape
```

Findings: [findings.md](findings.md)

## Producer/consumer seam

This module is the **producer**: scrape → warehouse → tokenize → embed → export a
read-only `personalization_export.db` ([docs/personalization_export_contract.md](../docs/personalization_export_contract.md)).
The **consumer** (HRM-Text conditional pretrain) imports nothing from the chain and
lifts into its own repo when the `token_catalog` schema + export builder are stable.
