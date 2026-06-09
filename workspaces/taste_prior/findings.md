# Taste prior findings (SoundCloud cohort)

## Warehouse (`data/taste/taste_warehouse.db`)

| Mix | Listeners | Likes | Comments | Playlists | Bots flagged |
|-----|----------:|------:|---------:|----------:|-------------:|
| BB11 `2nvzlh2k` | 21,155 | 833,106 | 1,589 | 13,984 | 674 (3.2%) |
| BB12 `1fsnxchk` | 115,551 | 1,416,833 | 690 | 16,811 | 17 (0.01%) |

Per listener: **likes** (`sc_likes`) + **playlist track lists** (`sc_playlists.track_ids_json`).

## Analysis completed (2026-06-09)

### Taste clusters (non-bot, ≥15 tracks)
| Mix | Users clustered | Largest clusters |
|-----|----------------:|------------------|
| BB11 | 2,822 | 1027, 950, 451, 312 |
| BB12 | **4,447** | **2845**, 814, 779 |

Algorithm: `mbk_track_v1` — sparse binary vectors, top 3k tracks vocabulary, k=12.

### Comment heatmaps vs GT
| Mix | Artifact | GT alignment |
|-----|----------|--------------|
| BB11 | `data/analysis/bb11_comment_heatmap.json` | — |
| BB12 | `data/analysis/bb12_comment_heatmap.json` | F1 ≈ 0.18 (15/154 GT starts matched) |

Pipeline summaries: `data/analysis/{mix_id}_pipeline_summary.json`

### MERT user priors (deferred)

Blocked until **aligner pretrain** completes — user taste priors come after synthetic
pretrain → GT fine-tune, not before. Existing v0 cache: 38 SC tracks, 29 BB11 priors
(from exploratory run; do not extend until pretrain gate clears).

Run only when ready:
`prior-mert --mix …` or `run-analysis --with-mert`

## One-shot pipeline

```bash
# Full analysis (bots → cluster → MERT → heatmap)
venvs/audio/bin/python -m workspaces.taste_prior.main run-analysis \
  --mix 2nvzlh2k --out-dir data/analysis

# BB12 heatmap vs Ableton GT
venvs/audio/bin/python -m workspaces.taste_prior.main comment-heatmap \
  --mix 1fsnxchk --gt labeling/fixtures/bb12_ground_truth.yaml \
  --set-id 1fsnxchk --out data/analysis/bb12_comment_heatmap.json
```

## pi-worker ops

Code must be pushed or rsync'd before `make deploy-worker`. Then:

```bash
make install-taste-scrape && make restart-taste-scrape
make logs-taste-scrape
```

Import BB11 archive on worker (copy `Archive/.../bb11` or rsync from Mac first).
