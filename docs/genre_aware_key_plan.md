# Genre-aware key extraction — implementation plan

**Status:** queued. Not started.
**Owner:** unassigned.
**Estimated effort:** 2–3 hours focused work + ~12 min Vast backfill.

## Why

`KeyExtractor` today runs hardcoded with `profile='edma'` for every track.
EDMA is purpose-built for electronic dance music; it biases toward minor
keys and the chromagram weights it uses are tuned for repeated synthesizer
patterns. On the ~5% of corpus tracks that aren't EDM (vocal hip-hop
samples, downtempo, jazzy house, classical interpolations, etc.) it
mis-labels keys and reports inflated `key_strength` confidence.

For an ML labeler trained on these features, that's noise we can remove.

## How — variant C ("hierarchical")

```
                     ┌──────────────────────────────────┐
audio  ──────────────┤  Run discogs_effnet (already in  │
                     │  pipeline; output currently      │
                     │  thrown away)                    │
                     └─────────┬────────────────────────┘
                               │ top-K genres + probs
                               ▼
                  ┌────────────────────────┐
                  │ P(electronic) > 0.7?   │
                  └────────────┬───────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
     KeyExtractor(profile=             KeyExtractor across
        'edma')                        {edma, temperley, krumhansl}
        — fast happy path              pick winner by key_strength
                │                             │
                └──────────────┬──────────────┘
                               ▼
                       persist key + profile
```

**Why hierarchical, not pure ensemble:** ~95% of the corpus is EDM; running
3 KeyExtractors on every track is wasteful when one is right by default.
Reserve the multi-profile path for the genuinely ambiguous tracks.

## Genre → profile mapping (draft)

| Discogs top-level genre | Profile | Reasoning |
|---|---|---|
| Electronic-* (House, Techno, Trance, DnB, IDM, Ambient…) | `edma` | designed for EDM |
| Hip Hop / Rap | `edma` | similar tonal character — looped progressions, minor-bias |
| Pop-Dance, Synthpop | `edma` | club-derived |
| Rock, Pop (other) | `temperley` or `noland` | well-tempered tonal music |
| Classical | `temperley` | designed for it |
| Jazz | `krumhansl` | not ideal but better than edma's electronic bias |
| Folk | `temperley` |  |
| **Default** | `edma` | corpus bias |

Open question: validate this empirically against Spotify API key estimates
on a 100-track random sample once implementation is done. The mapping is
the easy part to iterate on.

## Implementation steps

### 1. Capture genre output (10 min)

Extend `EssentiaFeatures` dataclass at
[audio_pipeline/analysis/models.py:115](../audio_pipeline/analysis/models.py#L115):

```python
genre_top1: str | None              # Discogs taxonomy string e.g. "Electronic-Techno"
genre_top1_prob: float | None       # P(top-1 genre) ∈ [0, 1]
genre_top5_json: str | None         # JSON tuple of (label, prob) for top 5
key_profile: str                    # already exists; semantics broadens to "the profile we picked"
key_alternatives_json: str | None   # JSON {profile: {key_pc, key_mode, key_strength}} when multi-run
```

### 2. Essentia worker changes (30–60 min)

In whichever file runs `KeyExtractor` and `discogs_effnet`
(`audio_pipeline/analysis/adapters/essentia_*.py` — find via grep):

- Always run discogs_effnet (already do this; just preserve output).
- Read top-1 genre and `P(top1)`.
- If `P(electronic) > 0.7` (sum across all Electronic-* tags): run `edma`
  alone. Set `key_alternatives_json = None`.
- Else: run `edma`, `temperley`, `krumhansl`. Compute per-profile
  normalized strength (need to handle profiles' different natural scales —
  see "Caveats" below). Pick winner.
- Set `key_profile` field to the *chosen* profile name.

### 3. Schema changes (5 min)

Add to [web_crawler/database/schema.sql](../web_crawler/database/schema.sql)
under the `track_audio_features` table block:

```sql
genre_top1            TEXT,
genre_top1_prob       REAL,
genre_top5_json       TEXT,
key_alternatives_json TEXT,
```

`key_strength` column already exists (added 2026-05-06). `key_profile` is
already in `confidence_json` blob — promote it to a real column too while
we're here:

```sql
key_profile           TEXT,
```

Then ALTER on canonical and on Vast scratch.db (idempotent — skip if
column exists). Pattern from previous ALTERs:

```bash
ssh pi-storage 'sqlite3 /mnt/storage/data/db/music_database.db "
  ALTER TABLE track_audio_features ADD COLUMN genre_top1 TEXT;
  ALTER TABLE track_audio_features ADD COLUMN genre_top1_prob REAL;
  ALTER TABLE track_audio_features ADD COLUMN genre_top5_json TEXT;
  ALTER TABLE track_audio_features ADD COLUMN key_alternatives_json TEXT;
  ALTER TABLE track_audio_features ADD COLUMN key_profile TEXT;
"'
ssh vast 'sqlite3 /workspace/scratch.db "<same five ALTERs>"'
```

### 4. Update `_write_essentia_row` (10 min)

In [audio_pipeline/adapters/db.py](../audio_pipeline/adapters/db.py),
extend the INSERT and the ON CONFLICT update branch to include the five
new columns. Keep the JSON blobs in `confidence_json` too for one release
cycle (defense in depth in case schema migration misbehaves).

### 5. Backfill existing tracks (~12 min Vast time)

Write `scripts/backfill_essentia.py` (~50 lines):

```python
# For each track_audio_id that has an essentia_v2 row but key_profile IS NULL
# (or has key_profile='edma' from the v1 era):
#   1. ssh-rsync audio file from pi-storage to local /workspace/audio/
#   2. Call new Essentia worker (skip Demucs/MERT/cue-detr/beats)
#   3. Call persist_essentia_features() — already exists
#   4. Delete local audio file
# Loop until queue drained.
```

Cost: 24 tracks × ~30s Essentia time = ~12 min. ≈ $0.10 Vast spot.

Critical: must run on Vast (Essentia has no aarch64 wheels). Same
Tailscale SOCKS proxy + `pi-storage` SSH alias the existing
`scripts/vast_loop.py` uses.

### 6. Validation (~30 min)

Spotify API cross-check on a 100-track random sample:

```python
# For 100 random tracks where we have both:
#   - new genre-aware key estimate
#   - a spotify URL in dj_set_track_media_links
# Hit Spotify Web API audio-features endpoint, compare key + mode.
# Report:
#   - agreement rate (matches Spotify exactly)
#   - off-by-fifth rate (close-relative key confusion, common error)
#   - mode-flipped rate (major↔minor flip, common Krumhansl error)
# Compare against pre-change edma-only baseline (re-extract the same
# tracks both ways for fair comparison).
```

If genre-aware lifts agreement by >5pp, ship it. If it's within noise,
investigate whether the genre mapping is wrong before discarding.

## Caveats

- **Profile normalization.** Profiles return `key_strength` on different
  natural scales — `edma` consistently reports higher strengths than
  `krumhansl` even on identical chromagrams. Naive `max(key_strength)`
  picks `edma` essentially every time. **Fix:** normalize each profile's
  output by its empirical mean strength on a held-out sample (compute
  once, store as constants). Or: rank-normalize across the 3 results.
- **Discogs Effnet taxonomy is messy.** "Electronic" includes 80+
  sub-genres. Compute `P(electronic)` as the SUM across all
  Electronic-prefixed tags, not just the top-1 tag's probability.
- **Re-runs of Essentia are non-deterministic** for the TF heads (model
  initialization seeds). Keep this in mind when comparing v1 vs v2
  results — pure SP results (KeyExtractor, BPM via signal processing)
  ARE deterministic.
- **`key_profile` name change is a breaking change** for anything that
  reads `confidence_json.key_profile`. Grep for `key_profile` before
  shipping. (Currently nothing relies on it.)

## Decisions to confirm before starting

1. **Mapping table** above — fine as drafted, or want to investigate
   different profiles per genre?
2. **Validation strategy** — Spotify API cross-check (quick), manual
   labeling (gold), or just ship it?
3. **`key_profile` promotion to column** — bundle with this PR or skip?
4. **Backfill vs version-tag** — backfill the 24 existing tracks
   (recommended), or keep them at `edma_v1` and version-tag forward?

## Files touched

- `audio_pipeline/analysis/models.py` — extend `EssentiaFeatures`
- `audio_pipeline/analysis/adapters/essentia_*.py` — multi-profile logic
- `audio_pipeline/adapters/db.py` — extend `_write_essentia_row`
- `web_crawler/database/schema.sql` — 5 new columns
- `scripts/backfill_essentia.py` — new file (~50 lines)
- `docs/genre_aware_key_plan.md` — this file (update with results)

## After this lands

Possible follow-ups (not part of this work item):
- Use detected genre as an ML training feature directly (not just for
  key selection)
- Add per-section genre detection (genre can shift across a DJ mix —
  intro vs drop vs breakdown)
- Compare Essentia's `KeyExtractor` against [librosa's chromagram +
  Krumhansl correlation](https://librosa.org/) as an alternative
  signal-processing baseline
