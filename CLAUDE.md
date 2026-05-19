# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tracklist Engine is a pipeline for analyzing recorded DJ mixes against the
tracklists scraped for them. The chain has three stages, each in its own
top-level module:

1. **scrape** — `web_crawler/` extracts DJ set metadata, track listings, and
   streaming links from 1001Tracklists.com.
   
Everything outside this chain is one of:
- A vendored dependency: `cue-detr/` (DETR-based cue-point detection model,
  consumed only by `audio_pipeline/analysis/canonical_cues.py`).
- Exploration / scratch: `data_analysis/` notebooks.
- Experimental forks of chain modules: `workspaces/` (e.g.
  `workspaces/alignment_workbench` is a fork of `browser_daw/`). Promote a
  fork out of `workspaces/` when it stabilizes (same pattern used for
  `ui/` → `browser_daw/`).
- Archive: `archive/` (e.g. the legacy Streamlit alignment-review app).

New features land inside one of the three chain modules. New top-level folders
require explicit justification.

## Key Commands

### Web Crawler
```bash
pip install -r requirements.txt
playwright install chromium
python web_crawler/main.py          # Run scraper (config-driven via config.yaml)
```

### CUE-DETR (cue point detection)
```bash
pip install -r cue-detr/requirements.txt
python cue-detr/cue_points.py -t /path/to/audio/dir   # Predict cue points
# Flags: -c <checkpoint_dir>, -s <sensitivity>, -r <min_distance>, -p (print)
```

### Data Analysis
Jupyter notebooks in `data_analysis/` — use `common.py` for shared DB access and DataFrame loading.

## Architecture

### Web Crawler (`web_crawler/`)
- **`main.py`** — Entry point. Loads DJ job files from `data/djs/*.json`, initializes DB, runs scraper.
- **`config.py`** — YAML config loader using dataclasses with a Result monad pattern for error handling.
- **`workers.py`** — Core scraping orchestration: page loads, captcha solving, AJAX media link fetching.
- **`scraper.py`** — HTML parsing: extracts set metadata, track info, media links from page content.
- **`database.py`** — SQLite interface. Schema lives in `web_crawler/database/schema.sql`.
- **`browser.py`** — Playwright browser context management with profile rotation.
- **`captcha_solver.py`** — Local CAPTCHA OCR via ddddocr (no API key, no network call). Optional `EmailCaptchaSolver` falls back to a human-in-the-loop email round-trip.
- **`data_models.py`** — Frozen dataclasses for type-safe immutable records (DJSet, DJSetMediaLink, etc.).

### CUE-DETR (`cue-detr/`)
DETR-based model for cue point detection in EDM tracks. Uses a custom COCO-like format with `position` instead of bounding boxes.
- `model/` — Training (`cue_detr_train.py` with W&B), evaluation, inference, data loading.
- `cue_points.py` — Main inference script. Downloads checkpoints from HuggingFace by default.
- Pretrained model: `disco-eth/cue-detr` on HuggingFace.

### Data Analysis (`data_analysis/`)
- `eda.ipynb`, `error_analysis.ipynb`, `tokenizer.ipynb` — Exploratory analysis notebooks.
- `common.py` — Shared utilities for DB queries and pydantic_ai agent integration.

### MERT embedding choice

We use `m-a-p/MERT-v1-95M` at **hidden layer 6** (not the final layer) for
both analysis and alignment paths. The MERT paper shows mid-layers transfer
best to music-ID / structural-matching tasks; the top of the stack is more
tagging-oriented and the bottom too acoustic. Constant lives in
[audio_pipeline/analysis/adapters/mert_adapter.py](audio_pipeline/analysis/adapters/mert_adapter.py)
as `MERT_DEFAULT_LAYER` and in [audio_pipeline/alignment/mert_align.py](audio_pipeline/alignment/mert_align.py)
as `DEFAULT_LAYER`. Keep them in sync — the alignment cache hashes the
layer into its key, so a divergence silently doubles the embedding cost.
When a learnable scoring head is added on top (post-ground-truth labeling),
replace the single-layer pick with a 13-channel learnable weighted sum
over all hidden states (SUPERB pattern) co-trained with the head.

## Corpus empirics

Empirical facts about the source corpus that constrain downstream modeling.

### Acapella/instrumental era choice is orthogonal in Big Bootie

Across Big Bootie Vols. 1–26 (Two Friends), the release year of the layered
acapella is statistically **independent** of the release year of the
instrumental host:

- n = 2,763 (instrumental_year, acapella_year) pairs (rows resolved to Spotify
  release dates)
- Pearson r = **−0.016**; Spearman r = 0.082
- Partial Pearson controlling for BB volume year: r = −0.032
- Shuffle-null 95% CI for Pearson: [−0.035, 0.037] — observed inside, two-sided
  p ≈ 0.40
- Gap (acap − instr year): mean = −4.25y, SD = 12.7y, median |gap| = 5y
- Era buckets: 24% have acapella >10y older than instrumental, 21% are same
  year (±1), the rest spread across all other gaps

**Within a mashup slot, year is independent — but the marginal distributions
of the two roles are very different:**

- corr(instrumental_year, BB_volume_year) = **0.357** — instrumentals are
  picked *fresh* (median instrumental ≈ within 0–2y of the volume's release)
- corr(acapella_year, BB_volume_year) = **0.101** — acapellas are pulled from
  a much wider historical window; per-volume SD is 9–12y vs. 3–7y for
  instrumentals
- Median per-volume gap (acap behind instr) grew from 1y (BB Vol. 5, 2014) to
  8y (BB24, 2024) — the "fresh beat, deep-catalog vocal" signature has
  intensified as the series aged
- This explains the −4.25y mean gap in the paired analysis: the *marginals*
  drift apart even though the *per-slot* pairing is independent

**Implication for modeling**: the mashup-sequence model (and any pair-scoring
head built from the aligned corpus) **must not condition on release-year
proximity** as a compatibility feature. Two Friends' aesthetic explicitly
mixes eras — classic vocals over modern beats and vice versa. Date-based
priors would learn an artifact, not the signal. Aesthetic / key / BPM / genre
alignment are the features that actually carry compatibility. If a date
prior is added later, model the two roles with **separate marginals**
(instrumental ≈ current year, acapella ≈ uniform over recent decades) rather
than a joint year-proximity term.

Reproduction: [scripts/bb_era_orthogonality.py](scripts/bb_era_orthogonality.py).
Cached release years: `data/analysis/spotify_release_dates.csv` (2902/2903
Spotify IDs resolved via unauthenticated `open.spotify.com/embed/track/{id}`).
Results: `data/analysis/bb_era_orthogonality.json`.

### Acapella choice IS driven by popularity (at-release + durable)

Era is a free parameter; popularity is the binding constraint on the acapella
side. Measured via two signals chosen specifically to avoid Spotify
popularity's "look-back bias" (recent streaming activity inflating long-tail
catalog):

- **Billboard Year-End Hot 100** (Wikipedia tables, 1958-2024) — was the track
  a US chart hit in its release year. Pure at-time-of-release signal.
- **Last.fm `track.getInfo`** (listeners + playcount) — cumulative scrobble
  footprint since ~2002. Community-skewed older/indie, so it captures cult /
  staying-power independent of streaming-era recency bias.

Headline numbers:

| | tracks resolved | Hot 100 year-end hit-rate | median Last.fm listeners |
|---|---|---|---|
| BB acapellas | 2,258 / 938 | **38.5%** | **261,134** |
| BB instrumentals | 759 / 310 | 12.9% | 1,254 |

Acapellas are **3× more likely to have charted Hot 100 year-end** in their
release year, and have **~200× more Last.fm listeners** than the
instrumentals. Effect-size on Last.fm listeners (Mann–Whitney rank-biserial)
= +0.43, a large effect with acapellas higher.

Four-quadrant split (chart hit × Last.fm ≥ 100k listeners):

| | hit+remembered | hit+forgotten | deepcut+remembered | deepcut+obscure |
|---|---|---|---|---|
| acapellas | 29% | 9% | **29%** | 32% |
| instrumentals | 7% | 4% | 10% | **79%** |

Two Friends pick **recognizable vocals from any era** (chart hit OR durable
listenership — both qualify) to layer over **EDM-world instrumentals** that
do not need to be famous outside dance music. 79% of BB instrumentals are
deep-cut + obscure to Last.fm users; 58% of BB acapellas are "remembered"
(≥100k Last.fm listeners) vs 17% of instrumentals.

**Implication for modeling**: the mashup-pair-scoring head should expect
a strong popularity asymmetry between the two roles. Treat the *vocal* side
as drawn from a high-popularity prior (chart hits + durable cult favorites
across decades); treat the *instrumental* side as drawn from a near-uniform
prior over EDM-genre-relevant tracks regardless of popularity. Don't use a
shared popularity feature; use role-conditional ones. Combined with the
era-orthogonality finding: era and popularity are the two independent axes
of acapella selection, while the instrumental is constrained on neither
popularity nor era-match.

Reproduction: [scripts/bb_popularity.py](scripts/bb_popularity.py).
Results: `data/analysis/bb_popularity.json`.

### Set popularity is driven by chart-hit-vocal *density*, NOT instrumental popularity

Per-volume YouTube view counts (n=23 BB volumes) correlated against
per-volume aggregates of track-level signals from aux.db.

**Music-only features** (no calendar-time leakage), univariate vs views:

| feature | r | r² | what it captures |
|---|---|---|---|
| `n_aca_charted` (raw count of Hot 100 hits in the vocal layer) | +0.47 | 0.22 | how many recognizable-hit vocals are stacked into the volume |
| `acap_chart_rate` (fraction of vocals charting Hot 100 year-end) | +0.51 | 0.26 | density of recognizable-hit vocals |
| `n_acapellas` (total count of vocals) | +0.47 | 0.22 | format maturity / vocal-layer density |
| `pct_aca_recent_3y` (vocals released within 3y of set) | −0.43 | 0.19 | freshness — but mostly collinear with format maturity |
| `mean_aca_lastfm_listeners` | ~+0.07 | ~0.00 | cumulative scrobble footprint — doesn't transfer |
| any instrumental feature (popularity or count) | ≤ \|0.18\| | ≤ 0.03 | instrumental popularity is neutral-to-negative |

**Best music-only multivariate fit**:

```
views ~ chart_rate + n_acapellas + n_aca_charted              R² = 0.39
+ pct_aca_recent_3y                                           R² ≈ 0.33–0.40 (marginal lift)
```

**~39% of set-views variance is explained by music-only features**, all
on the acapella side — specifically the *count* and *density* of charting
vocals in the volume, plus how many vocals are stacked in total (format
maturity).

**Instrumental popularity is uncorrelated** (mildly negative on some
signals) with set views. Pop-radio instrumentals dilute the BB formula
(unfamiliar EDM beat × familiar pop vocal); mainstream instrumentals
don't fit. Median instrumental Last.fm listeners is r ≈ −0.15 vs views.

**Cumulative Last.fm listeners on the vocal side doesn't transfer to
set views** (r ≈ +0.07). The signal that matters is **at-release-time
chart hit**, not durable scrobble footprint — listeners click for
immediate recognition ("oh, that's [current pop song]"), not for
"I know this from somewhere."

**What `set_year` does NOT do**:

`set_year` alone gives only **R² = 0.08**, and `chart_rate + set_year`
gives **R² = 0.26** — identical to `chart_rate` alone. What looked like
calendar-time / channel-growth signal is actually `n_acapellas` and
`n_aca_charted` in disguise: both correlate with set_year at **r ≥ 0.92**
because Two Friends added more vocals per slot as the format evolved.
Once those are named directly, `set_year` adds nothing. Same goes for
`pct_aca_recent_3y` (r = −0.81 with set_year): its univariate negative
correlation with views is mostly the same format-maturity confound.

**Implication for modeling**: if the downstream personalized-mix model
includes a "predicted listener engagement" head, the input features that
matter are (1) how many charting-hit vocals are stacked in (raw count),
(2) what fraction of vocals are charting hits (density), and (3) how
dense the vocal layer is overall. Instrumental popularity is
neutral-to-negative — the EDM-host should be optimized for compatibility
(key/BPM/genre/structure), not for popularity priors. Combined with the
prior two findings, the selection axes are:

- **Acapella**: high-popularity prior (chart hits drive engagement) ×
  era-uniform (no proximity term) × density matters (more is better)
- **Instrumental**: compatibility-driven (key/BPM/genre/structure) ×
  freshness-skewed (median 0-2y from set release) × popularity-neutral

**Unmeasured ~60% of variance**: NOT linear "channel growth" or "fandom
era" — those stories don't hold up once format-maturity is named
directly. Top residuals (Vol 11 +11.7M, Vol 15 +9.6M, Vol 17 +5.4M
over-perform) point at something **humped and time-specific** — a fandom
peak around 2017-2020 — but the shape is non-linear and outside our
current feature set. Plausible unmeasured drivers: production quality
(mashup transition smoothness — eventually measurable from the alignment
pipeline), viral moments (single mashup clipped on TikTok), tour
proximity (BB Land tour timing siphoning or boosting streams), YouTube
algorithm shifts. None are in aux.db today.

**Sample-size caveat**: n=23 volumes; all r values have wide CIs (e.g.
r = +0.51 has 95% CI roughly [+0.13, +0.76]). Signs are solid, magnitudes
approximate. Tightening would need Spotify/SoundCloud per-set play counts
as additional observations.

Reproduction: [scripts/bb_set_views_analysis.py](scripts/bb_set_views_analysis.py).
View counts persisted in `aux.set_views`; headline metrics in
`aux.analysis_results` under `analysis_name='bb_set_views_v1'`.

### Broadening the chart definition: peak position matters, breadth doesn't

The Hot 100 year-end finding above used a *narrow* popularity signal (a
song needs sustained chart presence over a calendar year to qualify).
Re-running with weekly Hot 100 history (1958-present, ~32k unique songs,
keyed on per-song all-time peak position and weeks-on-chart) shows:

**Two Friends DO draw from a wider pool than year-end captures:**

- 52.4% of BB acapellas (1,344 of 2,535) charted on weekly Hot 100 at any
  peak — vs. 37.9% on year-end. The +14pp gap is **384 acapellas** that
  appeared on Hot 100 weekly but never made year-end.
- **75% of those 384 have ≥100k Last.fm listeners** — confirming they're
  genuinely recognizable vocals, not obscure flukes. Most are brief #1-#10
  peaks (54 tracks) or top-40 hits (126 tracks) that didn't sustain a
  year-end run.

**But broadening the definition does NOT strengthen views prediction —
narrowing it does:**

| feature (acapella side) | r vs views | r² |
|---|---|---|
| **top-10 ever rate** (new, narrowest) | **+0.571** | **0.327** |
| year-end rate (original) | +0.533 | 0.284 |
| top-40 ever rate | +0.431 | 0.186 |
| weekly ever rate (broadest) | +0.458 | 0.210 |
| mean acapella peak position (continuous) | −0.145 | 0.021 |
| mean acapella weeks-on-chart (continuous) | −0.035 | 0.001 |

The predictive signal sharpens as the chart cut narrows toward "biggest
at-release-time hits." The 384-track "missed by year-end" expansion adds
recognizable songs but **dilutes** the engagement signal rather than
strengthening it.

**Refined interpretation**: the predictive variable isn't *popularity*
broadly or even *chart presence* — it's **peak-tier mass-culture intensity
at release time**. Two Friends *select* from a wider pool (52% weekly-charted),
but engagement is driven specifically by the top-tier subset within that
pool. Sustained chart presence (`mean_aca_woc` ≈ 0 correlation) and
average chart position (`mean_aca_peak` ≈ 0) don't add explanatory power
on top of "did this song peak in the top 10."

**Implication for modeling** (refining the prior section): the popularity
prior for the acapella role should weight **top-10 cultural-moment hits
most heavily**, not chart-presence breadth. Two Friends' *selection*
function admits a wider pool, but a *predicted-engagement* head should
key on the narrow top-tier subset.

**Sample-size caveat (still binding)**: n=20 volumes for the set-views
regression. All r values have wide CIs (top-10 r = +0.571 has 95% CI
roughly [+0.20, +0.80]). Differences in r below ~0.10 are within noise —
the directional pattern (narrower > broader) is the load-bearing finding,
not the specific R² values. Tightening would need additional per-set
play-count observations (e.g. Spotify per-track plays from charts.spotify.com
2017+, which would cover the modern half of the corpus).

Reproduction: [scripts/bb_weekly_chart_analysis.py](scripts/bb_weekly_chart_analysis.py).
Source data: [scripts/aux_db_sync.py](scripts/aux_db_sync.py) ingests
`data/analysis/billboard_weekly_current.csv` (utdata/rwd-billboard-data
public mirror, 700k weekly chart-rows → 32,561 unique songs with peak/woc
aggregates) into `aux.chart_song_history`. Headline metrics in
`aux.analysis_results` under `analysis_name='bb_weekly_chart_v1'`.

### Spotify Top 200 confirms the top-10 pattern with a different proxy

Adding **Spotify Daily Top 200 chart history** (via the kworb.net mirror
of charts.spotify.com) for tracks released after 2015 — a *streaming-era*
popularity signal independent of Billboard's radio + sales weighting —
tests whether the "top-10 peak intensity drives engagement" finding holds
when the proxy changes.

**Coverage**: 996 of 3,021 BB tracks (33%) appeared on at least one country's
Spotify Top 200. In the modern-window subset (release year ≥ 2015 where
the Spotify Charts archive applies), 539 of 1,238 acapellas (43.5%) charted.
Spotify Charts data starts ~Dec 2017, so pre-2017 acapellas — most of BB's
catalog vocals — get no signal here.

**Spotify peaks and Billboard peaks measure related but distinct things.**
Cross-validating the 465 BB tracks that appear on both charts:

- Pearson(spotify_peak_global, hot100_weekly_peak) = **+0.093** (linear is weak)
- Spearman = **+0.405** (rank-ordering moderately preserved)
- Top-10 overlap: 162 in both, 38 Spotify-only, 96 Billboard-only

The huge Pearson-Spearman gap means outliers dominate the linear fit while
the rank-ordering is genuinely correlated. Billboard top-10 catches a
strict superset of "US radio + chart" hits; Spotify Global top-10 picks
up international streaming hits and viral moments that Hot 100 misses,
and vice versa.

**Modern-window hit rates (release_year ≥ 2015):**

| chart cut | acap rate | instr rate | acap/instr ratio |
|---|---|---|---|
| billboard year-end | 27.8% | 7.2% | 3.87× |
| billboard weekly | 41.0% | 11.8% | 3.46× |
| billboard top-40 | 32.1% | 7.6% | 4.23× |
| billboard top-10 | 20.4% | 5.6% | 3.63× |
| **spotify global ever** | 34.2% | 13.8% | 2.48× |
| **spotify US ever** | 34.2% | 12.2% | 2.79× |
| spotify global top-40 | 24.3% | 7.6% | 3.21× |
| spotify global top-10 | 13.6% | 4.3% | 3.18× |

The acap/instr ratio is **lower for Spotify than Billboard** (2.5–3.2× vs
3.5–4.2×). Spotify's user base streams dance/EDM heavily, so the
instrumental side (which is mostly EDM-genre) gets disproportionately more
representation on Spotify charts than on Hot 100. **Spotify alone is a
less discriminating signal for the acap/instr selection asymmetry.** This
matters: when designing a "predicted listener engagement" head, a
Spotify-only popularity prior would underweight the asymmetry that
Billboard captures more cleanly.

**Set-views regression (n=20 volumes — wide CIs, directional only):**

| feature (acapella-side) | r vs views | r² |
|---|---|---|
| **sp_global_top10_rate** | **+0.586** | **0.343** |
| billboard weekly top-10 rate | +0.571 | 0.327 |
| billboard year-end rate | +0.533 | 0.284 |
| sp_global_top40_rate | +0.434 | 0.188 |
| sp_us_top40_rate | +0.435 | 0.189 |
| sp_charted (any Spotify Top 200) | +0.285 | 0.081 |

Multivariate fits:

| model | R² |
|---|---|
| billboard top-10 alone | 0.327 |
| spotify global top-10 alone | 0.343 |
| **billboard top-10 + spotify global top-10** | **0.432** |
| spotify top-10 + billboard year-end | 0.432 |
| all three top-tier signals | 0.444 |

**Combining Billboard and Spotify top-10 signals improves R² by ~+0.10 over
either alone** — meaningful given they catch partly disjoint culturally-peak
moments. The R² ceiling at ~0.44 with three correlated signals confirms
the prior conclusion: more popularity variance won't be unlocked by adding
more chart proxies; the remaining ~55% of variance is non-popularity (mix
quality, viral moments, channel-growth artifacts).

**Spotify-only acapellas (charted on Spotify but missed by both Billboard
signals): 112 tracks.** Median Spotify global peak = 59.5, only 40% with
≥100k Last.fm listeners (vs. 75% for the Billboard-weekly-but-not-yearend
bucket of 384 tracks). The Spotify-only group is weaker on the
recognizability axis — these are mid-tier streaming entries, not headline
cultural hits. Two Friends do reach into this pool, but the per-track
engagement contribution is smaller than the Billboard-charted subset.

**Refined picture across three popularity proxies:**

1. **Hot 100 year-end** (n=956 BB matches): narrowest, US-radio + sales,
   highest acap/instr ratio (2.4× on full corpus)
2. **Hot 100 weekly all-time** (n=1,344): broader, includes brief peaks
   without sustained chart runs
3. **Spotify Global Top 200** (n=996, modern half only): streaming-era,
   captures international/viral hits, less discriminating for acap/instr

All three converge on the same headline: **the strongest single predictor
of BB set engagement is the rate of top-10 culturally-peak acapellas in
the volume.** Spotify and Billboard top-10 are best used as *complementary*
signals (combined R² gain ≈ +0.10) rather than substitutes for each other.

**Implication for modeling (refined again)**: a predicted-engagement head
should combine top-10 culture-moment flags from multiple chart systems
(weighted toward at-release-time peak signals like Hot 100 weekly + Spotify
Daily, downweighted on cumulative streaming proxies like raw Last.fm
listeners which empirically don't transfer). For pre-2017 catalog
acapellas where Spotify gives no signal, the Billboard signals carry the
prediction. For modern acapellas (post-2017), use both.

**Still unmeasured**: SoundCloud popularity has no historical chart-archive
equivalent (only cumulative play counts, look-back-biased) and is the only
remaining major streaming platform without a usable at-release-time signal
for this corpus.

Reproduction: [scripts/bb_spotify_charts.py](scripts/bb_spotify_charts.py)
(scrapes kworb.net per-track) →
[scripts/aux_db_sync.py](scripts/aux_db_sync.py) (ingests into
`aux.track_spotify_charts`) →
[scripts/bb_spotify_chart_analysis.py](scripts/bb_spotify_chart_analysis.py)
(comparison + regressions). Headline metrics in `aux.analysis_results`
under `analysis_name='bb_spotify_chart_v1'`.

### Union coverage of popularity proxies across BB tracks

A natural follow-up to "which proxy is most predictive" is "how many BB
tracks does the union of proxies *cover at all*". Computed across the
combined Hot 100 year-end / Hot 100 weekly / Spotify Top 200 / Last.fm
(≥100k listeners) signals:

| signal | acapellas (n=2,536) | instrumentals (n=1,058) |
|---|---|---|
| Hot 100 year-end | 896 (35.3%) | 143 (13.5%) |
| Hot 100 weekly ever | 1,240 (48.9%) | 216 (20.4%) |
| Spotify Top 200 (any country) | 615 (24.3%) | 115 (10.9%) |
| Last.fm ≥100k listeners | 572 (22.6%) | 101 (9.5%) |
| **union: any chart** | **1,415 (55.8%)** | **265 (25.0%)** |
| **union: chart OR ≥100k Last.fm** | **1,557 (61.4%)** | **287 (27.1%)** |
| any chart top-10 ever (peak tier) | 760 (30.0%) | 130 (12.3%) |
| **truly obscure (no signal)** | **979 (38.6%)** | **771 (72.9%)** |

The acap/instr asymmetry persists across the union: ~61% of acapellas
are caught by at least one signal, but only ~27% of instrumentals are.
Almost three-quarters of BB instrumentals are obscure by every metric
we have — they're picked for compatibility (key/BPM/genre/structure),
not popularity, full stop.

The 38.6% "truly obscure" acapella tier (979 tracks) has two plausible
contributors that our current proxies can't separate:

1. **Real coverage gaps in our signals**: international hits never on
   US Hot 100 (UK / Latin / K-pop charts not yet in aux.db), pre-2017
   tracks Spotify never indexed, regional radio darlings. Adding UK
   Official Chart and Latin charts would likely shrink this tier.
2. **Genuinely non-popularity-driven picks**: chosen for vocal quality,
   melodic compatibility, or Two Friends' personal taste rather than
   recognizability.

Computed via the ad-hoc query in `/tmp/coverage.py` (not committed —
re-run by joining `aux.track_chart_match` + `aux.track_spotify_charts` +
`aux.track_lastfm` on track_id, role-tagged via the BB scrape).

### Hypothesis: user-history is the right feature for the *per-user* model, not the aggregate

A natural read of the ~55% unexplained variance in aggregate set-views
is "audience taste — what listeners actually want to hear." A sharper
read separates two distinct prediction problems:

**Aggregate views variance (~55% unexplained) is NOT primarily individual
taste.** Aggregate views sum across millions of distinct listeners, so
any individual-listener taste term cancels out — it gets captured in
population-average chart popularity (the chart_rate features already
predict 40-44% of variance precisely because they aggregate that). What's
left over is *volume-level idiosyncrasies* that no per-user feature can
explain: production / mashup-transition quality, viral TikTok clips of
specific mashups, YouTube algorithm shifts, BB Land tour timing.
Per-user listening data — no matter how rich — cannot push aggregate-views
R² up because it has the wrong cardinality.

**Per-user engagement prediction is a different problem entirely**, and
*there* user listening history is the dominant feature. The aligned BB
corpus this pipeline produces becomes the *training data* for a sequence
model that learns mashup compatibility. The personalized inference step
conditions on a user's SoundCloud (and/or Spotify) likes + playlists +
demographics as a few-shot prompt: "given this user's taste distribution
overlaps with the corpus like this, score these candidate mashups."
That model's variance budget *is* dominated by individual-listener
history — the population-level chart-rate findings would map to its
*priors over candidate tracks*, not its per-user discrimination.

**Practical translation**:

- Aggregate corpus modeling (what we're doing now): chart-rate features
  carry ~40-44% of variance; remaining ~55% is unmeasured production /
  viral / algorithmic factors and won't be improved by user-history data.
- Per-user personalization (the project endpoint): user listening history
  is the load-bearing feature. Two natural input sources are SoundCloud
  (likes + playlists + reposts) and Spotify (Top Tracks + saved tracks +
  playlists). SoundCloud is the more interesting target because its
  user-curated playlists tend to encode dance-music taste more densely
  than Spotify's; Two Friends' BB audience overlaps heavily with
  SoundCloud's EDM/mashup community.

**Feasibility of SoundCloud user-history access**:

- **OAuth (preferred for any user-facing path)**: SoundCloud's API
  supports OAuth 2.0 with `me/likes`, `me/playlists`, `me/tracks`
  endpoints. User consents, you act on their behalf, no ToS or scraping
  ambiguity. The practical bottleneck is API key approval — SoundCloud
  stopped accepting new app registrations broadly around 2023; an
  existing key or partnership is needed. Validate this is unblocked
  before depending on it.
- **Public-profile scraping (gray zone, only for corpus-side research)**:
  user profiles set to public expose likes / reposts / playlists at
  public URLs. *HiQ v. LinkedIn* (9th Cir. 2022) established that
  scraping public data isn't a CFAA violation, so federal-law exposure
  is low — but SoundCloud's ToS prohibits automated collection
  regardless of intent (civil-law exposure: account bans, possible
  C&D for high-volume use). Non-commercial use does not exempt; only
  authorization does. Honest cost-benefit: low risk for a one-time
  research scrape of N profiles to bootstrap a taste-cluster prior;
  meaningful risk for productionized continuous use. Robots.txt should
  be respected as courtesy but isn't a legal shield in either direction.
- **Spotify alternative**: Spotify's API is more permissive about new
  app registrations and has richer playlist/track-features endpoints.
  Worse genre fit (less dance-music taste density) but lower
  feasibility risk. Use as the primary signal for cold-start; fold in
  SoundCloud where users have it linked.

**Implication for the project plan**: the corpus-side popularity
investigation has plateaued near R² ≈ 0.44 for aggregate views, and
further chart proxies (UK, Latin, K-pop) would improve *coverage of the
"truly obscure" acapella tier* more than they'd improve aggregate-views
R². The next-meaningful-gain is per-user data — and the API access path
(OAuth on at least one of SoundCloud / Spotify) is the load-bearing
dependency to validate before designing the personalized inference head.

### Auxiliary research database

Research signals (release years, Last.fm, Billboard, chart matches) are
persisted in **`data/analysis/aux.db`** (SQLite, gitignored, ~1.8 MB,
rebuildable from cache files + main DB):

| table | rows | what it holds |
|---|---|---|
| `track_meta` | 3,342 | `track_id` (1001tl), artist, title, spotify_id, release_year (+ source) |
| `track_lastfm` | 3,342 | per-track Last.fm info: lfm_artist, lfm_title, mbid, listeners, playcount, error_code |
| `chart_yearend` | 6,621 | Billboard Hot 100 year-end 1958-2024 (chart_name + year + rank + title + artist) |
| `chart_song_history` | 32,561 | per-song all-time aggregates from weekly Hot 100 1958-2026 (peak_position, weeks_on_chart, debut_date, last_chart_date) |
| `track_spotify_charts` | 2,862 | per-track Spotify Top 200 history via kworb.net mirror — Global+US+per-country peak positions, lifetime streams, weeks on chart; charts.spotify.com data starts ~Dec 2017 so pre-2017 catalog tracks return `status='uncharted'` |
| `track_chart_match` | 2,300 | resolved BB-track → chart-entry pairings (year-end + weekly all-time), with `rank` for year-end matches and `peak_position`/`weeks_on_chart` for weekly matches |
| `set_views` | 24 | per-set platform view counts (BB YouTube counts to date) |
| `analysis_results` | ~125 | flattened headline metrics from corpus-empirics analyses (queryable by `analysis_name`/`metric`/`group_key`) |

All keyed on `track_id` (same identifier as `dj_set_rows.data-trackid` in the
main DB) so cross-DB joins are straightforward — `ATTACH DATABASE` aux.db
and join, or query through the script.

This is a holding pen, not production schema. Signals graduate to the main
DB only after they prove useful for downstream modeling. Rebuild any time
via [scripts/aux_db_sync.py](scripts/aux_db_sync.py), which is idempotent
and reads the existing CSV/JSON caches.

## Configuration

All crawler behavior is controlled via `config.yaml`:
- **paths** — Data dirs, database location, logs, captcha images
- **generator** — Job selection (testing mode, filtering, limits)
- **timing** — Crawl delays (10s default) with jitter
- **browser** — Headless Chrome settings, viewport, timeouts
- **profiles** — Browser profile rotation (retirement after 750 sites)
- **failure** — Error handling modes (fail-fast, ajax_failure behavior, consecutive failure limits)
- **captcha** — Solver mode (ocr/continue/wait/kill), wait timeout, max OCR attempts

## Database

SQLite with ~25 tables split into two groups, all with cascade-deleting FKs.

**Scraper tables** (populated by `web_crawler/`):
`dj_sets` (canonical metadata), `dj_set_crawls` (HTML snapshots with ETag dedup),
`dj_set_media_links`, `dj_set_rows`, `dj_set_track_media_links`, `scrape_failures`.

**Audio-pipeline tables** (populated downstream of the scraper):
`set_audio` / `set_stems` / `set_measures` (mix-side audio + demucs stems + beat
grid), `track_audio` / `track_stems` / `track_measures` (ref-track equivalents),
`track_analysis` / `track_identity` / `track_audio_features` / `track_mert_sections` /
`track_sections` (per-ref analysis outputs), `canonical_track_cue_points`
(cue-detr cues keyed by track_id, full-song @ sensitivity=0.5), `track_fingerprints` /
`set_fingerprint_hits` (chromaprint ingestion + mix scan), `set_section_alignment`
(sota.py output, `confidence_source='sota_v2'`), `measure_alignment`,
`set_playback_score`, `set_timeline`, `set_analysis`.

Schema lives in [web_crawler/database/schema.sql](web_crawler/database/schema.sql).

## Storage & cluster

The project runs across four machines (see [Makefile](Makefile) for cluster ops):

- **pi-storage** (Linux aarch64) — canonical state (DB + audio + stems) + scraper services + **CPU-side analysis** (yt-dlp downloads, beat_this, cue-detr, librosa, pyloudnorm). Long-running services live here. Reachable via Tailscale MagicDNS.
- **pi-worker** (Linux aarch64) — AJAX retry drain (`tracklist-ajax-retry.service`). Spare CPU available for batch CPU analysis when idle.
- **Vast.ai spot GPU** (rented, ephemeral) — **GPU-bound analysis** (Demucs, MERT) and **Essentia** (no aarch64 wheels — must run on x86_64). Job pulls audio from pi-storage over Tailscale, runs inference, writes results back, terminates. Cost target ≤$5–10 for the whole 16k-track corpus on a 3090/4090 spot.
- **Mac** (Apple Silicon) — dev driver *and* a second analysis worker. The full Demucs/beat_this/cue-detr/MERT/Essentia pipeline runs locally on the MPS backend via [scripts/mac_analyze_loop.py](scripts/mac_analyze_loop.py) (sibling of `vast_loop.py`); pulls audio from pi-storage over Tailscale, writes results back. Expect ~200–250 s/track vs ~85 s on a 4090 — useful when no Vast box is rented or for the long tail. Also drives the **alignment** workflow (see below).

**Analysis split (which dep runs where):**

| Component | Runs on | Why |
|---|---|---|
| yt-dlp / spotdl downloads | pi-storage | CPU-only; cross-arch wheels work |
| beat_this (beats/downbeats) | pi-storage CPU **or** Mac MPS | PyTorch has aarch64 + MPS wheels; small model |
| cue-detr (EDM cues) | pi-storage CPU **or** Mac MPS | DETR transformer; small model |
| librosa, pyloudnorm | pi-storage **or** Mac | pure Python |
| **Essentia** (key/BPM/valence/mood/etc.) | **Vast.ai** *or* **Mac** | no aarch64 wheels — Essentia ships only x86_64 manylinux + macOS arm64, so the Mac has a `venvs/essentia/` Py3.13 sandbox and runs Essentia as a subprocess |
| **Demucs** stems | **Vast.ai** *or* **Mac MPS** | GPU-bound; ~30s/track on Pi CPU vs ~1s/track on 4090 vs ~3–5s/track on M-series MPS |
| **MERT** embeddings | **Vast.ai** *or* **Mac MPS** | same; [audio_pipeline/analysis/adapters/mert_adapter.py](audio_pipeline/analysis/adapters/mert_adapter.py) auto-selects `cuda` → `mps` → `cpu` |

The Mac mirrors the pi-storage CPU stack (`venvs/audio/`) plus the `venvs/essentia/` Py3.13 sandbox, so the **entire production analysis pipeline** is exercisable locally — not just for development, but as an actual production worker for batches that don't justify spinning up Vast.

**Alignment workflow (Mac-driven):**

Ground-truth alignment work happens in `~/aligning/` on the Mac:

- [scripts/pull_set_for_alignment.py](scripts/pull_set_for_alignment.py) — queries pi-storage's canonical DB over SSH, rsyncs the mix recording + per-track stems into `~/aligning/<set_id>__<sanitized-title>/{mix.<ext>, tracks/, manifest.json, stems/}`, ready to drag into Ableton.
- [scripts/tag_aligning_folder.py](scripts/tag_aligning_folder.py) — reads `manifest.json`, queries pi-storage `track_audio_features`, injects BPM + Camelot key + feature comment into each M4A's iTunes tags so Ableton shows them in the browser.
- `~/aligning/phase-cancel/` — phase-cancellation instrumental extraction (see the `project_phase_cancel` memory; winner config is `adaptive --smooth 0.5 --fft 4096 --cap 4`).

**Consistency model.** The `~/aligning/<set>/` folder is a **read-replica of pi-storage**: the pull script is the only writer, and pi-storage's DB is the source of truth for what should be there. Two operations keep them consistent:

1. **Re-run the pull = delta refresh.** Rsync runs in archive mode (`-aL --partial --inplace`), so re-invoking `pull_set_for_alignment.py <set_id>` only transfers files that changed on pi-storage (regenerated stems, replaced audio). Unchanged files are skipped.
2. **`--prune` removes orphans.** When pi-storage's view diverges by *removal* — a track gets re-resolved to a different `track_audio_id`, a stem subdir-name changes, an audio file gets replaced with a different codec — old local files are stale. `--prune` walks `tracks/` and the plan's stem subdirs and deletes audio-extension files not in the freshly-rebuilt manifest. Combine with `--dry-run` to preview deletions without touching anything. Gated behind the flag so a fat-finger can't wipe in-flight work.

**Annotator rename convention (one-sided, Mac-only).** The human annotator renames track files and stem subdirs to expose tempo + key inline, e.g. `tracks/030__Going Deeper - Little Big Adventure [126bpm 8B].m4a` and `stems/001__Carmen Twillie - Circle Of Life [84bpm 6B]/`. This makes Ableton's clip browser show tempo/key at a glance during alignment, dramatically speeding the workflow. Two known tags:

- `[NNNbpm KK]` — tempo + Camelot key, e.g. `[126bpm 8B]`, `[84bpm 6B]`
- `[no-features]` — flags tracks without Essentia rows on pi-storage so the annotator knows to skip them

These renames are **never written back to pi-storage** — the canonical names there stay `{Artist} - {Title}.{ext}`. `--prune` recognizes these tag patterns (`_USER_TAG_PATTERN` in [pull_set_for_alignment.py](scripts/pull_set_for_alignment.py)) and treats tagged files/subdirs as user territory: never deleted. Likewise, anything inside a user-renamed stem subdir (e.g. `phase_cancel_v*.wav` artifacts) is left alone because the parent subdir isn't in the prune's plan-owned set.

Consequence: re-pulling a set will deposit *fresh un-tagged copies* of files the annotator previously renamed. That's expected — the annotator either re-runs the rename pass or ignores the duplicates. There's no automatic re-tag-on-refresh today.

**Folder lifecycle.** The folder is ephemeral — delete a set once alignment data has been written back to the canonical DB. (Write-back of Ableton-session alignment results to pi-storage is not implemented yet; that's the next missing piece.)

**Canonical paths on pi-storage:**

| Kind | Path |
|---|---|
| DB | `/mnt/storage/data/db/music_database.db` |
| Track audio | `/mnt/storage/objects/{track_id}/{track_id}__{platform}__{player_id}.{ext}` |
| Demucs stems | `/mnt/storage/stems/{track_audio_id}/{vocals,drums,bass,other,instrumental}.{ext}` |
| Human-readable library | `/mnt/storage/library/{Artist}/{Title}/...` (symlinks built by [library/builder.py](library/builder.py)) |
| Essentia TF model cache | `/mnt/storage/data/essentia_models/*.pb` (synced from Vast.ai or fetched on first use) |

The repo's `data/db/music_database.db` is **a stale local copy for development — never the source of truth.** Services on pi-storage write to the canonical DB continuously; the local copy diverges quickly. To inspect canonical state, query pi-storage directly: `ssh pi-storage 'sqlite3 /mnt/storage/data/db/music_database.db "..."'` or via the FastAPI jobqueue.

**Pi-storage venvs:**
- `venvs/web_crawler/` — scraper, materializer, FastAPI jobqueue (BeautifulSoup, lxml, ddddocr, FastAPI).
- `venvs/audio/` — yt-dlp + spotdl for downloads, plus the CPU analysis stack (PyTorch CPU, beat_this, cue-detr, librosa, pyloudnorm). **Does not include Essentia / Demucs / MERT** — those run on Vast.ai or the Mac.

**Mac venvs (same names, different role):**
- `venvs/audio/` — same stack as pi-storage but with MPS-backed PyTorch, so beat_this / cue-detr / Demucs / MERT all run on Apple Silicon GPU.
- `venvs/essentia/` — Py3.13 sandbox holding the `essentia-tensorflow` wheel (macOS arm64 wheel exists; pi-storage's aarch64 Linux does not). Invoked as a subprocess from `venvs/audio/` whenever the analysis pipeline needs Essentia.

Use [Makefile](Makefile) for cluster ops (`make deploy`, `make status`, `make ssh-storage`).

## Git workflow

Use your best judgement on when to commit and push. The default Claude Code rule "only commit when explicitly asked" is **overridden for this project** — proactively commit logical units of work and push them so pi-storage / pi-worker can pick them up via `make deploy`. Group changes into reviewable commits (one feature per commit, not one-giant-blob). Don't push directly to `main` if a pending change is still unstable; otherwise keep it moving.

## Environment

- Python project — no pyproject.toml, uses `requirements.txt` files
- `.env` file (loaded via python-dotenv) holds optional secrets for the email-based captcha fallback (CAPTCHA_EMAIL_SENDER / CAPTCHA_EMAIL_PASSWORD / etc.). The default OCR path needs no secrets.
- Virtual environments in `venvs/` (gitignored)
- `data/`, `profiles/`, `logs/` are gitignored — only `data/djs/*.json` job files are tracked

# Python Style Guide: Rust-Flavoured Functional Python

This document defines the programming style for all Python code in this project.
The guiding philosophies are:

- **Rust**: explicit over implicit, errors as values, ownership awareness
- **Lambda calculus**: pure functions, immutability, composition over mutation
- **Linear type theory**: resources are consumed, not shared; use-once semantics enforced structurally
