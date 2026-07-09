# Low-rank structure study — DJ-set selection space

Quantitative test of the repo's low-rank worldview ("a few generative truths
dictate all DJ sets") on the canonical corpus, 2026-07-09. Selection space
only: the binary set x recording incidence matrix from `set_track_slots`
(Representation B — audio features — was gated on coverage and **skipped**:
only 1.7% of slots have a `track_audio_features` row with bpm).

**Data:** 39,498 sets with >=10 distinct recordings x 217,825 recordings,
1,210,644 incidences (density 0.014%). Rows L2-normalized. DJ label per set
from the tracked `data/djs/*.json` scrape job files (108 DJs with >=8
qualifying sets; `bb*` files mapped to twofriends; sets on >=2 DJs' pages =
`__multi__`, excluded from per-DJ analyses).

## Verdicts

| # | Claim | Verdict | Key number |
|---|-------|---------|------------|
| 1 | Sets are low-rank **overall** | **REFUTED** (absolute); modest excess structure over null | top 500 of 39,498 comps capture **19.4%** energy (null: 14.1%) |
| 2 | Even lower rank **within a DJ** | **CONFIRMED** (relative to size-matched control) | **108/108 DJs** below control; median ratio **0.72** |
| 3 | Per-DJ **bases differ meaningfully** (similarity proxy) | **CONFIRMED** | LOSO own-DJ retrieval **76.5%** vs 0.93% chance (82x) |
| 4 | Metadata/ML: representation is DJ- and set-type-identifiable | **CONFIRMED** | DJ clf **45.3%** (49x chance); live-vs-show AUC **0.95** |

Net reading for the worldview: **"few global generative truths" fails in
selection space; "DJ-conditional low rank with DJ-identifying bases" holds.**
The compression is per-DJ, not corpus-wide.

## A1 — overall rank: REFUTED

Randomized SVD (k=500, n_iter=10) of the L2-row-normalized matrix vs a
bipartite configuration-model null (edge-list endpoint shuffle preserving
both degree sequences; dedup loss 0.15% of edges; 2 realizations).

| spectrum | energy @ k=500 | comps for 50/80/90% |
|---|---|---|
| real | 19.4% | all >500 |
| null (x2) | 14.1% / 14.2% | all >500 |

- Not even **50%** of energy is reachable within 500 components (1.3% of the
  row dimension). The corpus incidence matrix is high-rank; the spectrum is
  nearly flat, as expected at 0.014% density with ~30 tracks/set.
- There **is** real structure: 1.37x the null's energy concentration at
  k=500, top singular values 12.8, 10.5, 9.5... vs a flat null. But this is
  "noticeable correlational structure", not "a few generative dimensions".
- **Popularity check:** PC1 carries only **0.41%** of energy and its loading
  vector correlates r=0.83 with track corpus frequency — the largest single
  axis is essentially track popularity, and it is tiny. "Low rank" can NOT
  be manufactured here by a popularity effect.

Caveats: randomized SVD slightly underestimates tail singular values —
applied identically to real and null, so the comparison stands. The null
preserves degrees but not genre/era block structure; a blockwise null would
shrink the 1.37x gap further, not grow it.

## A2 — within-DJ rank: CONFIRMED (relative), refuted as "a few truths"

Per DJ (108 with >=8 qualifying sets): exact full Gram spectrum of the DJ's
own submatrix; r80 = components for 80% energy. Control: r80 of 10 random
same-size subsets of the whole qualifying corpus (controls for n exactly).

- **All 108 of 108 DJs have r80 below their size-matched control** (100%).
- Ratio real/control: median **0.724**, IQR 0.587–0.794, min 0.273
  (swedishhousemafia), max 0.976 (cid).
- Most compressed: swedishhousemafia 0.27, kygo 0.31, alanwalker 0.31,
  djsnake 0.31, zedd 0.33. Least: cid 0.98, blasterjaxx 0.92, drfresch 0.89
  (open-format/house DJs churn vocabulary; brand-heavy EDM DJs repeat).
- Absolute rank is still substantial: median r80/min(n_sets, vocab) =
  **0.538**. A DJ's set collection is reliably ~28% more compressible than a
  random collection of equal size, but it is not spanned by a handful of
  components.

## A3 — DJ bases as a similarity metric: CONFIRMED

Per-DJ top-k right-singular subspace over the shared 217,825-recording
vocabulary, k = min(r80, 10) (cap stated; r80 always >10, so k=10 except
where n_sets-1 < 10). DJ x DJ similarity = mean cosine of principal angles.

**Validation — leave-one-set-out retrieval** (up to 20 held-out sets/DJ,
2,144 total; own-DJ basis recomputed without the held-out set): top-1
own-DJ accuracy **76.5%** vs chance **0.93%** (1/108) — **82x chance**.
Weakest DJs: cid 20%, davidguetta 25%, dondiablo 30% (the high-r80,
vocabulary-churning DJs). Named DJs: johnsummit 95%, itsmurph 100%,
discolines 90%, twofriends 45%.

Off-diagonal similarity is small in absolute terms (mean 0.018 +/- 0.021 —
217k-dim supports barely overlap) but **ordinally structured**: affinity
table (`data/affinity.tsv`) shows a big-room/EDM block (davidguetta–afrojack
0.13, hardwell–afrojack 0.09, hardwell–dimitrivegasandlikemike 0.09) and a
house block — johnsummit/discolines/itsmurph mutually ~0.03 while ~0.00–0.01
to the trance/big-room DJs; arminvanbuuren is near-orthogonal to everyone
(max 0.04). Use it as a ranking metric, not a calibrated distance.

Small-n flag: itsmurph has only 17 qualifying sets; its row is directional,
not precise.

## M — metadata / ML extension: CONFIRMED

Set embeddings = top 200 SVD components of the global matrix.

- **DJ identifiability:** 5-fold logistic regression **45.3%** accuracy over
  108 classes (chance 0.93%, majority class 6.6%, n=37,994); nearest
  centroid 41.0%. Selection alone identifies the DJ ~half the time.
- **Set type** (title heuristic: " @ " = live gig, episodic patterns =
  radio/podcast show; 15,953 live / 21,729 show / 3,810 other): live-vs-show
  AUC **0.950**. The signal lives in the top of the spectrum: per-PC AUC =
  0.86 / 0.82 / 0.75 for PC1–3, ~0.5 beyond PC4. Top PCs by |coef| for the
  DJ classifier: 5, 1, 3, 2, 6, 30...; for set_type: 1, 2, 3, 5 — the
  leading components encode set format + popularity, DJ identity rides on
  mid-spectrum components.
- **PC ~ metadata (Spearman, PC1–5):** PC1 vs n_distinct 0.52, vs log views
  0.34; PC5 vs year 0.42. Popularity/length/era all load on the top PCs.
- **B2B:** explicit "b2b" appears in only 3 qualifying titles — no claim.
  Proxy = multi-DJ sets (on >=2 DJs' scrape pages, n=1,504): AUC 0.723,
  weakly detectable from selection.
- **Aligner-difficulty profile** (mean per set):

  | set_type | distinct tracks | acappella share | instrumental share |
  |---|---|---|---|
  | live | 43.1 | **6.9%** | 0.60% |
  | show | 21.8 | 0.8% | 0.49% |
  | other | 29.3 | 1.8% | 0.65% |

  Multi-DJ sets: 52.2 distinct tracks, 4.7% acappella (vs 29.8 / 3.2%
  single-DJ). Live and multi-DJ sets are ~2x longer and carry ~9x the
  acappella share — i.e. the alignment-hard axis (acappella is the worst
  axis per the BB12 baseline) concentrates exactly in live/festival sets.

## Honesty notes

- Selection space only. Audio-feature rank (Representation B) untested:
  1.7% slot coverage. A "sound-palette is low-rank" claim remains open.
- Single-source corpus (1001tracklists scrape, 108 scraped DJs, EDM-heavy).
  DJ label = whose page the set was scraped from; guest mixes on another
  DJ's radio show can mislabel.
- set_type is a title heuristic, unvalidated against ground truth.
- LOSO accuracy is not fully balanced: 20 held-out sets per DJ regardless
  of DJ size, so per-DJ accuracies are +/-11pp at n=20.
- Small-n: itsmurph (17 sets), partypupils (19), pawsa (31) flagged.
- The A1 "1.37x over null" gap is an upper bound on global structure; a
  genre-blocked null would attribute part of it to genre, not "truths".

## Reproduce

```bash
venvs/audio/bin/python eda/alignment/low_rank/pull_data.py     # ssh pi-storage (read-only)
venvs/audio/bin/python eda/alignment/low_rank/low_rank_study.py
# outputs: data/results.json, data/affinity.tsv, data/per_set.tsv
```
