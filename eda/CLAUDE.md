# eda/ — exploratory analysis (cross-cutting consumer)

`eda/` is not a chain stage — it reads from multiple stages and produces
findings, not pipeline state. Organized into per-concern subfolders:

- **`corpus_empirics/`** — empirical studies of the corpus (`bb_*.py` scripts +
  `findings.md`). House pattern below.
- **`alignment/`** — placeholder for alignment-side EDA (the aligner isn't built
  yet; see the root terminology block).
- **`queries/`** — ad-hoc query scratch.
- **`common.py`** — shared DB access + DataFrame loading + pydantic_ai agent
  integration, used by the notebooks (`eda.ipynb`, `set_structure.ipynb`,
  `tokenizer.ipynb`).

## Running

Jupyter notebooks use [common.py](common.py) for shared DB access. Scripts under
`corpus_empirics/` assume `data/analysis/` and `data/db/` paths **relative to
repo root** — run them from the project root, not from the subfolder.

## Corpus empirics

Full write-ups (numbers, tables, modeling implications) plus the scripts that
produced them live in [corpus_empirics/](corpus_empirics/). The findings
document is [corpus_empirics/findings.md](corpus_empirics/findings.md); each
section links to its reproducing script. Headline metrics are also queryable
from `data/analysis/aux.db` via the `analysis_results` table.

The `corpus-empirics` skill scaffolds a new study (script in
`corpus_empirics/bb_*.py`, results persisted to `aux.db`, a findings section
appended to `findings.md`).

Findings, in dependency order:

1. **Acapella/instrumental era choice is orthogonal** — within a mashup slot, release-year of the two roles is independent (r ≈ 0). The pair-scoring head must not condition on year-proximity.
2. **Acapella choice IS driven by popularity** — acapellas are 3× more likely to be Hot 100 year-end hits and have ~200× more Last.fm listeners than the instrumentals. Treat the two roles with separate popularity priors.
3. **Set views are driven by chart-hit-vocal density** — ~39% of per-volume YouTube-views variance explained by acapella chart-rate + count. Instrumental popularity is neutral-to-negative.
4. **Peak position matters, breadth doesn't** — top-10 hit rate (r = +0.57) beats weekly chart presence; the signal sharpens as the chart cut narrows toward "biggest at-release-time hits."
5. **Spotify Top 200 confirms the top-10 pattern** — combining Billboard + Spotify top-10 signals lifts R² to 0.44 (apparent ceiling for popularity features alone).
6. **Union coverage of popularity proxies** — ~61% of acapellas vs ~27% of instrumentals are caught by ≥1 popularity signal. 73% of BB instrumentals are obscure on every metric we have — picked for compatibility, not popularity.
7. **User-history is for the per-user model, not aggregate** — the remaining ~55% of aggregate-views variance is unmeasured production / viral / algorithmic factors, not individual taste. User-history data belongs in the personalized-inference head, not here.

The `aux.db` holding schema (release years, Last.fm, Billboard, Spotify charts,
BB-track ↔ chart-entry pairings, set views, headline results) is documented at
the bottom of [findings.md](corpus_empirics/findings.md#auxiliary-research-database).
Rebuild via [corpus_empirics/aux_db_sync.py](corpus_empirics/aux_db_sync.py).
