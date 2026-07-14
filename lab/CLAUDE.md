# lab/ — DJ-music research lab (north-north star, deferred)

Split out of `eda/` on 2026-07-12. `lab/` is the **north-north star**: a digital
lab that (eventually) fuses **SoundCloud + 1001Tracklists** to gain new knowledge
about music — *why we like it* — in service of empowering better DJ sets. It
**activates once alignment is solved at scale** (SOTA + rigorous across ~20,000
sets); until then it is *preserved but deprioritized* against the alignment gate.

**Boundary rules (keep the alignment engine lean):**
- `lab/` depends only on `labeling` ground truth. It must **not** be imported by
  the alignment engine (`workspaces/alignment_prototype/`, `analysis/`,
  `labeling/`). Alignment-support analysis stays in `eda/alignment/`, not here.
- **Not part of the alignment DAG.** Extractable to its own repo later (the only
  cross-boundary dependency is `labeling.ground_truth`).
- Full context: [docs/alignment_bearings_20260712.md](../docs/alignment_bearings_20260712.md).

## Subfolders

- **`corpus_empirics/`** — empirical studies of the corpus (`bb_*.py` scripts +
  `findings.md`). House pattern below.
- **`information_dynamics/`** — chroma/HuBERT information-dynamic "surprise" as a
  mashup-compatibility signal ("why did the DJ pick *this* pairing"). Weak-GO
  result (see `information_dynamics/FINDINGS.md`); persisted to `aux.db`.
- **`audience_prior/`** — listener/taste priors ("why we like it").
- **`appleseed/`** — the Appleseed empowerment layer: `appleseed_librarian.py`
  (the fulfiller that feeds the `mashup_compiler` product repo via one SQLite
  file), its runbook + empowerment-layer doc, and its design specs. Stdlib-only,
  no chain coupling; test in `tests/test_appleseed_librarian.py`.
- **`specs/`** — product design docs for the rest of the empowerment layer
  (compiler / cast), whose implementation lives in separate repos.

## Running

Scripts assume `data/analysis/` and `data/db/` paths **relative to repo root** —
run from the project root as modules, e.g.
`venvs/audio/bin/python -m lab.corpus_empirics.bb_popularity`.

## Corpus empirics

Full write-ups (numbers, tables, modeling implications) plus the scripts that
produced them live in [corpus_empirics/](corpus_empirics/). The findings
document is [corpus_empirics/findings.md](corpus_empirics/findings.md); each
section links to its reproducing script. Headline metrics are also queryable from
`data/analysis/aux.db` via the `analysis_results` table.

The `corpus-empirics` skill scaffolds a new study (script in
`lab/corpus_empirics/bb_*.py`, results persisted to `aux.db`, a findings section
appended to `findings.md`).

Findings, in dependency order:

1. **Acapella/instrumental era choice is orthogonal** — within a mashup slot, release-year of the two roles is independent (r ≈ 0). The pair-scoring head must not condition on year-proximity.
2. **Acapella choice IS driven by popularity** — acapellas are 3× more likely to be Hot 100 year-end hits and have ~200× more Last.fm listeners than the instrumentals. Treat the two roles with separate popularity priors.
3. **Set views are driven by chart-hit-vocal density** — ~39% of per-volume YouTube-views variance explained by acapella chart-rate + count. Instrumental popularity is neutral-to-negative.
4. **Peak position matters, breadth doesn't** — top-10 hit rate (r = +0.57) beats weekly chart presence; the signal sharpens as the chart cut narrows toward "biggest at-release-time hits."
5. **Spotify Top 200 confirms the top-10 pattern** — combining Billboard + Spotify top-10 signals lifts R² to 0.44 (apparent ceiling for popularity features alone).
6. **Union coverage of popularity proxies** — ~61% of acapellas vs ~27% of instrumentals are caught by ≥1 popularity signal. 73% of BB instrumentals are obscure on every metric we have — picked for compatibility, not popularity.
7. **User-history is for the per-user model, not aggregate** — the remaining ~55% of aggregate-views variance is unmeasured production / viral / algorithmic factors, not individual taste. User-history data belongs in the personalized-inference head, not here.
8. **Mix pitch offsets = an acappella transposed to its instrumental's key (harmonic mixing), NOT varispeed** — a calibrated cents estimator over BB11/BB12 GT clips rejects tempo-pitch coupling (H1: R²=0.005) and shows the integer-semitone transpose is a key-match to the paired bed: bed-compatibility flips 1/56→50/56 after the dialed transpose, 53/56 move toward the bed (five vocals over *Faded* 2B all pitched 7B→2B). Sub-semitone fine = tuning-match on top (applied even when keys already agree). Same-song both-stems offset = master difference, not pairwise. Wrong-rips (Pat Benatar +99¢) are a minority. Buried acappellas must be measured on the vocal band or the bass bed fakes a −200¢ detune. One over-dialed label found: Coldplay – The Scientist (+122¢ dialed vs +66¢ measured; coarse right, fine wrong). Aligner pitch head predicts the transpose jointly from (bed key, acappella key), not per-clip regression.

The `aux.db` holding schema (release years, Last.fm, Billboard, Spotify charts,
BB-track ↔ chart-entry pairings, set views, headline results) is documented at
the bottom of [findings.md](corpus_empirics/findings.md#auxiliary-research-database).
Rebuild via [corpus_empirics/aux_db_sync.py](corpus_empirics/aux_db_sync.py).
