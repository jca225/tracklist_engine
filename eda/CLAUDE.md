# eda/ — exploratory analysis (alignment-support consumer)

`eda/` is not a chain stage — it reads from multiple stages and produces
findings, not pipeline state. After the 2026-07-12 lab split, what remains here
is **alignment-support** analysis (the aligner imports from `eda/alignment/`):

- **`alignment/`** — mix structure analysis (MERT probes, section/event
  boundaries) **that the algorithmic aligner depends on**:
  `workspaces/alignment_prototype/mert_store.py` imports
  `eda.alignment.mert_vectors`, and `eda/alignment/failure_analysis/` **is the
  scorecard** (`make scorecard`), importing `path_decode` / `score_timeline_vs_gt`
  from the aligner. See [alignment/README.md](alignment/README.md) and
  [docs/aligner_attention_design.md](../docs/aligner_attention_design.md). Not the
  aligner itself (`workspaces/alignment_prototype/`).
- **`queries/`** — ad-hoc query scratch.
- **`common.py`** — shared DB access + DataFrame loading + pydantic_ai agent
  integration, used by the notebooks (`eda.ipynb`, `set_structure.ipynb`,
  `tokenizer.ipynb`).

> **Moved to `lab/` (2026-07-12):** `corpus_empirics/`, `information_dynamics/`,
> and `audience_prior/` — the *music-understanding* research (north-north star),
> not alignment-support. See [lab/CLAUDE.md](../lab/CLAUDE.md).

## Running

Jupyter notebooks use [common.py](common.py) for shared DB access. Scripts assume
`data/analysis/` and `data/db/` paths **relative to repo root** — run them from
the project root, not from the subfolder.
