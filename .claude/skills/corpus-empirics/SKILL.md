---
name: corpus-empirics
description: Scaffolds a new Big Bootie / corpus-empirics analysis following the repo's house pattern — script in eda/corpus_empirics/bb_*.py, results persisted to aux.db `analysis_results` table, and a findings section appended to eda/corpus_empirics/findings.md (CLAUDE.md keeps only a one-line pointer). Use when the user wants to run a new empirical analysis on the BB corpus (popularity, era, chart, listener, set-views, etc.), correlate a new signal against set views or another target, or add a finding to the corpus-empirics section. Triggers on phrases like "analyze X across BB", "is X correlated with views", "add a new bb_ analysis", "test whether X drives Y in the corpus".
---

# Corpus Empirics Analysis

Repeatable pattern for adding a new empirical finding to the Big Bootie corpus analysis. Follow this exactly — the format is load-bearing for the published findings in CLAUDE.md.

## The pattern, end to end

A corpus-empirics analysis has three artifacts:

1. **Script** at `eda/corpus_empirics/bb_<name>.py` — pure Python, reads `data/db/music_database.db` (main, scraper-side) + `data/analysis/aux.db` (research signals), computes a finding, persists headline metrics to `aux.analysis_results`.
2. **Persisted metrics** in `aux.analysis_results` under a unique `analysis_name` (e.g. `bb_<name>_v1`). One row per (metric, group_key) tuple. Re-running the script upserts.
3. **Findings section** in `eda/corpus_empirics/findings.md`, in the established format (see template below). CLAUDE.md's `## Corpus empirics` keeps only a one-line summary + pointer — update it if you add a headline finding worth surfacing there.

Don't skip any of the three. The script without persistence makes the finding unreproducible; the persistence without the CLAUDE.md section makes it undiscoverable for the next session.

## Step 1 — Check what already exists

Before writing a new script, check whether an existing one covers part of the question:

```bash
ls eda/corpus_empirics/bb_*.py
```

Existing analyses (as of this skill's writing):
- `bb_era_orthogonality.py` — release-year independence of acap/instr roles
- `bb_popularity.py` — acap vs instr popularity asymmetry (Last.fm + Hot 100 year-end)
- `bb_set_views_analysis.py` — per-volume views vs aggregated track features
- `bb_weekly_chart_analysis.py` — Hot 100 weekly history (peak-position, weeks-on-chart)
- `bb_spotify_charts.py` + `bb_spotify_chart_analysis.py` — kworb.net Spotify Top 200 ingestion + regression
- `bb_han_weighted.py`, `bb_demographic_baseline.py` — listener-side experiments

Check what tables/columns exist in aux.db before writing new SQL:

```bash
ssh pi-storage 'sqlite3 /mnt/storage/data/db/music_database.db ".schema"' | head -100
sqlite3 data/analysis/aux.db ".tables"
sqlite3 data/analysis/aux.db ".schema <table>"
```

## Step 2 — Write the script (`eda/corpus_empirics/bb_<name>.py`)

Follow the structure used in `eda/corpus_empirics/bb_set_views_analysis.py` and `eda/corpus_empirics/bb_popularity.py`:

```python
"""<One-paragraph statement of the question being asked, the data being
used, and the headline finding's interpretation. This docstring is the
"abstract" — write it as if for an external reader.>
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

AUX_DB = Path("data/analysis/aux.db")
MAIN_DB = Path("data/db/music_database.db")
ANALYSIS_NAME = "bb_<name>_v1"  # version suffix lets you supersede later


def pearson(xs, ys):
    n = len(xs)
    if n < 2: return float("nan")
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((x-mx)*(y-my) for x,y in zip(xs, ys))
    dx = (sum((x-mx)**2 for x in xs))**0.5
    dy = (sum((y-my)**2 for y in ys))**0.5
    return num/(dx*dy) if dx and dy else float("nan")


def main() -> int:
    conn = sqlite3.connect(MAIN_DB)
    conn.execute(f"ATTACH DATABASE '{AUX_DB}' AS aux")
    conn.row_factory = sqlite3.Row

    # ... compute the analysis ...

    # Persist headline metrics
    findings = [
        # (metric, group_key, value)
        ("pearson_x_y", "all", float(...)),
        ("n_samples",   "all", float(n)),
    ]
    cur = conn.cursor()
    for metric, group, val in findings:
        cur.execute("""
            INSERT INTO aux.analysis_results
              (analysis_name, metric, group_key, value)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(analysis_name, metric, group_key) DO UPDATE SET
              value = excluded.value, computed_at = CURRENT_TIMESTAMP
        """, (ANALYSIS_NAME, metric, group, val))
    conn.commit()
    print(f"persisted {len(findings)} metrics to aux.analysis_results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Conventions to follow:

- **Style is functional Python, no classes.** Match the existing scripts — `from __future__ import annotations`, pure stat helpers (`pearson`, `partial_pearson`, `spearman`), `dict[str, ...]` typing.
- **Stats math is hand-rolled** in the existing scripts (no numpy/scipy dependency). Continue that unless the analysis genuinely needs scipy.
- **Always include `n_<group>` rows** in the persisted findings so the reader sees the sample size.
- **`group_key`** distinguishes sub-population analyses ("acapella", "instrumental", "modern_window", "all"). Use `"all"` when the metric is global.
- **`analysis_name`** is unique and versioned (`bb_<name>_v1`). Bump to `_v2` if you change the metric definition; don't overwrite v1 silently.
- The script is **idempotent**: re-running upserts. Don't make it append-only.

## Step 3 — Run it

```bash
python eda/corpus_empirics/bb_<name>.py
```

Output should be human-readable: a header, the correlations table, and a final "persisted N metrics" line. Capture the output verbatim — you'll quote numbers from it in the CLAUDE.md section.

## Step 4 — Add findings to CLAUDE.md

Append a new `###` subsection under `## Corpus empirics` in `CLAUDE.md`. The template is non-optional — every prior finding follows it, and the consistency matters for readers (including future Claude sessions). Use this template:

```markdown
### <Headline finding as a declarative sentence>

<1–2 sentence framing of the question and method.>

<Markdown table of headline numbers — feature, r, r², interpretation —
or a quadrant table where appropriate. Bold the strongest row.>

<Prose interpretation: what does the strongest finding mean, what
alternative explanations were ruled out, what the magnitudes say.>

**Implication for modeling**: <how this should shape the downstream
mashup-sequence model, per-user prediction head, or feature engineering.
This is the load-bearing paragraph — every prior finding has one.>

**Sample-size caveat** (if n < 50): <wide CIs, directional-only,
what would tighten it>.

Reproduction: [eda/corpus_empirics/bb_<name>.py](eda/corpus_empirics/bb_<name>.py). Results in
`aux.analysis_results` under `analysis_name='bb_<name>_v1'`.
```

**Style rules from existing sections:**

- Lead with the *conclusion*, not the question. Header is a declarative sentence ("Set popularity is driven by chart-hit-vocal *density*, NOT instrumental popularity").
- Numbers are presented in tables wherever 2+ comparable. Bold the row that carries the finding.
- Always include an `Implication for modeling` paragraph — the section's purpose is to constrain downstream model design.
- Include a sample-size caveat when n is small (the existing sections do this for n=20–23).
- Include a `Reproduction:` line with a clickable script path and the `analysis_name` to query.

## Step 5 — Commit

One commit per analysis, message style:
```
feat(analysis): <one-line finding>
```
Match the existing commit log (`feat(analysis): Spotify Top 200 popularity signals via kworb.net`).

## When to bump to v2 (instead of writing a new analysis)

If you're revisiting the same question with a refined method (broader chart cut, different control, larger n), bump the `analysis_name` to `_v2` and add a *new* findings subsection rather than editing the old one. The CLAUDE.md history of refined findings is itself a valuable artifact — see how `bb_weekly_chart_v1` and `bb_spotify_chart_v1` each got their own section that refines but doesn't overwrite the year-end finding.

## Anti-patterns (don't do)

- ❌ Presenting feature-engineered R² values from n<50 as if they were findings without the sample-size caveat. (Memory: `feedback_small_sample_regressions` — user has flagged this.)
- ❌ Adding a release-year-proximity feature to a mashup-pair-scoring head. (Era is independent in this corpus — see the era-orthogonality section.)
- ❌ Using `Spotify popularity` (the look-back-biased streaming number) as a popularity proxy. Use the at-release-time chart signals already in aux.db.
- ❌ Skipping the `Implication for modeling` paragraph. Without it, the finding is just trivia.
- ❌ Editing an old section instead of adding a v2. Refinements are additive.
