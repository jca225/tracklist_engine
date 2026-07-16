# Grammar-coverage fingerprint — design

**Date:** 2026-07-16 · **Branch:** `cotrain-grammar-coverage` · **Status:** approved (owner delegated methodology)

## Purpose

First deliverable of the **co-training data arm** (state-of-record §3, decision #9/#11):
select which of the ~41k scraped DJ sets to download next so the aligner's training
substrate spans the **DJ-move grammar**, not popularity. The current ~561 downloaded
sets are popularity-seeded → EDM/mashup-skewed. This tool produces a **grammar-coverage
map** (where the downloaded corpus is thin vs the full corpus) and a **stratified,
ingestable candidate list** ranked by how much each set fills the thin corners.

Supersedes the popularity-scoring in `eda/alignment/generalization/rank_ingest_queue.py`
as the *selection* axis (that script's fetchability/coverage machinery is reused).

## Non-goals (YAGNI)

- No downloads, no acquisition-queue wiring, **zero canonical mutation** — read-only.
- No Tier-2 audio-only moves (loops/jumps/tempo-ride) — those are invisible in metadata
  and are revealed by probes *after* download (a later sub-project).
- The download executor is a separate sub-project and wants the ingest bug-fixes first.

## Home & reuse

`eda/alignment/generalization/grammar_coverage.py`, sibling to `rank_ingest_queue.py`.
Reuse: `_fold` (diacritic-safe matching), the read-only `ssh pi-storage sqlite3 -readonly`
pattern (but pin `encoding="utf-8"` per the ingest audit A2 — do not repeat the mojibake
boundary bug), `out/*.tsv` output convention. Not `lab/` (that's the deferred lab).

## Fingerprint (one row per set, all ~41k, read-only from `dj_sets` + `set_track_slots`)

| Field | Definition | Proxies |
|---|---|---|
| `density` | `total_tracks / play_time_min` (parse `"1h 28m"`) | chops/overlay vs straight/blend |
| `w_frac` | `mean(is_concurrent)` | mashup / overlay intensity |
| `version_frac` | frac `claimed_version` ≠ `original` | remix/edit/rework/VIP/bootleg |
| `stem_frac` | frac `claimed_stem` ∈ {acappella, instrumental} | vocal/instrumental layering |
| `id_frac` | frac `title LIKE 'ID%'` | unreleased/ID material |
| `cue_gap_med`, `cue_gap_cv` | median + CV of `cue_time_seconds` spacing | transition density |
| `styles` | `dj_sets.styles` | genre |
| `has_audio` | `set_audio` present | already downloaded (coverage baseline) |
| `fetchable` | has media link (SC/YT), not yet `set_audio` | ingest candidate gate |

**Stratification axes = the strongest 4:** `w_frac, version_frac, stem_frac, density`.
`id_frac, cue_gap_*, styles` are descriptive/secondary (kept in the row, not primary
grid axes) to avoid a noise-sparse high-dimensional grid.

## Coverage map

1. **Per-dimension:** for each of the 4 axes, bin (quantile bins) and compare the
   **downloaded** subset (`has_audio`) distribution vs the **full corpus** → flag bins
   where `downloaded_share / corpus_share` is lowest (thin bins).
2. **Joint grid:** bin the top-3 (`w_frac × version_frac × stem_frac`) into a small grid
   (e.g. 3×3×3) → flag cells where downloaded/corpus coverage is thinnest (catches
   combined moves like mashup+acappella that per-dimension misses).

## Stratified candidate list

Score each **fetchable & not-downloaded** set by an **inverse-coverage fill weight**:
sum over its bins of `1 / (downloaded_coverage_of_that_bin + ε)`, so sets landing in
starved corners rank highest. Tie-break with light data-quality (id_rate, audio-coverage
of constituents) from the rank_ingest_queue signals. Output `out/grammar_coverage.tsv`.

## Outputs (artifacts only)

- `out/grammar_coverage.tsv` — per-set fingerprint + fill-weight, candidates ranked first.
- Printed **thin-corners summary**: the starved per-dimension bins + joint cells, with the
  top-N candidate sets that fill each.
- (optional) `out/grammar_coverage_map.json` — the coverage histograms/grid for reuse.

## Testing

- Pure functions (`play_time → minutes`, fraction computations, binning, fill-weight)
  unit-tested with fixture rows. The SQL fetch is I/O at the edge (integration-run once
  against pi-storage, results eyeballed). Diacritic folding covered by a RÜFÜS case.

## Verification

Run against pi-storage (DB now free post-box-destroy), confirm: row count ≈ 41k, the
downloaded-vs-corpus skew is visible (expect EDM/mashup over-represented), and the thin
corners are plausible (e.g. low-density straight-blend sets, non-mashup genres).
