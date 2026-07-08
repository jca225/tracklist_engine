# Alignment failure analysis — BB11 + BB12

Where the aligner falls short, *why* (impact-weighted, single binding cause per
span), and what to do about it. Reproduce:

```bash
venvs/audio/bin/python -m eda.alignment.failure_analysis.build_span_table   # -> out/span_table.csv
venvs/audio/bin/python -m eda.alignment.failure_analysis.analyze            # -> out/analysis_report.txt
```

## Method (and one important data caveat)

- **Unit:** one row per predicted span, matched to the nearest same-recording GT
  row (the exact logic `score_timeline_vs_gt.py` uses — aggregates reproduce the
  harness identically: BB12 identity 84% / set_start median 6.3 s, BB11 83% / 7.9 s).
- **Timelines scored:** `out/<set>_predicted_timeline_lt.json` (looptrace variant —
  the only on-disk timelines that carry `ref_segments`, so trajectory is scorable).
- **Headline metric:** `traj_strict` = fraction of a span's GT seconds whose decoded
  ref lands within 2 s. Impact is weighted by **GT-seconds lost** = `duration·(1−traj)`,
  not span count, so a long multiseg span counts more than a 10 s stab.
- **Instance-ambiguity** comes from the cached `looptrace/out/audit_<set>.json`
  (`frac_distinct`), not a fresh HuBERT pass.

> **⚠ AXIS SOURCE (corrected).** The `claimed_stem` baked into the predicted-timeline
> JSON is the materialized `set_track_slots` value, corrupted by the row-text
> `(Instrumental)`/`(Acappella)` drop bug (`project_claimed_stem_rowtext_fix`,
> fixed 888caca but not yet re-materialized into these timelines). It reports only
> **2 instrumentals per set**; the hand-labeled GT has **25 (BB12) / 26 (BB11)**.
> This analysis therefore takes the axis from the **matched GT row**, never the
> timeline. Every per-stem scorecard in the repo that reads the timeline's
> `claimed_stem` (including `score_timeline_vs_gt`'s own stem breakdown) is
> contaminated by this bug — see Finding 1.

## Corpus composition (by GT truth, 283 matched spans / 15 433 GT-seconds)

| axis | spans | GT-sec | % of corpus-seconds | mean traj |
|---|---|---|---|---|
| acappella | 175 | 7 871 | **51%** | **11%** |
| regular | 66 | 4 244 | 28% | 27% |
| instrumental | 42 | 3 318 | 21% | 18% |

Acappella is **half the mix by time and the worst-decoded axis.** 85% of all
GT-seconds are currently lost (13 176 / 15 433).

## Binding cause of loss (dependency-ordered; one cause per span)

| cause | share of loss | traj in bucket | tractability |
|---|---|---|---|
| **decode-residual** (right song, placed, routed — still wrong ref offset) | **39%** | 27% | hard (core wall) |
| **placement** (set_start off >15 s → decode window misses) | **37%** | 12% | medium (partially shipped) |
| **mis-route** (acappella sent to chroma via stale `claimed_stem`) | **9%** | 1% | **easy — data fix** |
| identity (wrong recording) | 6% | 5% | medium / upstream |
| tempo/octave (odd stretch) | 4% | 5% | low priority |
| instance-ambiguity (distinct-take repeat) | 3% | 17% | hard |
| loop | 2% | 12% | hard |

Cross-set consistent: decode-residual (BB12 40% / BB11 37%) and placement (35% /
39%) dominate in **both** sets — this is structural, not set-specific.

---

## Findings

### 1. A data bug is silently costing ~9% of loss and hiding instrumentals entirely
The pipeline routes features/decoders on `claimed_stem` (acappella→HuBERT,
else→chroma). With the stale-materialize value:
- **39% of true acappellas (69/175) are routed as "regular" → chroma.** Those
  score **traj 2%** vs **17%** for correctly-routed acappellas. This is a data
  problem masquerading as a modelling problem.
- **90% of true instrumentals (38/42) are mislabeled** (mostly as regular). Routing
  is unaffected (regular & instrumental both use chroma), but every prior per-stem
  scorecard counted only ~2 instrumentals — the axis was **invisible**, not fine.
- **Fix (cheap, high-ROI):** re-materialize `claimed_stem` on pi (the parser fix is
  already committed), then re-run `infer` + `joint_ref_decode` so routing is correct,
  and **re-run this analysis.** The current decode-residual/placement numbers are
  *pessimistic* — they include damage from 69 mis-routed acappellas that the data fix
  removes. This must happen before trusting the modelling-side numbers below.

### 2. Placement is necessary but nowhere near sufficient
Spans placed within 15 s still decode at only **19% traj**; spans placed worse than
15 s decode at **9%**. So better placement caps out low — the residual decode error
(Finding 3) is the real ceiling. Placement still accounts for 37% of loss because
98 spans are placed >15 s off (p90 set_start ≈ 51–54 s). The open tail is **acappella
placement off the full mix**, where the landmark fingerprint is weak; per-stem HuBERT
placement (`--stem-placement`) is partially shipped and cuts median but not the p90 tail.

### 3. The core wall is acappella ref-offset under repeats (decode-residual)
Ref-offset MAE on straight clips is a clean X-ray of the decoder:

| set · axis | n | median err | <2 s |
|---|---|---|---|
| BB12 regular | 11 | **0.1 s** | 73% |
| BB11 regular | 15 | 0.8 s | 53% |
| BB12 acappella | 33 | **33.9 s** | 9% |
| BB11 acappella | 30 | **38.4 s** | 27% |

Regular/instrumental ref-offset is essentially solved (chroma matched-filter, sub-second).
Acappella is off by **30–40 s median** — the "which chorus" repeat-instance problem.
This is the largest modelling lever (acappella multiseg alone = **34% of all loss**),
and also the hardest: `looptrace/NOTES.md` lists six dead threads (single-window
matched filter, Viterbi-over-HuBERT, GT-start soft prior, backward-jump penalty,
post-hoc discriminative re-selection, evidence-rate router). The remaining honest
lever is a **learned instance-selection model**, gated on a third complete GT set.

### 4. Odd-ratio and loop spans decode near-zero, but are small impact
Odd-ratio (n=46, tempo median 1.30 — genuine odd stretches, only 2% near half/double)
decode at 3–5%; loops at 1–12%. Together ~6% of loss. Not octave-fold artifacts, so
the existing octave-band tightening won't help; low priority vs Findings 1–3.

### 5. Coverage / identity loss is partly upstream
BB12 has **11 GT recordings never matched by any predicted span** (BB11: 0) — these
are ingest/tokenizer losses (hand-added online-candidate acappellas with no id-map,
w-layer mistags), invisible to the decoder. Identity is 6% of loss; BB11 higher
(9%) than BB12 (4%), i.e. cross-set transfer degrades identity somewhat.

---

## Re-measure: corrected routing (local, 2026-07-08)

Confirmed the mis-route cost by relabeling both timelines' `claimed_stem` from GT and
re-running `joint_ref_decode --decoder looptrace` locally (reproducers:
`relabel_stems.py` → `_gtstem.json`; table `out/span_table_gtstem.csv`; report
`out/analysis_report_gtstem.txt`). This **isolates decode routing** — set_start for
the formerly-mis-routed acappellas was still placed by `infer` without HuBERT
stem-placement (they read `regular`), so these are a **lower bound**; a full `infer`
re-run would lift further.

| metric | mis-routed baseline | corrected routing | Δ |
|---|---|---|---|
| **acappella traj (all)** | 11% | **21%** | **+10 pp (~2×)** |
| acappella multiseg | 13% | 21% | +8 |
| acappella linear | 17% | 29% | +12 |
| acappella oddratio | 3% | 11% | +8 |
| acappella loop | 1% | 17% | +16 |
| acappella ref-offset (BB12 median / <2 s) | 33.9 s / 9% | 23.4 s / 30% | better |
| placement ≤15 s → traj | 19% | 27% | +8 |
| total GT-seconds lost | 85% | 81% | ~700 s recovered |
| instrumental axis | invisible ("2 spans") | n=42, traj 21% | now measurable |

**Takeaways:** (a) the data bug was a real, large lever — correct routing ~doubles
acappella trajectory, and this is a floor. (b) The **core wall persists**: even
correctly routed, 81% of GT-seconds are still lost, decode-residual is 45% of loss at
31% traj, and acappella multiseg remains the single biggest bucket (34% of loss, 21%
traj). Fixing the data does not fix the "which chorus" problem — that's still the
modelling prize (Finding 3). (c) BB11 acappella ref-offset barely moved (39 s) because
only 69/147 BB11 tracks have a vocals stem on disk, so fewer got HuBERT ref audio — a
stem-coverage gap, not a decoder gap.

---

## Prioritized solutions

1. **[data, do first — now quantified] Land `claimed_stem` fix on `main` → deploy →
   re-materialize → re-infer.** Measured to recover ~+10 pp acappella trajectory (a
   floor), un-hide the instrumental axis, and de-contaminate every per-stem scorecard.
   The fix (888caca) is on feature branches only, **not `main`**; pi deploys from
   `main`, so this needs a merge/cherry-pick + `make deploy` + corpus-wide
   `tokenizer.materialize`, then a full `infer` re-run (which also fixes the acappella
   **placement** routing this local re-measure left untouched). Lowest effort, highest
   certainty.
2. **[modelling, biggest prize] Acappella ref-offset instance selection.** 34% of loss
   sits in acappella multiseg at 13% traj / ~35 s median error. Six decode-layer
   threads are dead; the live lever is a learned selector over {HuBERT diagonal
   evidence, fiber μ/ambiguity, fp sharpness} — needs the third GT set for leave-one-set-out.
3. **[placement, incremental] Close the acappella set_start p90 tail** off the full
   mix (weak fingerprint). Still 31% of loss after routing is fixed; upside is bounded
   (well-placed spans still only 27% traj), so it's a secondary lever behind #2 — but a
   full `infer` re-run (part of #1) will re-place the formerly-mis-routed acappellas via
   HuBERT stem-placement and re-measure the true placement gap.
4. **[upstream] Recover the 11 never-matched BB12 recordings** (id-map for
   online-candidate acappellas; w-layer stem tagging) — ingest/tokenizer, not aligner.

## Artifacts
- `out/span_table.csv` — 303 rows, one per predicted span (both sets); the reusable dataset.
- `out/analysis_report.txt` — full stratification / attribution / diagnostics dump.
- `build_span_table.py` / `analyze.py` — reproducers (read-only; import harness scoring).
