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

## Update 2026-07-08 — oracle↔e2e gap decomposed: placement, not decode, is the shippable lever

The decode-residual share above (45% of *total* loss vs 100%) is real but it is
**not** the biggest lever for closing the gap between what we ship and what the
decoder can already do. Decomposing the acappella oracle(37%)↔e2e(16%) gap with
`score_timeline_vs_gt`'s own pairing (`oracle_e2e_gap.py` →
[ORACLE_E2E_GAP.md](ORACLE_E2E_GAP.md)):

- **91% of the oracle→e2e gap is "content never entered the scored window"**
  (placement/windowing/pairing); only 9% is in-window decode error. When the
  right content is in the window, e2e ≈ oracle (BB12 38% vs 40%).
- Repairing **set_start placement** to oracle quality lifts pooled acappella e2e
  **15.7→30.2%** — the largest single lever, **> decode-residual (+11.5 pp)** and
  **not gated on BB10.** The decode wall (oracle ~37%) is a separate, higher
  ceiling; the instance-selection model raises *that*, and still needs BB10.
- The correspondence/scoring buckets (w-layer 1:1 starvation + span extent) total
  ~10.6% of GT-sec — real but not the story; the grid numbers are honest to ~1 pp.

So the full `infer` re-run's HuBERT `--stem-placement` on the formerly mis-routed
acappellas (item 1 below) is *also* the top modelling-adjacent lever, and BB11's
placement gap is partly the 15-reference-track vocals-stem coverage hole. Priority
order below is unchanged in sequence but **placement (was #3) is now the primary
e2e lever, ahead of the BB10-gated instance selector (#2)** for near-term gains.

## Re-measure 2 — full post-fix re-run (2026-07-08 evening, BB12): solution 1 DONE

The data fix is landed end-to-end: `f678f3a` on `main` (identical patch to
888caca, verified by patch-id), pi deployed, corpus-wide
`tokenizer.materialize` completed 2026-07-08 (BB12 slots 84 acap / 19 instr /
62 regular; corpus instrumentals 2552). BB12 then got a clean full re-run:
`infer` (fresh spine over ssh; HuBERT `--stem-placement` now fires for the
formerly mis-routed acappellas) → `joint_ref_decode --decoder looptrace` →
scored at `out/1fsnxchk_predicted_timeline_postfix_lt.json`:

| metric | mis-routed baseline | corrected routing (floor) | **full re-run** |
|---|---|---|---|
| acappella traj | 11% | 21% | **31%** |
| HEADLINE multiseg+loop | 26% | — | **38%** |
| regular traj | 27% | — | **51%** |
| instrumental traj | invisible | 21% | **33%** (n=21) |
| acappella ref-offset (median / <2 s) | 33.9 s / 9% | 23.4 s / 30% | **15.3 s / 31%** |
| set_start (median / <15 s / p90) | 6.6 s / 63% / 61 s | unchanged | **5.0 s / 73% / 48.9 s** |
| identity | 84% | — | 84% |

The routing-only re-measure was indeed a floor: re-placing with the corrected
axis ~tripled acappella trajectory vs the contaminated baseline and lifted
every placement figure — consistent with the oracle↔e2e decomposition above
(placement is the shippable lever).

**BB11 (cross-set transfer, + `--instr-stem-placement`,
`out/2nvzlh2k_predicted_timeline_postfix_lt.json`):** identity **84%** (was 79%
in the transfer scorecard), set_start median 7.1 s / <15 s 65% / p90 53.8 s,
headline multiseg+loop 30%, regular traj 49%. Two per-axis stories:
- **instrumental traj 46%** (ref-offset median 7.8 s) — the stem↔stem fp channel
  (54300c3) delivers beyond BB12's 33% *without* the channel; the axis went from
  invisible to the second-best stem on a transfer set.
- **acappella traj 18% / ref-offset 34.3 s — barely moved**, as predicted: BB11's
  acappella gap is the vocals-stem coverage hole (69/147 refs have a vocals stem
  on disk), a stem-coverage problem, not routing or decode. That backfill is now
  the cheapest BB11-specific lever.

## Prioritized solutions

1. **[data — DONE 2026-07-08, see Re-measure 2] Land `claimed_stem` fix on `main` → deploy →
   re-materialize → re-infer.** Measured to recover ~+10 pp acappella trajectory (a
   floor), un-hide the instrumental axis, and de-contaminate every per-stem scorecard.
   ~~The fix (888caca) is on feature branches only, **not `main`**~~ — landed as
   `f678f3a`, deployed, re-materialized; BB12 re-inferred (BB11 pending). The floor
   estimate held: the full re-run delivered +20 pp acappella trajectory.
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

---

# EDA 2026-07-09 — failure correlates (north-star framing) + generalization profile

Per-span correlate study on the **post-fix** timelines (`build_span_table
--suffix _postfix_lt` → `out/span_table_postfix.csv`, n=283 matched spans;
cross-checks harness exactly: identity 84/84%, set_start 5.0/7.1 s), scored
against the north-star laws (docs/architecture_north_star.md). Reproducer:
`failure_correlates.py` → `out/failure_correlates.txt`. Corpus numbers from
read-only pi-storage queries (2026-07-09).

## A. Why we fail — new span-level findings

### A1. The largest un-attributed acappella bucket is a SILENT looptrace-empty fallback
`ref_decoder` is set only when looptrace emits segments. On acappella:

| set | looptrace decoded | traj | silently fell back | traj |
|---|---|---|---|---|
| BB12 | 63 | **0.31** | 20 | 0.08 |
| BB11 | 47 | **0.21** | 44 | **0.03** |

Attribution via the run's own `~/aligning` manifest (§6b of the report):
**46/64 fallback spans had the local vocals stem present — looptrace ran and
returned EMPTY** (~1,885 GT-sec ≈ 24% of all acappella GT-seconds); 14 lacked a
local vocals stem; 4 were manifest misses. The span then silently degrades to
the legacy matched filter and scores ~0. This is a *decode-coverage* loss that
prior attribution filed under decode-residual/placement. Levers: (i) emit
looptrace-emptiness as a `Diagnostic` + abstention (north-star laws 5/7), (ii)
investigate why `decode_span` returns empty on stems that exist (quiet/sparse
separated vocals? landmark threshold?), (iii) BB11 refresh: 23 spine vocal
stems landed on pi **2026-07-09, after the Jul-8 run** — re-pull + re-infer
BB11 is cheap and directly feeds the 14 no-local-stem spans.

### A2. `confidence` is anti-calibrated; `start_source` is the real abstention signal
spearman(confidence, traj) = **−0.24 ALL / −0.31 acappella** — dropping the
lowest-confidence spans makes the kept set *worse* (0.20 → 0.15 GTsec-weighted
at 50% abstention). Law 3 ("abstain, never lie") cannot be built on this field.
But the probe that placed the span stratifies cleanly:

| axis · start_source | n | traj | ss_med |
|---|---|---|---|
| acappella · lyrics | 114 | **0.26** | 2.8 s |
| acappella · mert | 32 | **0.04** | 15.1 s |
| acappella · fp | 19 | 0.09 | 6.7 s |
| regular · fp | 57 | 0.30 | 4.1 s |
| regular · mert | 9 | 0.08 | 40.0 s |

**MERT-fallback placement ≈ span lost** (41 spans, all axes). A rule as dumb as
"abstain when start_source==mert" is a calibrated, shippable abstention channel
today, and `start_source` should flow into the Timeline's abstention fields.

### A3. North-star B1 (warp) is a first-order failure dimension; B2 (key) is handled
60% of GT spans are tempo-stretched >2%, **31% >10%**; traj decays
monotonically with |ratio−1| (0.28 → 0.11 from <2% to >12%), rho −0.33 on
acappella — the strongest continuous correlate we have. The warp axis, not
crosstalk, is where decode difficulty concentrates. Conversely re-pitched
acappellas decode *no worse* than unpitched (traj 0.21 vs 0.18, ref-offset
median 16.9 s vs 28.5 s) — the HuBERT key-invariant routing is doing its job.

### A4. Refuted / bounded
- **Crosstalk (GT layer depth) does NOT predict span failure** (rho +0.15
  acappella — mildly *positive*; identity holds 83–92% even at depth ≥3).
- **Long spans are a placement catastrophe:** >90 s spans (n=34, all axes)
  place at ss_med 31–44 s vs ~4–7 s otherwise — the p90 placement tail is
  substantially a long-span problem.
- **Position in set, audibility: no signal** (audible_frac coverage is thin).
- **Stale instance-ambiguity audits:** BB12's `looptrace/out/audit_*.json`
  uses pre-w-layer slot keys → 0/83 join to current slots. Regenerate before
  any learned instance-selection work; frac_clone/distinct is currently
  unusable as a feature.

## B. Generalization — the corpus vs the north-star 20k target

Canonical-DB profile (pi-storage, 2026-07-09; columns audited in the query log):

- **The runnable corpus today is 547 sets, not ~20k.** dj_sets = 41,492;
  **561 have set audio** (the hard input-contract requirement) — and where set
  audio exists, per-set track-audio coverage is already ~93% median. The
  binding constraint on generalization is **set-audio ingest**, not track
  audio and not the aligner.
- **Reference wiring exists only for BB spines:** `is_reference=1` on 425
  track_audio rows corpus-wide; fingerprints 422 recordings; Roformer vocal
  stems 562 (2.9% of track_audio). P5 scale-out needs pick-reference →
  separation → fingerprint backfill over ~19k rows as preflight stages;
  no data blocker, pure compute.
- **The acappella wall is BB-specific, not corpus-relevant first-order:**
  corpus slots are 94.4% regular / 5.0% acappella / 0.6% instrumental;
  per-set acappella share median **3.1%** vs BB11/BB12 at **43/51% (~p98)** —
  BB is an outlier even within Two Friends (their average set: 10.5%). The
  median corpus set lives almost entirely on the *regular* axis, where the
  stack is strongest (traj 51% BB12, fp placement 4 s). BB-trained transfer
  numbers therefore likely **understate** typical-set performance — but our GT
  measures almost nothing about the median set's failure modes; the next GT
  sets should include at least one low-acappella "normal" set.
- BB set length is typical (~p50 ≈ 62 min); stale memory correction: the
  "69/147 BB11 vocals stems" hole is closed in the DB (147/149 as of
  2026-07-09; the *local* pull is what the Jul-8 run was missing).

Companion study: `eda/alignment/low_rank/` (set×track SVD, per-DJ bases,
metadata/ML probes — low-rank worldview test).

---

# Fix round 2026-07-09 (same day) — intervention results & corrections

Findings A1's causal reading was tested by intervention the same day. Two
corrections and one new finding; scripts unchanged, timelines
`out/2nvzlh2k_predicted_timeline_postfix_lt_diskfix.json`.

## C1. CORRECTION: the silent-fallback bucket was confounded — coverage alone lifts ~nothing
`joint_ref_decode` now resolves stem-routed ref audio through
`stem_resolve.resolve_stem` (disk-truth; the manifest hint bit it exactly as
`0960565` documented) and emits `ref_decode_status` per span
(`looptrace | looptrace-empty | skip-no-ref-audio | skip-manifest-miss |
skip-too-short | legacy`) instead of silently degrading. BB11 A/B
(identical scorer, non-fibered): looptrace coverage **47 → 73 spans, zero
un-diagnosed fallbacks** (only `skip-too-short: 12` remains) — but headline
**flat** (21→20%), acappella traj flat (14%), with real small wins:
oddratio traj 5→9%, instrumental ref-offset median 7.8→3.9 s. So A1's
0.03-vs-0.21 gap was **selection bias** (decodable spans were also the
well-placed ones), not a causal coverage loss — consistent with the
oracle↔e2e decomposition: placement gates decode. The fix stands on
robustness/diagnostics grounds (the stale-manifest class can no longer
silently eat spans), not as a headline lever.

## C2. NEW: the REAL residual axis problem — 40 GT-acappella spans still routed `regular`
The claimed_stem fix (solution 1) left a residual 5× bigger than the "~6/set
class-1 gaps" estimate: **BB11 29/92 + BB12 11/83 GT-acappella spans are
mis-axed as `regular` in the post-fix timelines** (traj 0.03/0.06 vs
0.19/0.28 correctly-axed). Verified upstream: the pi slot rows genuinely say
`regular` with no `(Acappella)` row-text — the scrape never knew, so **no
parser can fix these**; routing must come from structure or audio at
inference. Pooled ceiling if routed correctly: ~+3–5 pp acappella traj per
set, plus the (larger) placement-channel effect — mis-axed spans never get
HuBERT/lyrics placement, and lyrics-placed spans run at ss_med 2.8 s.

**Quantified structural prior (unwired — sensor-phase freeze, filed in
looptrace/NOTES.md):** across both sets, **P(acappella | w-layer slot) =
82%** (175/213), **100% of GT acappellas are w-layers** (0 in main slots),
and main slots are 54% instrumental / 46% regular. Slot position is a strong
axis prior the pipeline currently ignores; `set_track_slots.layer_role`
(bed/payload/…) exists in the DB and is consumed nowhere in the prototype.

## C2b. Round 3 (same day) — three more problems, two verdicts

- **Long weaves are 40% of ALL loss.** Spans with GT duration >90 s are 34
  spans (12%) but 5,543 GT-sec (36% of corpus-seconds) and **40% of all
  GT-seconds lost** (sec-weighted traj 0.12 vs 0.24 for ≤90 s). Not a pairing
  artifact — single-appearance long spans still place at ss_med 31.6 s. These
  are finale-style in-and-out weaves (BB12 slot 42: an 839 s GT span placed
  895 s off); one `set_start` + a ±45 s decode band structurally cannot cover
  them even though `ref_segments` could represent them. The lever is
  multi-anchor placement (fp diagonal votes already produce multiple
  candidates) / windowing per slot-appearance — kernel-lane, and now the
  single largest quantified bucket.
- **Synthetic slot rows: CORRECTED — cost is ~1–2 spans, not 5 pp identity.**
  7 BB11 spans predict raw `tlp*` recording ids (`source='synthetic'`, the
  Rvmor sided-row gap). Initial read (29% of identity misses) was wrong:
  the GT yaml inherited the same tlp ids from the spine, the pull fetched
  audio for them, so **5/7 identity-HIT and decode normally** (traj up to
  0.97). Real damage: slots 001/027 have no GT anchor (2 spans), plus the
  namespace pollution itself — tlp recordings can't join `track_metadata` /
  fingerprints / canonical features corpus-wide, and title-only name-match
  is UNSAFE for reconciling them (Gazzo "Nothing To Lose" ≠ VASSY "Nothing
  To Lose"; several have no canonical recording at all, e.g. The Scrantones,
  Rent). Verdict: upstream artist+title reconcile is hygiene + a preflight
  check for new GT sets, **not an aligner lever on BB11**.
- **REFUTED: BB12's fingerprint-coverage gap is not a placement lever.** BB12
  has fingerprints for only 74/139 matched spans' recordings, but no-fp spans
  place *no worse* (ss_med 5.9 vs 4.4 s, traj 0.27 vs 0.25) — w-layer
  acappellas ride lyrics/HuBERT placement, not fp. Backfilling BB12
  fingerprints is hygiene, not a lever.
- **BB12 diskfix A/B: flat**, mirroring BB11 (headline 25→24%, coverage
  cells slightly up: multiseg ≥80%covered 14→16, acappella 14→17). Both sets
  now confirm: decode-coverage repairs don't move the board; placement and
  the long-weave window do.

## C2c. Fix measurements (2026-07-09 evening) — routing floor + long-weave oracle

**Routing fix floor (GT-axis relabel on postfix timelines → re-decode only,
placement held; `_postfix_gtstem_lt` timelines).** Fixing the 40 mis-axed
spans' decode routing alone is worth, vs the identically-scored baseline:

| set | acappella | instrumental | oddratio | headline |
|---|---|---|---|---|
| BB11 | 14→17% | 33→**41%** | 9→16% | 20→22% |
| BB12 | 25→25% | 20→**28%** | 18→16% | 25→25% |

This is the floor — the earlier full re-run (Re-measure 2) showed the larger
gain arrives when the corrected axis also drives *placement* (HuBERT/lyrics
channels) in `infer`. Verdict: **wire the axis fix (w-layer prior or
audio-gate) ahead of infer, not just decode** — decode-only captures the
instrumental win but little acappella.

**Long-weave oracle-window bound (`_longoracle_lt` timelines: >90 s spans get
GT window + GT axis, decode otherwise unchanged).** Sec-weighted traj on the
long spans: **BB12 0.10→0.21, BB11 0.14→0.42**; pooled long bucket
0.12→0.29. Structure of the win:

- **90–300 s spans (3,865 GT-sec) go 0.17→0.41** — many near-perfect
  (0.02→1.00, 0.05→0.91): for this class the *window is the wall*, decode is
  already good enough.
- The two **839 s mega-weaves stay dead** (0.01→0.03) even with oracle
  windows — decode-hard, exclude from the windowing fix; abstain them.
- **4 regressions** (e.g. 0.49→0.02) where the larger window admits rival
  content — any wiring needs an accept-guard (keep the wider decode only if
  path inlier evidence beats the tight decode's, the same comparison the
  slope competition already runs).

**Pooled value if wired: ~+6 pp corpus-wide trajectory** (+5.9 pp from the
90–300 s class alone) — the largest measured unbuilt lever, not BB10-gated.

**Per-axis decomposition of the 90–300 s oracle gain (changes the wiring):**

| true axis | GT-sec | base → oracle | pooled value |
|---|---|---|---|
| instrumental | 1,609 | 0.21 → **0.58** | **+3.9 pp** |
| regular | 1,110 | 0.20 → **0.51** | **+2.2 pp** |
| acappella | 1,146 | 0.08 → 0.06 | **≈ 0** |

**Acappella gets NOTHING from a correct window** — its long-weave wall is
decode-under-repeats, same as its short spans. So do NOT build acappella
window growth. Wiring: (1) instrumental is looptrace-routed → an
evidence-gated window-growth ladder inside `joint_ref_decode` (pads
45/120/240 s; grow while `ev_out_frac` ≥ gate AND the rung's evidence_rate
holds ≥ 0.8× the previous rung — within-span guard, since absolute
evidence-rate floors failed to transfer twice) captures the +3.9 pp lane —
**implemented and A/B'd 2026-07-09 evening (KEPT)**; (2) regular is
legacy-path → needs the `infer`-side multi-appearance anchors from the fp
vote clusters (kernel lane, +2.2 pp).

**Weave-ladder A/B (`_weave_lt` vs `_diskfix` baselines, identical scorer):**
BB12 instrumental traj **20→28%**, headline **24→26%**, from 3 growth events
— span-level: slot 2 (97 s) 0.04→**0.63 (= its oracle bound)**, slot 6 (53 s)
0.00→0.87, slot 30 (61 s) 0.07→0.34 (it also rescues mis-windowed 50–90 s
spans below the "long" cutoff). BB11: 1 growth, +1 pp instrumental.
**Zero regressions on any axis in either set** — acappella/regular
byte-identical; the rate-margin guard held. Remaining long-weave upside:
regular (+2.2 pp, infer-side) and the instrumental spans whose first-rung
`ev_out_frac` never trips the gate.
BB12's GT yaml uses plain numeric slots (`002`) where timelines carry
w-layers (`2w1`); BB11's GT has w-layers. The audit was faithful to GT all
along — the slot-keyed join was the bug. `build_span_table` now joins audits
by `track_id` too (coverage 15→42 spans, BB12 audit regenerated alongside).
With the fixed join: **instance-ambiguity fractions do NOT meaningfully
predict span outcome** (|spearman| ≤ 0.15, n=34) — frac_clone/frac_distinct
are weak span-level features for the learned selector at current n.
