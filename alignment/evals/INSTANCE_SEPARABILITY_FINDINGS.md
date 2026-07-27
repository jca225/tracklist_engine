# Oracle instance-selection separability — findings (Task A)

**Date:** 2026-07-18 · **Sets:** BB12 (`1fsnxchk`), BB11 (`2nvzlh2k`) · n=2
**Harness:** `evals/instance_separability.py` (+ `tests/…/test_instance_separability.py`)
**Spec:** `docs/superpowers/specs/2026-07-18-instance-separability-design.md`
**Verdict (one line):** the recoverable acappella instance gap is captured almost
entirely by a **trivial positional prior — play the earliest fiber instance
(~0.93, transfers)** — while the three frozen content features
{HuBERT diagonal, fiber μ, fp} do **not** separate the correct instance beyond
chance. So: **STOP on a learned content-feature selector; the win is a one-line
earliest-instance tie-break that needs no BB10.**

## What was measured

Within the *recoverable* population (GT acappella rows the frozen looptrace decoder
placed in the right fiber but wrong instance — where the ~22 pp strict→fiber
headroom lives), for each row we enumerate the GT fiber's member instances
(`fiber_intervals`), label the one containing GT `ref_start_s` as correct, and ask:
does any of {HuBERT matched-filter score at the instance offset, fiber soft
membership μ, landmark vote-mass at the offset} **rank the correct instance top**,
and does a fitted ranker **transfer** BB11↔BB12? Scoring is **tie-fair** (expected
credit under random tie-break) so list order cannot leak. Full denominator honesty:
fraction is over *selectable* rows only (GT fiber ≥2 members).

## Population

| set | gt_acap | recoverable | selectable | single-member | mean members |
|-----|--------:|------------:|-----------:|--------------:|-------------:|
| BB12 (1fsnxchk) | 97 | 39 | 18 | 21 | 2.67 |
| BB11 (2nvzlh2k) | 91 | 24 | 11 | 13 | 2.73 |

## Separability — fraction of the strict→fiber gap recovered by top-1 (tie-fair)

| feature | BB12 | BB11 | chance |
|---------|-----:|-----:|-------:|
| **earliest instance** | **0.944** | **0.909** | 0.394 |
| latest instance | 0.000 | 0.000 | 0.394 |
| fiber μ | 0.583 | 0.227 | 0.394 |
| fp vote-mass | 0.472 | 0.364 | 0.394 |
| HuBERT diagonal | 0.278 | 0.364 | 0.394 |
| equal-weight combo | 0.361 | 0.409 | 0.394 |

- **Position dominates.** The correct instance is the *earliest* fiber member in
  17/18 (BB12) and 10/11 (BB11) rows. `latest` is 0/… — the signal is direction-
  specific and strong on both sets.
- **The content features do not separate.** μ/fp/HuBERT hover at or below chance,
  and none is consistent across sets (μ 0.58 vs 0.23; HuBERT below chance on BB12).
- The earlier headline "μ = 0.778 on BB12" was an **artifact**: μ ties (repeats have
  near-identical membership) were silently resolved to the first-enumerated =
  *earliest* instance. Tie-fair scoring removes it and μ drops to 0.58 / 0.23.

## Transfer — fitted 3-feature logistic ranker (LOSO, both directions)

| train → test | top-1 (tie-fair) | fitted weights (hubert, μ, fp) |
|--------------|-----------------:|--------------------------------|
| BB12 → BB11 | 0.273 | (−0.19, **+1.12**, +0.02) |
| BB11 → BB12 | 0.333 | (−0.98, **−1.38**, −0.30) |

Both directions land **below chance (0.394)**, and the fitted **μ weight flips sign**
between sets (+1.12 vs −1.38). A content-feature selector not only fails to beat
chance out-of-set — the two sets disagree on the *direction* of every feature. This
is the decisive negative: the three features carry no transferable instance signal.

## Decision (against the pre-registered rule)

- **STOP — learned content-feature selector over {HuBERT, μ, fp}.** Separability is
  at/below chance tie-fair and transfer is below chance both directions with sign
  reversal. Per the spec's stop rule, the selector is bounded and **BB10 will not
  rescue it** — do not label BB10 *for this purpose*, and do not build the fitted
  selector. (This retires the module's "learned selector over {HuBERT diagonal, μ,
  fp sharpness}" lever noted in `alignment_prototype/CLAUDE.md`.)
- **GO — earliest-instance positional prior (unexpected, free).** A parameter-free
  "pick the earliest fiber instance" rule recovers ~0.93 of the instance gap and
  **already transfers at n=2** (0.944 / 0.909), needing no fit and no BB10. It
  matches the independently-observed corpus prior that DJs play the first chorus
  (`mashup_grammar_prior`: vocal ≈ first chorus).

## Recommended next step (not done here — needs the e2e scorecard)

Test an **earliest-fiber-instance (ref-position) tie-break** in the acappella
decoder and measure via `make scorecard`, rather than investing in a content
selector. The current decoder's within-fiber tie-break is a *continuity/warp*
prior (nearest-to-previous), which is **not** the same as earliest-in-ref — so this
is a genuinely untested lever. Expected to convert a large share of the acappella
strict→fiber gap (instance selection is ~34% of all alignment loss). Caveat: this
measurement is on the recoverable subpopulation; net e2e effect (including rows the
decoder already gets right) must be confirmed on the scorecard before adoption.

## Caveats

- **n=2, small rows (18 / 11 selectable).** The earliest-instance result is strong
  and consistent, but confirm on BB10 when labeled — that confirmation is a **re-run,
  not a labeling task for a selector** (this harness drops BB10 in as a third file).
- The **content-feature negative is robust**: below chance on both sets *and* sign-
  flipping weights — not a small-sample wobble in one direction.
- fp on Roformer-separated vocals is known noise-dominated (looptrace NOTES); its
  ~chance result is expected, not a bug. HuBERT's failure is the scale-invariance of
  the matched filter: near-identical fiber siblings are genuinely indistinguishable
  by normalized correlation (verified in the unit tests).

## Artifacts

- `evals/instance_separability.py` — harness (population → candidates → features →
  tie-fair separability + transfer; persists per-candidate dataset).
- `out/instance_separability/{1fsnxchk,2nvzlh2k}_candidates.jsonl` — per-candidate
  `{features, is_gt_instance}` dataset (BB10 drops in as a third file).
- `out/instance_separability/summary.json` — the tables above.
