# looptrace — running results

All numbers: per-second segment accuracy (fraction of sampled mix seconds
whose predicted song-time is within tol of GT), acappella spans, oracle
placement (GT set_start), unless noted. "legacy" = the frozen
`path_decode.trajectory_acc` tol=2.0 s. Never tuned on the answer keys.

## Baseline (legacy decoder: Viterbi over HuBERT matched-filter windows)

Strict / clone-aware / repeat-aware at ±0.25 / ±1.0 / ±2.0 s
(`looptrace.eval` on `path_decode --dump`; ±2.0 strict ≡ frozen legacy):

| set | class | n | strict | clone-aware | repeat-aware |
|---|---|---|---|---|---|
| BB12 | ALL | 21 | 37/42/**44** | 37/42/44 | 42/50/56 |
| BB12 | linear | 8 | 62/62/62 | 62/62/62 | 62/62/62 |
| BB12 | multiseg | 7 | 15/24/**27** | 15/24/27 | 19/34/45 |
| BB12 | loop | 1 | 66/81/81 | 66/81/81 | 66/81/81 |
| BB12 | oddratio | 5 | 22/27/32 | 22/27/32 | 39/47/55 |
| BB11 | ALL | 17 | 31/35/**35** | 31/35/35 | 39/45/47 |
| BB11 | linear | 5 | 31/42/43 | 31/42/43 | 31/42/43 |
| BB11 | multiseg | 7 | 10/12/**12** | 10/12/12 | 25/32/34 |
| BB11 | loop | 1 | 0/0/0 | 0/0/0 | 33/35/43 |
| BB11 | oddratio | 4 | 75/75/75 | 75/75/75 | 75/75/75 |

Legacy fiber-aware (path_decode --fibers, HuBERT fibers): BB12 ALL 47%,
multiseg 31%; BB11 ALL 35%, multiseg 12%.

Phase-1 ceilings (AUDIT.md): clone-unwinnable ≈ 0% of GT seconds ⇒ the
honest ceiling is ~100%, not the feared 35–47%.

## Phase 3 — looptrace decoder (landmark Hough + cover DP, 2026-07-06)

`looptrace.run`, oracle placement, same population/metric as the baseline.
Strict / repeat-aware at ±0.25/±1.0/±2.0 s (baseline in parentheses at ±2):

| set | class | strict | repeat-aware | vs baseline strict@2 |
|---|---|---|---|---|
| BB12 | ALL | 26/34/**36** | 35/47/50 | 36 (44) − |
| BB12 | linear | 38/44/44 | 48/54/54 | 44 (62) − |
| BB12 | multiseg | 16/24/**26** | 23/34/36 | 26 (27) ≈ |
| BB12 | loop n=1 | 25/84/94 | 25/84/94 | 94 (81) + |
| BB12 | oddratio | 21/22/26 | 33/46/55 | 26 (32) − |
| BB11 | ALL | 21/38/**38** | 32/57/**60** | 38 (35) + |
| BB11 | linear | 39/60/60 | 39/60/65 | 60 (43) + |
| BB11 | multiseg | 15/21/**22** | 19/36/38 | 22 (12) **+83% rel** |
| BB11 | loop n=1 | 17/20/24 | 33/37/52 | 24 (0) + |
| BB11 | oddratio | 8/43/43 | 46/93/**93** | 43 (75) − |

Read: complementary decoders. looptrace wins BB11 across ALL/linear/
multiseg/loop and finds the right CONTENT far more often (repeat-aware
BB11 ALL 60 vs baseline 47; oddratio 93) but lands on the wrong repeat
instance (strict↔repeat gap) — the Phase-4 lever. Baseline HuBERT still
wins BB12 linear/oddratio. Slope picks correct on 14/21 BB12 spans;
4 weak-evidence spans still pick wrong slopes by small peakiness margins.

## Lever 2 — DP-path-evidence slope selection (2026-07-06)

Top-3 slopes by peakiness compete on tight-tolerance (±0.3 s) path-inlier
evidence with an MDL charge per segment. Strict traj-acc(<2s), oracle
placement (previous looptrace / frozen baseline in parens):

| set | ALL | linear | multiseg | loop | oddratio |
|---|---|---|---|---|---|
| BB12 | **43** (36 / 44) | 50 (44 / 62) | **35** (26 / 27) | 94 (94 / 81) | 32 (26 / 32) |
| BB11 | **44** (38 / 35) | 60 (60 / 43) | 21 (22 / 12) | 24 (24 / 0) | **68** (43 / 75) |

Both sets now beat the baseline on multiseg (35 vs 27; 21 vs 12). BB12
ALL is within 1 pp of baseline; BB11 ALL +9 pp. Two evidence-measure bugs
found on fixtures along the way: cross-candidate median floor
self-annihilates when all candidates are true (jump spans) → random-probe
noise floor; loose-tolerance evidence favors wrong slopes (a 0.6 s tol +
1.5 s kernel lets a 7.5% slope error keep ~16 s of true points) →
±0.3 s path-inlier currency.

## Lever 1 — looptrace|legacy router: NULL (2026-07-06)

LOSO-thresholded routing on `evidence_rate` (`router.py`). After lever 2
the complementarity mostly evaporated: oracle-per-span 48/50 vs
looptrace-only 43/44, and the signal does not transfer across sets
(BB12's best theta applied to BB11 routes good looptrace spans away:
34% vs looptrace-only 44%). Routed never beats the better single decoder.
Verdict: not wired; looptrace-only is the best single decoder overall.

## Lever 3 — residual tiebreak in the DP: inert as specified (2026-07-06)

Built (`residual.py`): zero-sum mel-similarity bonus between repeat-image
rival diagonals, discriminability-weighted, audit-covered frames only,
vote-tie gated. Finding: legitimate small rival groups DON'T occur on
real spans — the Hough candidate set is dense (~24 diagonals) and the
audit-lag matching chains transitively into one degenerate group. With
degenerate groups allowed, the term acts as a GLOBAL mel-verification
prior: BB11 ALL 44→50, oddratio 68→93 (!), BB12 oddratio 32→52, but it
also flips a correct BB12 linear span (051, 100→0) — net +2 spans, one
material regression. With sound grouping (≤4 rivals, tol 0.75 s) it never
fires. Default = strict/inert (no-regression rule). The REAL discovery:
a mel-consistency emission for ALL candidate diagonals (hybrid landmark +
matched-filter evidence in one DP) is the mechanism behind those gains —
future work, needs its own calibration to avoid the 051-style flip.

## Final standing (best config: lever 2, oracle placement, strict <2 s)

| | BB12 lt | BB12 baseline | BB11 lt | BB11 baseline |
|---|---|---|---|---|
| ALL | 43 | 44 | **44** | 35 |
| linear | 50 | 62 | **60** | 43 |
| multiseg | **35** | 27 | **21** | 12 |
| loop (n=1) | **94** | 81 | **24** | 0 |
| oddratio | 32 | 32 | 68 | 75 |

Combined (38 spans): ALL 43 vs 40 baseline; **multiseg 28 vs 20** — the
target class improved ~+45% relative on the honest cross-set average.

## Hybrid mel-consistency emission (2026-07-06): flat overall, default OFF

Bounded (±0.12 share cap) per-candidate mel-verification contrast in the
DP (`segments.mel_emission`, `run.py --hybrid`). Real-mix A/B vs lever-2:
BB11 oddratio 68→95 (the lever-3 mechanism, now bounded and principled)
and BB11 ALL 44→47, but small erosion nearly everywhere else (BB12 ALL
43→41, BB11 multiseg 21→17, BB11 loop 24→0 n=1). Combined ALL 43.7 vs
43.4 — flat. Default OFF per the no-regression rule. Next refinement if
revisited: scale the mel weight by inverse landmark-evidence density
(mel should fill landmark deserts, not argue where landmarks are dense).

## End-to-end (REAL placement) — looptrace wired into the pipeline

`joint_ref_decode --decoder looptrace` routes ACAPPELLA spans through the
landmark decoder (others keep legacy); same timelines/placement both arms;
scored with `score_timeline_vs_gt --fibers` (fiber-aware, within 2 s):

| | BB12 legacy | BB12 looptrace | BB11 legacy | BB11 looptrace |
|---|---|---|---|---|
| acappella traj-acc | 10% | **13%** | 12% | **13%** |
| acappella ≥80% covered | 2% | **7%** | 4% | **7%** |
| HEADLINE multiseg+loop | 24% | 25% | 22% | 23% |

Consistent gains, no regressions — but the oracle-placement gap (43–44%
ALL with GT set_start vs ~13% real) says **placement error is now the
binding constraint for end-to-end acappella**, not the ref-trace decoder.

## Self-placement (widened-window decode): mechanics work, gating doesn't

Controlled experiment (BB12, oracle ± simulated error, `run.py
--jitter-s/--pad-s`):

| config | ALL |
|---|---|
| 15 s placement error, tight window | 6% |
| 15 s placement error, ±45 s pad | **23%** |
| perfect placement, ±45 s pad | 24% (vs 43% tight) |

Padding makes the decode placement-INVARIANT (23≈24) — the Hough
diagonal's extent self-places — but costs ~19 pp on well-placed spans.
Two gating signals tried:

1. `evidence_rate` floor (fixture-calibrated at 20): did NOT transfer —
   real separated vocals sit far lower; 29/33 BB12 spans mislabeled →
   e2e 13→12. Dead (2nd evidence_rate failure after the router).
2. **`ev_out_frac`** — fraction of the padded decode's path-inlier
   EVIDENCE outside the believed span window. A ratio within one decode,
   so it survives the fixtures→real density shift (segment-TIME fraction
   does not: real padded windows tile fully with weak segments, measured
   identical placed vs jittered). Controlled real-data test (jitter as
   the manipulated variable, no accuracy tuning): placed median 0.53 vs
   15s-misplaced 0.82; gate simulation at θ=0.8: placed 43%→43% (zero
   cost), misplaced 6%→17%; robust across θ 0.7–0.85.

**SHIPPED as the production default** (`gate_out_frac=0.8`,
`gate_pad_s=45` in SegmentConfig; joint_ref_decode decodes padded first,
keeps it when ev_out_frac ≥ gate, else decodes tight). End-to-end:

| acappella traj-acc (fiber-aware) | legacy | lt ungated | **lt + gate** |
|---|---|---|---|
| BB12 (n=41) | 10% | 13% | 12% (9/33 gated) |
| BB11 (n=67) | 12% | 13% | **16%** (25/55 gated) |
| combined (n=108) | 11.2% | 13.0% | **14.5%** |

The gate fires where placement is worse (BB11) and wins there (+33% vs
legacy); BB12's −1 pp is ~0.4 spans (noise). Combined e2e acappella is
now **+30% relative over legacy**.
