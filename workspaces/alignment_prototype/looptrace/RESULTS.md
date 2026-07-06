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
