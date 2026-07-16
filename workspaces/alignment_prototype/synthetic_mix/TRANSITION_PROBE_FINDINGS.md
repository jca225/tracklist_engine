# Transition-recovery probe — findings (§12 Aug-1 scope)

Answers state-of-record decision #12: *representation of continuous tempo is a
permanent commitment; recovery-by-Aug-1 is empirical — the synthetic-transition
probe decides the branch.* This is that probe.

## Method
`transition.py` renders a real ref stem through a continuous tempo **ride** (linear
instantaneous ratio ramp `r0→r1`; ref-time is the integral, so mix↔ref is
quadratic), then `transition_probe.py` recovers `ref_time(mix_time)` by a sliding
chroma matched-filter (`refine_ref_offsets.detect_offset` per window) + a
longest-non-decreasing-subsequence path constraint (a cheap Viterbi proxy), and
scores vs the known curve. Contrast: a single global chroma match = the
constant-warp straight line the current one-`tempo_ratio`-per-span decoder assumes.

Run (BB11 `mix_instrumental.flac`, 90 s ride, 12 s window, 15 windows):

```
curvature |  path MAE | path median | constant MAE |  kept  | gap
    flat |    0.01s |      0.01s |       0.00s | 15/15  | -0.0s
  gentle |    0.07s |      0.05s |       0.73s | 14/15  | +0.7s   (±4% tempo)
  medium |   10.08s |      0.14s |      43.02s |  9/15  | +32.9s  (±10%)
   steep |   18.39s |     17.67s |      36.38s |  8/15  | +18.0s  (±18%)
```

Cross-stem robustness (BB12 `1fsnxchk` mix_instrumental, same params):

```
    flat |    0.01s |      0.01s |       0.00s | 15/15
  gentle |    0.06s |      0.07s |       0.63s | 15/15
  medium |    3.70s |      0.06s |       0.57s | 12/15
   steep |    6.61s |      0.07s |       0.99s | 10/15
```

On BB12 the **median recovers to sub-0.1 s at every curvature incl. steep** — the
gentle/medium conclusion holds on both stems, and steep is **stem-dependent** (BB11
steep ~18 s, BB12 steep 0.07 s median). The MAE-vs-median gap (outliers the LNDS
proxy doesn't catch) is consistent across stems; the real `path_decode` Viterbi
would tighten the MAE toward the median.

## Verdict
- **Gentle–medium rides (±4–10%, the realistic DJ transition range) are recoverable**
  to **sub-0.2 s median** — inside the ±2 s alignment tolerance. Keep the transition
  regime IN Aug-1 scope with a **windowed + path-constrained** decoder.
- The **constant-warp decoder structurally fails** on rides (0.7 s → 43 s as curvature
  grows) — confirms the representation must carry continuous tempo (already #12).
- **Steep rides (±18%) stay hard (~18 s)** — within-window tempo smear breaks the
  chroma match → **abstain** there (consistent with the "open tail → abstain" rule).

## Caveats (do not over-read)
- One instrumental stem, chroma-only, **varispeed** (pitch rides with tempo); a
  keylocked ride would behave differently — a v2.
- `windowed_recover` + `monotonic_filter` is a **proxy** for the real `path_decode`
  Viterbi, not path_decode itself. Median is the fair number; MAE is inflated by the
  residual outliers LNDS doesn't catch.
- Numbers here are a **probe result, not a headline** — deliberately NOT in
  `alignment_status.md` (that owns scorer-regenerated numbers only).

## Next
Route real rendered rides through the actual `path_decode`/`joint_ref_decode`
(instead of the proxy) for a canonical number; add a keylocked-ride variant;
generate ride spans inside the `generate_v2` curriculum so the learned decoder can
train on them.
