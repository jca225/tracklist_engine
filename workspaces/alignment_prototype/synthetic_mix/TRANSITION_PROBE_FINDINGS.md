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

Aggregate eval (12 rides = 4 refs × 3 curvatures × directions, `eval_transitions.py`
over a `generate_transitions` dataset — the statistical version, not 2 hand stems):

```
curvature |  n | median-of-medians | p75 median | median MAE
  gentle  |  4 |      0.06s        |    0.07s   |    0.06s
  medium  |  4 |      0.04s        |    0.06s   |    0.76s
   steep  |  4 |     19.69s        |   36.65s   |   17.59s
```

On aggregate, **gentle + medium recover robustly (sub-0.1 s median-of-medians across
refs and accel/decel), steep genuinely fails (~20 s)**. This SUPERSEDES the earlier
"steep is stem-dependent" hedge — the one BB12-steep-easy case was ref-specific luck;
across refs, ±18% rides are not recoverable by the windowed chroma decoder.

## Canonical decoder (production `path_decode.decode_path`, NOT the proxy)

`transition_pathdecode.py` re-runs the same rides through the **shipping Viterbi**
(`decode_path`) — the decoder the aligner actually uses on real spans — and
reconstructs the curve from its segment list via `core.timebase.Trajectory`. This
replaces the LNDS proxy with the real thing (the "route rides through the real
path_decode" next-step). Both BB stems, 90 s ride, 12 s window, 2 s hop, lam 0.15:

```
          |  BB11 median  |  BB12 median  |  (proxy median, for contrast)
   flat   |     0.00s     |     0.00s     |     0.01s
  gentle  |     0.17s     |     0.17s     |     0.05s
  medium  |    30.47s     |     0.41s     |     0.14s   <- proxy was optimistic
   steep  |    37.32s     |    11.54s     |    17.67s
```

**The production decoder is worse than the proxy on medium+, and this is
structural, not noise.** `decode_path` searches `for s in stretches` and keeps the
single best-scoring stretch for the WHOLE span — so it is piecewise-linear in
*offset* (it can jump sections) but **constant-slope across the span**. It has no
per-window/per-segment tempo freedom. The proxy's `windowed_recover` searched a
local stretch *per window*, which is why the proxy recovered medium and the real
decoder does not.

Consequence for the scope decision:
- **Gentle rides (±4%, the realistic DJ transition band) recover on BOTH stems at
  ~0.17 s median through the ACTUAL decoder** — transitions stay IN Aug-1 scope,
  no decoder change needed for the common case.
- **Medium+ rides expose a concrete decoder lever, not a representation gap:** the
  single-global-stretch assumption is the wall. To follow a curved ride the Viterbi
  must carry stretch in its state (or re-decode each decoded segment with a local
  stretch). That is the upgrade if medium rides prove common enough to matter; until
  then the honest behavior on medium+ is **abstain** (BB11 medium 30 s median is a
  lie the decoder must not emit as a confident straight line).

## Verdict
- **Gentle rides (±4%, the realistic DJ transition range) are recoverable by the
  PRODUCTION decoder** to **~0.17 s median on both stems** — inside the ±2 s
  tolerance. Keep the transition regime IN Aug-1 scope.
- **Medium rides (±10%) are recoverable in principle** (the per-window proxy gets
  sub-0.2 s) **but the shipping decoder's constant-slope-per-span assumption fails
  them** (BB11 30 s / BB12 0.4 s, stem-dependent). Lever = stretch-in-state Viterbi
  or per-segment local stretch; NOT a representation change.
- The **constant-warp decoder structurally fails** on rides (0.7 s → 43 s as curvature
  grows) — confirms the representation must carry continuous tempo (already #12).
- **Steep rides (±18%) stay hard (~12–37 s)** — within-window tempo smear breaks the
  chroma match → **abstain** (consistent with the "open tail → abstain" rule).

## Caveats (do not over-read)
- One instrumental stem, chroma-only, **varispeed** (pitch rides with tempo); a
  keylocked ride would behave differently — a v2.
- `windowed_recover` + `monotonic_filter` is a **proxy** for the real `path_decode`
  Viterbi, not path_decode itself. Median is the fair number; MAE is inflated by the
  residual outliers LNDS doesn't catch. **The canonical section above supersedes the
  proxy's medium median** — the real decoder is single-global-stretch and does worse.
- Numbers here are a **probe result, not a headline** — deliberately NOT in
  `alignment_status.md` (that owns scorer-regenerated numbers only).

## Next
- ~~Route rendered rides through the actual `path_decode`~~ **DONE** (canonical
  section above; `transition_pathdecode.py`). It surfaced the single-global-stretch
  wall — the concrete decoder lever.
- If medium rides prove common: prototype **stretch-in-state** (Viterbi state =
  (offset, stretch)) or **per-segment local-stretch re-decode**, and re-run
  `transition_pathdecode` to confirm it lifts BB11 medium off 30 s.
- Add a **keylocked-ride** variant (pitch-preserving; the varispeed here rides pitch
  with tempo — a keylocked ride behaves differently for chroma).
- Generate ride spans inside the `generate_v2` curriculum so the learned
  (`trajectory/`) decoder can train on the regime the classical decoder can't follow.
