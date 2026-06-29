# Acappella segment-decode plan — Phase 0 result + revised direction (2026-06-29)

Follows the stem-axis handoff (`docs/agent_handoff_stem_axis_findings_20260629.md`,
commit e82aa07). That handoff diagnosed the acappella traj-acc deficit (10% on
BB12) as **warp**: "55% non-linear AND ~48% median stretch … acappella placement
needs a warp-aware / wide-stretch path decode." I built a Phase 0 probe to split
that one word "warp" into the two distinct decode failures it bundles, because they
need very different fixes. **The result kills half the premise.**

## Phase 0 probe — what it measures

`workspaces/alignment_prototype/acappella_warp_decode_probe.py`. The decode's
representable class (`path_decode.decode_path`) is: **piecewise-linear with ONE
global slope `s` and FREE per-segment offset** — staying on a diagonal is free, an
offset jump (loop / section-jump) costs `lam` but is allowed. So a span is decode-
representable iff its GT segments share one slope (offsets may jump) AND that slope
is in the grid. The probe loads BB12 GT clip rows and buckets each span:

- **A — already representable** (one slope, in grid) → failure is *placement/feature*, not the decode.
- **B — Phase-1 grid fix** (one slope, OUT of the current grid) → cheap.
- **C — Phase-2 varying-slope DP** (slope genuinely varies within the span) → expensive.

The first probe run miscounted because the `const_ok` test forbade offset jumps,
dumping every loop/section-jump into C. Fixed to credit free offset-jumps (the
decode's actual capability) and to segregate degenerate >400s "spans" (whole-mix
recurring vocal motifs, not single clips).

## Result (BB12 `1fsnxchk`)

| stem | n | A (placement) | B (Phase-1 grid) | C (Phase-2 warp) | median \|s_rep−1\| |
|---|---|---|---|---|---|
| acappella | 85 | **72%** | **28%** | **0%** | 0.480 |
| regular | 70 | 79% | 21% | 0% | 0.002 |
| instrumental | 7 | 71% | 14% | 14% (n=1) | 0.027 |

And the clincher: **every clip in the set has exactly 2 warp markers** (start+end)
— `markers/clip median=2, max=2, clips_with>2_markers=0%` across all 173
acappella/regular/instrumental clips. There is **no intra-clip phrase-by-phrase
warp anywhere in BB12**. Each clip is a single constant-stretch line; loops and
section-jumps are separate clips (= free offset jumps in the decode).

## What this overturns

1. **Phase 2 (varying-slope / warp-aware DP rewrite) is NOT justified.** 0% of
   acappella spans need it; the GT contains no varying-slope warp. The handoff's
   "warped phrase-by-phrase" reading is false — the "55% non-linear" stat counted
   loops/multiseg as non-linear, but those are *free* in the Viterbi. **Do not
   build the 2D (offset×stretch) DP.** It would solve a problem the data doesn't have.

2. **The decode-fixable part of the acappella deficit is the 28% B spans only** —
   constant half-time-ish vocal stretches (`s_rep` spread across [0.34, 1.79]) that
   miss the narrow anchored grid. The grid is dense only in a ±4% window (2% steps)
   around octave multiples {0.5, 1.0, 2.0} of a beat-grid-anchored center
   (`path_decode._stretch_band`); the in-between stretches fall through the gaps.

3. **The dominant 72% (A) is placement/feature, not representation.** Those spans
   are already expressible by the decode and still score ~10% traj-acc → the error
   is wrong region / wrong offset / repeat-ambiguity. This is the known acappella
   wall (set_start median 42.5s; ref_start repeat-ambiguous) — the lane the
   handoff and prior probes already showed has **no cheap DSP lever left** (HuBERT
   ref_start doesn't generalize; continuity-stack is a no-op).

## Revised plan

### Phase 1 — wide acappella stretch grid (cheap; DO, but bounded)
In `path_decode._stretch_band`, when the span is acappella (and/or the ref beat
grid is low-confidence), stop trusting the ±2% anchored window and search a
broad **log-spaced grid (~0.4–2.0 at ~2% steps)**. Stem-routed so `regular`/
`instrumental` are untouched (they already work). Converts the 28% B from
"guaranteed-wrong slope" to "right slope IF placed right."

**Honest ceiling:** Phase 1's realized gain is bounded by placement. A B span only
improves if its *offset* is also right, and acappella placement is bad (72% A are
already wrong despite being representable). So expect a small traj-acc bump, not a
fix. Land it anyway — it's a few lines, strictly correct, and removes a real
representation gap. Measure the A/B/C-restricted traj-acc before/after to attribute
the gain honestly.

### Phase 2 — CUT. Replaced by placement.
No varying-slope DP. The acappella lever is **placement**, which has no remaining
cheap DSP fix → it belongs to the learned model (`project_alignment_bootstrap_flywheel`)
and to acquiring more acappella GT to train/validate against. Track that under the
existing "open lane = acappella ref_start / set_start" work, not here.

### Phase 3 — eval + regression gate (unchanged)
BB12 acappella traj-acc + set_start after Phase 1, with a no-regression check on
`regular`/`instrumental` (feature/stem-routed, should be clean). Use
`score_timeline_vs_gt --fibers`; also report A/B/C-restricted traj-acc so the
Phase-1 grid gain is separable from placement noise.

## Caveat on generality (one set)
All numbers are BB12. BB11 acappellas were mostly straight (16% non-linear,
med|ratio−1|=0.083) — even *less* warp — so Phase 2 is unjustified there too. If a
future set shows clips with >2 warp markers (real intra-clip warp), re-open the
Phase-2 question; the probe (`--max-span-s`, warp-marker count) is the gate.

## Repro
```
venvs/audio/bin/python -m workspaces.alignment_prototype.acappella_warp_decode_probe \
  --als "$HOME/aligning/1fsnxchk__*/BB12 align.als" \
  --set-dir "$HOME/aligning/1fsnxchk__*" --stems acappella,regular,instrumental --show 8
```
