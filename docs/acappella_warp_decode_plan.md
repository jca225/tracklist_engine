# Acappella segment-decode — Phase 0 findings + result (2026-06-29)

Follows the stem-axis handoff (`docs/agent_handoff_stem_axis_findings_20260629.md`,
commit e82aa07), which diagnosed the acappella traj-acc deficit (BB12) as **warp**
and recommended a "warp-aware / wide-stretch path decode." I probed that claim and
then tested the cheap fix. **Net: there is no cheap decode fix for acappella —
widening the stretch grid measurably REGRESSES it. The lever is placement + the
learned model.**

> **Correction (important):** an earlier version of this doc concluded "0%
> varying-slope, every clip has 2 warp markers." That was WRONG — it read
> `BB12 align.als`, which is the **seeded/re-exported** session (800 clips, each a
> 2-marker stretch — the seeder splits every warped clip into one 2-marker clip per
> segment). The canonical hand-label is **`big bootie 12 labeling_fast.als`**
> (307 clips, 33% with >2 warp markers, max 121). Nonlinear warps DO exist inside
> acappella clips. All numbers below are from the canonical file / `bb12_ground_truth.yaml`.

## What the warp actually is (canonical file)

Within-clip warp curve deviation from a straight line, by stem:

| stem | clips | multimarker (>2) | within-clip dev: median / p90 / max | clips >2s |
|---|---|---|---|---|
| acappella | 100 | **39%** | 0.00 / 0.05 / **0.12 s** | 0 |
| instrumental | 25 | 12% | 0.00 / 0.00 / 0.03 s | 0 |
| regular | 41 | 15% | 0.00 / 0.00 / 2.47 s | 1 |

So the warp markers are real and plentiful, but the warp is **near-linear**: the
curve deviates ≤0.12 s from a straight line on acappella (vs the 2 s scoring
tolerance). The annotator places many markers to lock a constant stretch bar-by-bar
against the grid, not to bend the timeline. Section skips are *separate clips* (→
separate segments = free offset jumps in the Viterbi), not internal warp. The GT
export (`export_als_to_gt._clip_row`) reduces each clip to its two endpoints through
the warp map anyway, so the scored GT is one straight segment per clip.

## Phase 0 probe — decode-representability decomposition

`workspaces/alignment_prototype/acappella_warp_decode_probe.py`. The decode's
representable class (`path_decode.decode_path`) is **one global slope `s` + free
per-segment offset** (loops/section-jumps cost `lam`, allowed). A span is
representable iff its GT segments share one slope AND that slope is in the grid.
On the canonical GT:

| bucket | acappella | meaning |
|---|---|---|
| A already-representable (1 slope, in grid) | **44%** | failure is placement/feature |
| B 1 slope, OUT of grid | **46%** | candidate for a wider grid |
| C slope VARIES across segments | **9%** | cross-segment multi-tempo (not intra-clip warp) |

median \|s_rep−1\| = 0.075 (stretches mild, near 1:1; tail to [0.13, 2.96]); the
current narrow grid reaches only **47%** of acappella spans.

## Phase 1 (wide stretch grid) — TESTED and REJECTED

The probe says 46% of spans are out-of-grid, so a wider grid looks obvious. It
isn't. Oracle-placement eval (`path_decode --eval --stems acappella --feature
hubert --fibers`, n=21 measurable spans), acappella traj-acc / fiber-aware /
linear-subset:

| grid | traj-acc | fiber-aware | linear |
|---|---|---|---|
| **narrow anchored (±2% × octaves {0.5,1,2}) — baseline** | **44%** | **47%** | **62%** |
| moderate (octaves × fine ±8%) | 34% | 42% | 50% |
| wide dense (0.45–2.2 log-spaced ~2.5%) | 33% | 40% | 36% |

**Monotonic: wider grid → worse.** Widening even destroys the *easy* linear spans
(62→36). The narrow anchored grid is **load-bearing regularization**: extra stretch
candidates let the Viterbi find spurious high-reward paths at wrong stretches on the
noisy HuBERT reward surface, costing more than the coverage gains. The "46%
out-of-grid" is real but *not recoverable by widening* — the matched-filter reward
can't localize the true stretch among many candidates. Phase 1 is dead.

## Phase 2 (varying-slope DP) — NOT justified

The intra-clip warp is ≤0.12 s (near-linear), so continuous within-segment warp
buys nothing. The only genuine varying-slope population is the 9% C — *cross-segment*
multi-tempo slots (same acappella played at two tempos on one slot). The targeted
fix would be per-segment stretch selection in the Viterbi, not a continuous-warp DP.
But since widening the global stretch set already regresses (spurious matches), per-
segment stretch selection would very likely overfit the same way. Low expected
value on a 9% population; do not build it without a reason to expect different.

## Conclusion

No cheap decode-grid lever for acappella. The narrow grid is already near-optimal.
The acappella deficit is **44% placement/feature** (the known wall — set_start
median 42.5 s, ref_start repeat-ambiguity; HuBERT ref_start doesn't generalize,
continuity-stack is a no-op) plus **46% out-of-grid stretch that resists widening**.
Both point at the same place: the learned model
(`project_alignment_bootstrap_flywheel`), which can disambiguate stretch+placement
jointly from richer context than a per-window matched-filter reward, and more
acappella GT to train/validate against. This converges with the handoff's plateau
conclusion via a different route.

## Caveat
All numbers BB12 (`1fsnxchk`), n=21 in the oracle eval (only spans with cached
HuBERT/audio survive). BB11 acappellas are even straighter (16% non-linear). The
grid-width regression is large and monotonic, so the conclusion is robust to the
small n, but re-confirm on the next labeled set.

## Repro
```
# decomposition (use the CANONICAL hand-label, not BB12 align.als):
venvs/audio/bin/python -m workspaces.alignment_prototype.acappella_warp_decode_probe \
  --als "$HOME/aligning/_backups/.../big bootie 12 labeling_fast.als" \
  --set-dir "$HOME/aligning/1fsnxchk__*" --stems acappella,regular,instrumental
# grid A/B (revert path_decode between runs; baseline narrow grid is the winner):
venvs/audio/bin/python -m workspaces.alignment_prototype.path_decode \
  --eval --stems acappella --feature hubert --fibers
```
