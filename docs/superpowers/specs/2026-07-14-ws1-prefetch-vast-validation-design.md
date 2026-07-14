# Design: streaming_mir WS1 prefetch + WS2 GPU validation

**Date:** 2026-07-14
**Status:** Approved (Approach A selected by John)
**Companion docs:** [streaming_mir RESEARCH_BRIEF](../../../workspaces/streaming_mir/RESEARCH_BRIEF.md) ·
[three-regime platform design](2026-07-14-three-regime-lab-platform-design.md)

---

## 1. Motivation & scope

streaming_mir's WS1 (corpus throughput) is the remaining "fast downloads" arm:
`vast_loop.py` already hides the *output* tail (stems rsync + DB push) behind a
single-slot background thread (~30% win), but the *input* side — `next_task`
SSH pick + `rsync_in` audio pull — is serial with GPU work. The corpus backlog
is real: 18,825 of 19,595 `track_audio` rows have no `track_analysis`
(pi-storage, 2026-07-14), so per-track wall-clock directly prices the Aug-1-era
corpus scale-up.

**Scope decisions (John, 2026-07-14):**

- Build **input prefetch now**; **measure** (not build) model batching — the
  same rental instruments where GPU time actually goes so batching gets a
  data-driven go/no-go.
- **Bundle WS2's** pending real-set before/after (the `--overlap-sec` seam fix,
  commit `cad14a3`, never validated on a real GPU/set) into the same rental.
- **`vast_loop.py` only** this pass; `mac_analyze_loop.py` mirror is a later
  trivial follow-up.

**Restructure note:** all changes live in `scripts/` (stays root under the
post-Aug-1 three-regime restructure) and `analysis/` (log-only lines). Nothing
new is created that the W1 `labs/` move would have to relocate.

## 2. Prefetch design (Approach A — one-slot thread)

Mirror of the existing persist thread, pointed at the input side.

Per main-loop iteration:

1. Join the prefetch thread → receive `(tid, local_audio_path, asset)`,
   already rsync'd and `fetch_asset`-hydrated.
2. Start the **next** prefetch thread immediately.
3. `analyze_track` on the current item (GPU, the long pole).
4. Hand the output tail to the persist thread (unchanged).

First iteration: one synchronous pick+pull (nothing to overlap yet).
`--no-prefetch` flag preserves the legacy serial path for the A/B.

**In-flight exclusion (the correctness core).** `next_task` picks from
canonical, where an in-progress track's rows haven't landed. The prefetch pick
therefore passes `skip_tids = failed_tids ∪ inflight`, with
`inflight = {tid currently analyzing, tid in the persist thread}`. Semantics
vs today:

- N succeeds → excluded naturally once its rows land. Unchanged.
- N fails analysis → enters `failed_tids`. Unchanged.
- N's *push* fails → re-picked one iteration later than today (once it leaves
  `inflight`); same eventual retry, slightly deferred. Documented in a comment.

**Error handling.** A prefetch-side rsync/SSH failure is surfaced to the main
loop and handled like today's per-track subprocess failure: log,
`failed_tids.add(tid)`, continue with a fresh synchronous pick. The startup
orphan-wipe of `LOCAL_AUDIO` already covers crash leftovers (a prefetched file
from a killed run).

**Testability.** The exclusion-set / result-holder logic lives in small pure
helpers, unit-tested without threads; `make check` covers them. The threading
itself is validated by the Vast run.

## 3. Instrumentation (Phase-0 baseline + batching decision data)

One structured log line per track: `pull_s`, `analyze_s` with per-analyzer
stage split (separator / beats / cue-detr / MERT / features — added as
log-only timing in `analysis/pipeline.py` if not already present), and bg
`push_s`. Aggregated over the A/B run this yields:

- the honest wall-clock baseline the research brief requires before any
  further optimization, and
- the stage-time table that decides whether model batching justifies touching
  `analysis/pipeline.py` internals. That table is the batching go/no-go
  criterion — not intuition.

## 4. Vast test protocol (one rental, two deliverables)

Rent one CUDA box per `vast_bootstrap.sh` + the instance-picking recipe.

**WS1 A/B:** same scope + shard; ~30 tracks `--no-prefetch`, then ~30 with
prefetch, drawn from the unanalyzed backlog (real corpus progress, not
throwaway compute; exact `--set-ids` slice chosen at run time from what has
audio on pi-storage). Report mean per-track wall, total wall, per-stage table.

**WS2 before/after:** `render_set_stems.py` on one real set mix (BB11
`2nvzlh2k` or BB12 `1fsnxchk`) at `--overlap-sec 0` vs `10`. A 60–90 min set
has no full-file offline reference (VRAM-impossible), so the metric is a
**shifted-grid pseudo-reference**: a third render with the chunk grid offset by
half a chunk, whose interiors span the other renders' join points;
boundary-local SDR of each render vs the pseudo-reference quantifies the seam.
Plus an ear check at 2–3 join timestamps. Results land in
`workspaces/streaming_mir/RESEARCH_BRIEF.md`; WS2 marked closed.

**Cost:** ~2–4 GPU-hours ≈ $1.50–4 on a 4090-class spot.

## 5. Success criteria

- **WS1:** prefetch-on wall-clock reduction ≈ mean(`pull_s`), with outputs
  identical by construction (same analyze/persist code path). Honest
  expectation ~5% — most of the brief's ≥20% WS1 target was already banked by
  the persist-hiding thread.
- **Instrumentation:** per-stage time table over ≥30 tracks; explicit
  batching go/no-go recorded in the brief.
- **WS2:** overlap-10 boundary SDR within 0.5 dB of interior SDR against the
  pseudo-reference, and no audible seam → workstream closed.

## 6. Non-goals

- Model batching implementation (measure only).
- `mac_analyze_loop.py` mirror.
- Any prefetch queue depth > 1.
- Causal/streaming estimates as canonical output (banned by the brief's
  design law).
- New modules/directories that the post-Aug-1 `labs/` restructure would have
  to move.
