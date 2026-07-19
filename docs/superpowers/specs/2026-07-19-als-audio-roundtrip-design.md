# Design: Ableton denotational audio round-trip

**Date:** 2026-07-19  
**Status:** implemented (phase 1 offline); phase 2 Live goldens scaffolded

## Problem

`labeling/als/roundtrip.py` proves XML/codec fidelity (`parse ∘ print = id`).
That law stayed green while GT shaping bugs (distant-clip sliver merges) corrupted
trajectories. Structural fidelity is necessary but **not** the release ruler.

## Law

Hand `.als` arrangement (pre-merge clips) and exported GT must **sum to the same
labeled-layer audio** under an offline warp/gain summer:

```text
render(arrangement_clips(als)) ≈ render(gt_yaml)
```

Caught by synthetic test: two reprises of the same file separated by 12s must
match as two clips; a single 0–16s merged clip fails correlation.

## Phase 1 (gate-hard)

- `labeling/als/render_offline.py` — `RenderClip` + `sum_render_clips`
- `labeling/audio_roundtrip.py` — compare + CLI
- `collect_kept_clip_rows(..., arrangement_denotation=True)` — skip merge/loops
- `make gt-gate` runs round-trip before stamping; write-back requires
  `audio_roundtrip.ok` on the stamp

Does **not** replace `als_audit` (mix-vs-label). Round-trip is session
self-consistency.

## Phase 2 (Mac Live goldens)

See [labeling/live_export_roundtrip.md](../../../labeling/live_export_roundtrip.md):
operator Export Audio of hand `.als` vs re-seeded session; same
`assert_audio_equivalent` on WAVs. Not CI-blocking until Live automation exists.

## Non-goals

Bit-identical PCM, Live plugin DSP, treating codec round-trip as a release gate.
