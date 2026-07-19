# bb11_34 dig — wrong high-vote diagonal

Date: 2026-07-18  
Branch: `fp-hit-decoder-clean`  
Case: BB11 slot `034`, recording `g99m0t5`, instrumental, **multiseg n=4**

## GT structure

Audible onset **2990.55s**. Four `ref_segments` (mashup/cut-up), including a
later piece starting at mix **3023.05s**. Placement scoring keys off audible
onset, not the middle segment.

## What vote-argmax did

Instrumental landmark fp top candidates (cluster votes):

| rank | set_start | duration | votes | density | err to GT onset |
|---|---|---|---|---|---|
| 1 (argmax) | 3023.8 | 41.0s | 403 | 9.8/s | **+33.2s** |
| 2 | 2992.5 | 9.0s | 163 | **18.1/s** | **+1.9s** |
| 3 | 3018.1 | 48.6s | 338 | 6.9/s | +27.6s |

Classical / `instr_fp` / `_lt` all locked onto rank 1. Rank 1 is a **long false
diagonal** (ref≈175s), not GT segment 3 — it just sits near 3023 by coincidence.
Rank 2 matches the audible-onset segment (ref≈151 ≈ GT seg1 ~148).

## Signal that separates them

**Vote density** (`votes / cluster_duration`) among candidates with
`votes ≥ 0.3 × max_votes`. Rank 2 wins; weak dense strays below the floor lose.

Validated on the other decoder_wall cases with available fps: bb11_39 / bb12_42w5
stay at their good argmax; bb12_3w2 unchanged (still soft-miss).

## Fix

`mix_fp_hits.pick_dense_competitive` + densest-competitive-first ordering in
`offset_candidates`; `span_from_offset_votes` and `decode_placements` curve
weights use density. See `tests/test_mix_fp_hits.py`.
