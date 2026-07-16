# Agent handoff — wire the placement pipeline into the aligner (2026-06-28)

## Goal of the next session
Turn the **validated, committed placement pipeline** into a single end-to-end
aligner output: **wire `decode_placements` into `infer.py`** (set-level placement
stage) and **add per-stem placement** for acappella/instrumental. Output: a full
predicted timeline (per span: recording_id, set_start, set_end, ref_start) scored
vs GT, round-trippable to `.als`.

## The one thing to understand first — the placement reframe (why this works)
The ~30s set_start "wall" was a **decomposition error**, not a real limit. The
landmark fingerprint localizes the mix↔ref **alignment diagonal** `d` to **0.2s
median / 76%** (BB12 regular). set_start looked stuck only because we measured it
as `d`, but **DJs start tracks mid-song** (GT ref_start median ~56s), so
`set_start = ref_start + d`. The fingerprint's vote-density extent along `d` gives
the played span directly. Full memory: `project_placement_wall_was_decomposition_error`.
MERT is identity-only (cannot localize, ~900s argmax) — do NOT use it for placement.

## What's already built + committed (reuse, don't rebuild)
All in `workspaces/alignment_prototype/` unless noted. 17 commits this session.

**Placement pipeline (`mix_fp_hits.py`):**
- `span_from_offset_votes(mix_hashes, ref_fp) -> (set_start_s, set_end_s, votes, offset_s)`
  — single-ref vote-extent. set_start median 5.7s (BB12 regular).
- `offset_candidates(mix_hashes, ref_fp, topk=6) -> [(set_start,set_end,votes,offset_s)]`
  — top-K alignment diagonals per span.
- `decode_placements(mix_hashes, ref_fps, *, mix_dur_s) -> [(set_start,set_end)|None]`
  — **THE function to wire in.** `ref_fps` in TRACKLIST (slot) order; monotonic
  decode rejects out-of-order wrong-diagonal/repeat picks. BB12 regular: set_start
  median **4.1s, <15s 73%**. `min_step=0` (mashup layers start near-simultaneously).
- `load_mix_mono(path)`, helpers `_vote_pairs`/`_cluster_at`.
- Convention: `off = ref_frame - mix_frame`; build mix hashes ONCE per set via
  `landmark_fp.hashes(*constellation(mix))`, reuse across refs.

**Runnable eval:** `eval_placement.py` (`python -m workspaces.alignment_prototype.eval_placement`)
— runs decode_placements vs GT. Reproduce: set_start median 4.1s, <15s 73%.

**Identity / axes / fibers (for fusion + per-stem):**
- `harness/axes.py` `route_for_stem(stem)` → (mix_file, ref_stem, invariant_feature,
  placement_probes in priority order). vocals→hubert, instrumental→chroma+fp.
- `harness/hubert_probe.py`, `harness/continuity_probe.py`, `harness/merge.py`
  (`source_priority` = axis-priority arbitration — use it, raw cross-axis conf is unsound).
- `ref_fibers.compute_fibers_soft` (μ + per-fiber conf), `fiber_ambiguity`.
- `section_hsmm/similarity_probe._hubert(y, layer=9)` — HuBERT features on SR/HOP grid.
- `acappella_ref_offset_eval.py` — HuBERT vocal ref-offset (2.1s median); uses the
  PLACED winner files directly (= the implicit winners).

**Winners (`labeling/extract_winners.py`):** `extract_winners(als, set_dir) -> [Winner]`
— per (slot, stem) the placed candidate file = the winner reference. BB11: 60.

## Task 1 — wire `decode_placements` into `infer.py`
`infer.py` today: trains MERT head on BB12 → `MertLearnedAligner.predict_sequence`
(monotonic DP over MERT curves + cue-anchor prior, the ~30s placement) → optional
`fine_refine` DTW + `fp_placement_refine` → writes `out/<set>_predicted_timeline.json`.

Replace/augment the **placement** (keep identity = MERT/HuBERT candidate selection):
1. After identity picks the candidate recording per slot, gather `ref_fps` =
   `[fp_index.load(FpKey(rid, "regular")) for rid in chosen, in slot order]`.
2. Hash the target mix once: `hashes(*constellation(load_mix_mono(set_dir/"mix.m4a")))`.
3. `placements = decode_placements(mix_hashes, ref_fps, mix_dur_s=dur)`.
4. **ref_start**: decode_placements returns (set_start,set_end) only. For the
   timeline you also need ref_start (which part of the song). The diagonal offset
   is available — `ref_start = set_start + offset_s` (sign verified: `off=ref-mix`,
   `set_start-ref_start = -off_s`). EXTEND `decode_placements`/`offset_candidates`
   to also return the picked `offset_s`, then `ref_start = set_start + offset_s`.
5. Emit the timeline JSON in the existing schema (set_start/end, ref_start,
   recording_id, confidence=votes-normalized). Keep `predict_sequence` available
   behind a flag for comparison.
6. Score with `score_timeline_vs_gt.py`.

## Task 2 — per-stem placement (acappella / instrumental)
Only `stem='regular'` fingerprints are backfilled, and **fingerprint is weak on
vocals** — use the AXIS-INVARIANT feature per `axes.py`:
- **acappella**: HuBERT vote-extent — slide the winner acappella (HuBERT features,
  `_hubert`) over `mix_vocals.flac` HuBERT to find WHERE it plays (set_start), the
  HuBERT analog of `span_from_offset_votes`. The winner file comes from
  `extract_winners` (the placed clip). `acappella_ref_offset_eval` already does the
  ref-offset direction (which part); set_start is the complement (where in mix).
- **instrumental**: `mix_instrumental.flac` vs the winner instrumental stem; chroma
  + fingerprint of the stem (may need an instrumental-stem fp backfill, or chroma).
- Fuse per-stem placements with the full-mix `decode_placements` at decode (the
  acappella's set_start should be consistent with / refine the full-mix span).

## Gotchas / do-not-retry (measured this session)
- **Fingerprint backfill is done corpus-wide for `stem='regular'` only** (414/424,
  `scripts/backfill_track_fingerprints.py`, now resilient to pi SSH hiccups). Local
  cache: `.cache/fp_index/`. Acappella/instrumental stem fps are NOT backfilled.
- **The mean error is noise** — dominated by 1–2 extreme wrong-diagonal spans. Use
  **median / <15s / outlier-count**, never mean, as the metric.
- **Rejected outlier-fixes (don't retry):** per-span cluster-strength selection
  (mean 44→146s), isotonic monotonization (median 5.7→13.2s), boundary-snap
  (precision only, no outlier fix). The remaining ~10/37 outliers are weak-fp/repeat
  spans whose true diagonal is absent from top-K → per-stem + fibers, NOT post-hoc.
- **Key changes** (31% of acappellas re-pitched): chroma breaks, transposition
  search adds spurious peaks, DJ transpose ≠ fix (ref is an arbitrary-key rip).
  HuBERT is key-invariant — that's why it's the vocal placer. (`project_key_change_breaks_chroma`)
- **BB11 is mid-audition** (winners = placed clips via `extract_winners`); its
  `.als` uses the master-tempo/unwarped-mix convention (export fixed:
  `TempoArrangementMapper`, `project_bb11_master_tempo_export`). recording_id for
  BB11 winners needs corpus ingest (`scripts/ingest_stem_url.py`) — not required for
  local placement eval.
- **MERT is identity-only.** Synthetic pretrain is a closed negative (flat v1+v2).

## Verification
- `python -m workspaces.alignment_prototype.eval_placement` → set_start median 4.1s, <15s 73% (regression).
- After infer wiring: `infer` on BB12 (in-domain) then `score_timeline_vs_gt --set-id 1fsnxchk`
  — set_start should drop from the old ~30s to single-digit median.
- `path_decode.py --eval --fibers` → fiber-aware traj-acc (53→59% baseline).
- Generalization: pull + label a non-BigBootie set (Disco Lines / Muroh) and run
  eval_placement — the untested claim is generalization beyond mashup style.

## Pointers
- Plan: `.claude/plans/and-then-can-we-cuddly-sparrow.md` (unified aligner; Phase D reframed).
- Memories: `project_placement_wall_was_decomposition_error`, `project_hubert_vocal_ref_offset`,
  `project_key_change_breaks_chroma`, `project_fusion_needs_axis_prior`,
  `project_boundary_novelty_placement_prior`, `project_bb11_master_tempo_export`.
- Module guide: `workspaces/alignment_prototype/CLAUDE.md` (updated with the reframe).
- Scratch experiments (not committed): `scratchpad/d1_*.py`, `d1d2_fusion_test.py`,
  `boundary_novelty_test.py`, `key_test.py` — the measurements behind the above.

## Project-level context (what's left overall)
Three buckets: (1) **integration** [this handoff] — no labeling needed; (2)
**generalization** — label a few diverse sets (Disco Lines/Muroh) for eval, NOT a
big training corpus (the aligner is mostly unsupervised); (3) scope-finish —
tolerances (`alignment_objective` UNSET) + open-set. Identity ✅, which-part ✅
sub-second, placement ✅ ~4s median; remaining is integration + generalization.
