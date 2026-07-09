# Handoff — alignment failure analysis + claimed_stem data-fix

Context: started as "analyze where the aligner falls short and why." Root cause of
the #1 finding was a **data bug**, not a model failure — the materialized
`claimed_stem` mis-routed ~39% of acappellas and hid the instrumental axis. Fixed
end-to-end. Full write-up: [FINDINGS.md](FINDINGS.md). This file = state + TODO.

## DONE (do not redo)
- **Failure-analysis tooling** (`eda/alignment/failure_analysis/`): `build_span_table.py`,
  `analyze.py`, `relabel_stems.py`, `FINDINGS.md` — committed `125c3c5`. One row per
  span, axis sourced from **GT** (never the timeline). `make scorecard` wraps it.
- **Bug fix** `joint_ref_decode` 10→12 job tuple (`fwd/back_slope` drift) — committed
  `817b90f`. NB the parallel agent has since refactored `joint_ref_decode`; confirm the
  fix is still present / subsumed.
- **claimed_stem parser fix** `888caca` landed on `main` as `f678f3a` (cherry-pick, same
  patch-id) → pushed → **pi deployed** to `f678f3a`.
- **Corpus-wide re-materialize DONE** (0 errors, 1.24M slots). BB12/BB11 instrumental
  slots **2 → 19**; corpus instrumental slots now in the thousands (query showed 6,966;
  a memory note says 2,552 — verify if it matters). **DB backup on pi:**
  `/mnt/storage/data/db/music_database.db.bak_premat_20260708_140345` (restore to revert).
- `score_timeline_vs_gt` now takes axis from the matched GT row — commit `794b76b`.
- **Post-fix re-runs committed** (`763fb8c`): BB12 acappella traj **31%** (3×), headline
  38%, regular 51%, instr 33%, set_start 5.0 s median, acap ref-offset 15.3 s. BB11
  identity 84%, instr **46%**, acappella 18% (coverage-limited).
- **Oracle↔e2e decomposition** ([ORACLE_E2E_GAP.md](ORACLE_E2E_GAP.md)): **91% of the
  acappella oracle→e2e gap is placement/windowing, not decode.** Placement is the top
  UNblocked lever (pooled acappella e2e 15.7 → 30.2% at oracle placement).
- **BB11 stem-coverage diagnosis + delta refresh.** The "69/147 refs have a vocals
  stem" number was the **stale local pull**, not reality — pi already had 132/147
  (filesystem, not manifest, is ground truth). Delta refresh
  (`labeling/pull_set_for_alignment.py 2nvzlh2k`) done → 80/148 local. Of the
  remaining gap, ~53 are acappella-master rows that self-resolve via the
  `_vocal_ref_path` source-audio fallback (`fca5061`); only **~15 regular/instr refs
  genuinely need separation** (queued, see in-flight).
- **BB11 acappella-flatline mechanism identified** (why 18% traj DESPITE the fca5061
  fallback): the fallback keys on the manifest row of the *predicted recording*. When
  identity resolves an acappella span to its **regular sibling** recording, the row has
  `stem=="regular"` and (pre-refresh) often no vocals stem → `_vocal_ref_path` returns
  None → HuBERT/lyrics channels silently skip the span. So acappella-span coverage is
  gated on **regular-row vocals stems** (67 → 80 after refresh; ~13 regular refs left).
  NB `0960565` (shared stem-path resolver, disk-truth fallback) may partly subsume
  this — verify against the driver's fresh numbers.

## IN-FLIGHT — DO NOT COLLIDE
`scripts/reinfer_driver.sh` (task `bk4unxwn7`) is running now: BB12 infer → BB11 infer
→ `joint_ref_decode --decoder looptrace` ×2 (writes `out/<set>_predicted_timeline_lt_v2.json`)
→ scorecard `--fibers`. It **owns `out/` and the placement code path.** Do not run
infer / joint_ref_decode / score, and do not edit `infer.py`, until it finishes and
wakes with the post-fix grid. Logs: `logs/reinfer_bb11.log`, `logs/jointdecode_*.log`.

**Queued behind it (other session, task `bmo144bfr`): BB11 ref separation.** A watcher
polls for `alignment_prototype.(infer|infer_fused|lyrics_align)` to be gone 2 min, then
runs `timeout 10800 venvs/audio/bin/python scripts/mac_analyze_loop.py --set-ids 2nvzlh2k
--separator roformer --only-reference` (the ~15 missing refs; Roformer on MPS). Log:
session scratchpad `bb11_separation.log`. Caveat: the gate does NOT match
`joint_ref_decode`, so separation may start while the driver's decode stage runs —
possible MPS contention, tolerable. **Do not launch a second mac_analyze_loop.**

## TODO (prioritized)
1. **[after driver] Reconcile + prune.** Diff the driver's fresh `_lt_v2` fibered
   scorecard against the committed `763fb8c` numbers in FINDINGS; then delete stale
   pre-fix timelines in `out/` (`*_predicted_timeline_lt.json`, `*_gtstem_lt.json`,
   pre-fix scalar `*_predicted_timeline.json`), keeping only corrected-axis (`_lt_v2`).
2. **[TOP lever, UNblocked] Acappella set_start placement.** The p90 tail off the
   full mix (weak fingerprint) is the biggest e2e lever and is NOT BB10-gated. Concrete:
   add a **confidence floor** on the HuBERT `--stem-placement` peak (fixes the known
   `<4s` over-override regression), and improve full-mix acappella placement. Touches
   `stem_placement.py` / `infer.py` — coordinate with the parallel agent (they own this
   path right now).
3. **[BB11-specific cheap win, IN PROGRESS] Vocals-stem backfill.** Corrected scope:
   pi already has 132/147; delta refresh done; only ~15 refs need separation and that
   job is **queued** (see in-flight). After it lands: verify stems on pi
   (`track_stems` rows + files), delta-refresh the pull once more, then a **second BB11
   re-run** (infer → looptrace → score, namespaced output) — the driver's current BB11
   pass runs WITHOUT these stems, so its acappella numbers remain coverage-limited.
   That second run is the first full-coverage test of whether 18% was coverage or the
   regular-sibling routing above; if still flat, chase the `_vocal_ref_path`
   sibling path in `infer.py`.
4. **[highest ceiling, BLOCKED] Instance-selection model** for the "which chorus" decode
   wall (oracle ~37%). Blocked on: (a) **John labels BB10** (`w1mgcjt` — pulled,
   unlabeled, no GT yaml) for leave-one-set-out; (b) wire **multi-set co-train**
   (`SpanTarget += set_id`; per-set store map) — "not wired" per prototype CLAUDE.md.
5. **[upstream] Recover 11 never-matched BB12 GT recordings** (id-map for online-candidate
   acappellas; w-layer stem tagging) — ingest/tokenizer, not the aligner.

## Gotchas / coordination rules
- **Axis rule:** take `claimed_stem` from the GT row, never the timeline span. The pi DB
  is fixed, but pre-fix timeline JSONs on disk are still contaminated.
- `infer` reads slots **live from the pi DB over ssh** (`_ssh_sql`), so it now gets
  corrected stems automatically — no manifest refresh needed. The `~/aligning/*/manifest.json`
  metadata is stale (audio paths in it are still valid).
- A parallel aligner agent is active in `workspaces/` — pull + scan `git log` before
  editing; land `main`-bound changes via a `git worktree` (main is checked out elsewhere).
- Never revert the parallel agent's workspace changes.
- **For "do we have X" coverage questions, ls the drive** (pi filesystem), not the
  manifest/DB — the 69/147 red herring cost half a day and nearly triggered a
  redundant mass-separation.
- **John's one non-delegable task: label BB10** (`w1mgcjt`) — it gates TODO #4, the
  highest-ceiling lever. Everything else on this list an agent can drive.
