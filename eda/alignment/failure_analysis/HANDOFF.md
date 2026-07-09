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

## RESOLVED 2026-07-09 — driver's "BB11 catastrophe" was a scorer footgun, NOT the manifest
The reinfer driver (`bk4unxwn7`, DONE 07:06) reported BB11 identity 0% / placement
~1000 s. Root cause: `score_timeline_vs_gt --gt` **silently defaulted to
`bb12_ground_truth.yaml`** — the driver passed no `--gt`, so BB11's timeline was
scored against BB12's GT (the "GT titles" in that scorecard are BB12 songs). The
manifest `slot_label=None` theory was wrong — infer's spine comes live from pi and
the raw timeline was verified sane (151/151 (slot,recording) pairs identical to the
committed run). **Fixed in `43c24a6`:** `--gt` now resolves from `--set-id` by
scanning fixtures (errors on no match), and `pull_set_for_alignment` emits
`slot_label` so the manifest-schema drift is closed too (all 5 aligning manifests
patched in place, `.pre_slotlabel_fix` backups alongside).

**Real BB11 lt_v2 numbers** (rescored with correct GT; scratchpad
`bb11_ltv2_score.txt`): identity **84%**, set_start median **7.1 s** / <15 s 64%,
headline multiseg+loop **31%**, regular 49%, instrumental **45%** (ref-offset
median 3.9 s, <2 s 50%), acappella 19% — i.e. consistent with the committed
`763fb8c` run; instrumental confirms the stem-fp channel on a second set. The
BB12 lt_v2 leg was valid all along (default GT happened to be BB12's):
identity 84%, placement median 4.5 s / <15 s 74%.

## IN-FLIGHT — DO NOT COLLIDE (updated 2026-07-09 ~11:05)
**Vast RTX 4090 (contract `44320363`, label `bb-separation-fable-20260709`,
~$0.39/hr) is running `scripts/vast_loop.py`** in tmux `analyze`: BB11 refs
first (5/6 done), then chained onto **all of BB10** (~120 tracks, full
analysis, done ~15:00). MPS on the Mac is FREE again (mac loop killed;
superseded). Do NOT rent a second box without checking `--shard`; do NOT
destroy this one — it is mine and gets torn down (plus the pi authorized_keys
line removed) when BB10 drains. An armed chain on the Mac fires
pull-refresh → infer → looptrace → fibered score into
`out/2nvzlh2k_predicted_timeline_lt_v3.json` the moment BB11's separable
refs hit zero.

**Known data bug found en route (encoding class):** `track_audio` 23070
(DJ Kool acappella) has a double-encoded DB path (`IvÃ¡n…`) vs proper UTF-8 on
disk → rsync fails. One-line path UPDATE needed on pi (agent-blocked).

## Manifest audit 2026-07-09 (all 5 ~/aligning sets)
All 488 distinct `pi_path`s exist on pi; local_path + stem paths verified on disk for
BB10/BB11/BB12/Murph. `slot_label` now emitted by the pull and patched into all
manifests. Two holes remain: **Disco Lines (1rfb0yl9)** — every local path broken
(oldest manifest schema, no `label` key); heal by re-pull when next touched. **BB10
bed rows absent from the manifest** for mashup slots (e.g. 001 bed = "Let The Drummer
Kick", a tlp-id row with no audio) — the pull drops slot rows it can't resolve to a
track_audio, so the als interpreter can't map those beds; needs a decision (emit
audio-less rows vs leave dropped). Do NOT re-pull BB10 until the Birdy `is_reference`
flip below is reverted, or the pull will fetch the instrumental as the slot's main file.

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
3. **[RESOLVED NEGATIVE 2026-07-09 11:33] Vocals-stem backfill — coverage was NOT the wall.**
   Full-coverage re-run (all separable BB11 refs stemmed via Vast 4090; chain
   pull→infer→looptrace→score, `out/2nvzlh2k_predicted_timeline_lt_v3.json`):
   acappella trajectory **19%**, identical to pre-backfill (18–19%); identity 84%,
   placement 6.8 s, headline 31%, instrumental 46% — all flat. Both the coverage
   theory AND the regular-sibling routing theory are falsified as binding
   constraints. Remaining acappella levers are therefore #2 (placement) and #4
   (instance selection, BB10-gated) — as the oracle decomposition predicted.
   ~~ORIGINAL:~~ **Vocals-stem backfill.** Corrected scope:
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
- **BB10 Birdy (slot 001w, track `2bsw9zvf`):** official Don Diablo Remix instrumental
  ingested as `track_audio` 23733 (correction 1267). Side effect: `acquire_variant`'s
  default-promote flipped `is_reference` to the instrumental — **needs manual revert
  on pi** (permission-blocked for the agent):
  `sqlite3 /mnt/storage/data/db/music_database.db "UPDATE track_audio SET is_reference=1 WHERE track_audio_id=21151; UPDATE track_audio SET is_reference=0 WHERE track_audio_id=23733;"`
  Also fix the footgun: stem-sibling adds should not steal the reference
  (`--no-promote-reference` should be the default for canonical stem adds).
- The claimed "Discord instrumental candidate" for Birdy KYHU does not exist in
  `/mnt/storage/staging/discord_stems` — the correct candidate was the local YouTube
  fetch (cand1, now canonical). Ref file content verified = Don Diablo Remix (Radio
  Edit), matching its player_id; what exactly sounded "wrong version" still needs
  John's ear (full-length remix vs radio edit is the leading theory).

## EDA-lane session note (2026-07-09 afternoon, Fable agent — failure correlates + fixes)

Full numbers in FINDINGS.md ("EDA 2026-07-09" + "Fix round" sections). For whoever
owns `infer.py`/`stem_placement.py` and the abstention contract:

- **`joint_ref_decode` now uses `stem_resolve.resolve_stem`** (disk-truth) and
  emits per-span **`ref_decode_status`** (`looptrace|looptrace-empty|skip-*|legacy`).
  BB11 A/B flat on headline (coverage 47→73 spans; oddratio +4 pp, instr
  ref-offset 7.8→3.9 s) — a robustness/diagnostics fix, not a lever. Namespaced
  output `out/2nvzlh2k_predicted_timeline_postfix_lt_diskfix.json`; canonical
  timelines untouched (your lt_v3 chain unaffected).
- **Abstention: do NOT use span `confidence`** — anti-calibrated (spearman −0.24
  vs traj; abstaining on it makes the kept set worse). **`start_source` is the
  calibrated signal**: mert-fallback-placed spans ≈ lost (traj 0.04–0.08,
  ss_med 15–40 s, n=41); lyrics-placed acappellas 0.26/2.8 s. When you wire the
  abstention field, route it off start_source (+ ref_decode_status).
- **40 GT-acappella spans are still mis-axed `regular`** (class-1 inventory gaps,
  no row-text to parse). w-layer prior quantified in looptrace/NOTES.md
  (P(acap|w-layer)=82%; 100% of acappellas are w-layers) — filed unwired per the
  freeze; `layer_role` in set_track_slots is consumed nowhere in the prototype.
- BB12 audit regenerated (`looptrace/out/audit_1fsnxchk.json`, backup
  `.pre_wlayer_regen`); BB12 GT uses plain-numeric slots so audit joins must key
  by track_id (build_span_table does now). Instance-ambiguity fracs are weak
  span-level predictors (|rho|≤.15, n=34) — temper expectations for them as
  selector features.

## Kernel-lane session note (2026-07-09, Fable agent — W0/W1 of docs/kernel_data_engine_plan.md)

For the model-lane agent; none of this touches your interfaces:

- **`--instr-stem-placement` is now default ON** (81ac097) — your armed BB11
  chain passes it explicitly, which is now a harmless no-op. Verified the pi
  re-materialize you rely on: BB12 19 instr / BB11 19 visible (GT-only
  residual ~6/set = class-1 inventory gaps).
- **`SetContext.for_set` now runs a boot preflight** (validates the manifest
  via `core.contracts.load_manifest`, checks pull completeness). Escape
  hatch: `for_set(..., preflight=False)` if you construct contexts for sets
  without a local pull.
- **New scoped ratchet `kernel_flags` = 39** (add_argument on infer /
  joint_ref_decode / drivers). Adding a kernel CLI flag now fails `make
  check` — add to the plan's burn-in table instead, or raise with
  justification.
- **A guarded `make determinism SET=1fsnxchk` is queued in background**
  (waits for GPU idle ×2). It will regenerate
  `out/1fsnxchk_predicted_timeline.json` + `1fsnxchk_classical_timeline.json`
  — with instr placement ON, so BB12 classical numbers may shift (expected:
  instr axis improves). Re-run `make race` before quoting the old board.
- Both pis deployed to `b2b2edd` (the branch merge landed on main).
- Plans landed: docs/architecture_north_star.md (OS map, P0–P6) +
  docs/kernel_data_engine_plan.md (estimation contracts W2 = the factor/
  posterior records; I'll hand off the `ProbeFactor` shape here before
  touching `harness/contract.py`).
