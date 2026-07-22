# Operation Crush — exit program (overarching, sequential)

**Date:** 2026-07-21
**Owner-in-the-loop:** ops steps mutate canonical pi state — pause at every **GATE**.
**Goal:** land the content-addressed GT identity fix, clean the upstream `track_audio`
mis-registrations it surfaced, then re-pull → re-export → re-measure the de-poisoned
GT — the actual **Crush exit**.

Context: content binding is implemented + verified (PR #69). Three independent
methods agree it kills the `slot_id_map` poison (028→Beatles, 031→CCR, 144→abstain;
0 cross-song stamps; audit-gt 281/290 audio-OK). Verification also surfaced ONE
upstream bug it does not fix but *inherits*: `track_audio` mis-registration
(`track_audio_id 20911`: a "Come On Over Baby" cover acappella filed under recording
`42wv4vp` "Good Time"). This program closes that class for the GT sets, then exits.

Do phases **in order**. Each ends with a checkable result; consequential mutations
are GATEs.

---

## Phase 0 — Land the foundation

0.1 **Pre-merge gate:** on the branch tip, run the FULL `make check` (not just the
    fast subset) — it was last green at `331aeba`, before the two final fixes
    (`d03e63d` collision-abstain + mdat guard). Confirm green.
0.2 Confirm PR #69 CI green + mergeable (already: guardrails pass, MERGEABLE/CLEAN).
0.3 **GATE (outward):** merge PR #69 into `main`.
0.4 `make deploy` — pi pulls `main`, so `labeling.build_content_catalog` + the new
    exporter exist on pi. (Deploy caveat: no `audio_pipeline` unit rename needed here.)
    Verify: `ssh pi-storage 'cd ~/tracklist_engine && python3 -m labeling.build_content_catalog 1fsnxchk | head -c 60'` returns JSON.

## Phase 1 — Upstream mis-link scan (read-only)

1.1 Build a scan (`scripts/audit_stem_recording_links.py`, or a query) over `track_audio`
    for the mis-link smell: rows whose **source song** (from `path`/`player_id`, e.g.
    `ingest_stem_cand*__<song>`) disagrees with their **recording title**. Focus first
    on `stem in (acappella,instrumental)` + `platform='manual'` candidate rows (the
    class 20911 belongs to). Rank by title-token disjointness.
1.2 Report scope: total suspects, how many touch BB11/BB12 (`recording_id` in those
    sets' `set_track_slots`), and whether the `w`-layer-inherits-base-recording pattern
    dominates. **Log what was NOT auto-fixed** (no silent caps).
1.3 Result: a triaged suspect list. `20911` must appear.

## Phase 2 — Fix the GT-affecting mis-registrations (canonical mutation)

2.1 **Snapshot first:** back up `track_audio` rows to be touched (dump the suspect rows
    to a timestamped file) before any change.
2.2 For each GT-affecting suspect (BB11/BB12), decide per the stem-pipeline bands
    (accept/review/**abstain**): re-link `recording_id` to the correct recording if one
    exists; if none exists (e.g. no Christina Aguilera "Come On Over" recording in DB),
    **null the wrong `recording_id`** (abstain) rather than leave it cross-linked — a
    null is honest, a wrong link is the bug. Use the `replace-track-audio` /
    `track_audio_correction` path; log each correction by axis.
2.3 **GATE (canonical `track_audio` write):** apply corrections on pi. Defer non-GT
    corpus suspects to a tracked backlog (do NOT block exit on full-corpus cleanup).
2.4 Verify: re-running the Phase-1 scan shows the GT suspects resolved (correct or null).

## Phase 3 — Crush exit (re-pull → re-export → re-measure)

3.1 **Snapshot GT (#51):** back up canonical `set_ground_truth` (BB11+BB12) + the current
    `labeling/fixtures/{bb11,bb12}_ground_truth.yaml`; confirm tag `wip/bb12-enrichment-backup`.
3.2 Re-pull BB12 (`1fsnxchk`) and BB11 (`2nvzlh2k`) — **no `--prune`** (annotator tags are
    user territory). This refreshes the truncated manifest + emits fresh
    `content_catalog.json` (now built from the *corrected* `track_audio`) + re-syncs stems
    (fixes the stale-demucs coverage gap seen in verification).
3.3 Re-export both via `labeling.export_als_to_gt`. **Verify per set:**
    - slots 028/031/144 (BB12) bind correct-or-abstain; **no** `2p25k23p`/`1q8nc02p`/`2uq9800f` on a wrong slot.
    - 148w1 now correct or abstain (Phase-2 fix flowed through the fresh catalog).
    - `id_source` stamped on every row; content coverage above `ID_COVERAGE_MIN` (0.5).
    - `gt_als_gate` green (yaml == export(.als)).
3.4 **GATE (canonical GT write):** `write_back_ground_truth` to canonical `set_ground_truth`
    for both sets (dry-run first).
3.5 `/align-checkpoint` → regenerate `docs/alignment_status.md` on the de-poisoned GT —
    the first honest post-Crush numbers. **This is Crush exit.**

## Rollback

- Phase 2: restore the snapshotted `track_audio` rows.
- Phase 3: restore `set_ground_truth` from 3.1 snapshot; the yaml fixtures are in git.

## Out of scope (tracked, not this program)

- Full-corpus mis-link cleanup (Phase-1 suspects not touching BB11/BB12).
- **The systemic fix for the wrong-recording mis-attach class** (this program only
  cleans the GT-affecting instances in Phase 2). The prevent+represent+remediate full
  fix — ledger `recording` axis, fail-closed same-song guard on `acquire_variant`,
  corpus audit — is its own effort (PR #70):
  [docs/superpowers/specs/2026-07-22-stem-recording-misattach-fix-design.md](../specs/2026-07-22-stem-recording-misattach-fix-design.md).
- Same-title / wrong-version `track_audio` registration errors (the acknowledged blind
  spot content binding inherits) — owned by ingest/identity, separate effort.
- Step 3 audio round-trip law (#37).
