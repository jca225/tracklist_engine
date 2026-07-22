# Operation Crush — frontal assault plan (2026-07-20)

> **⚠ SUPERSEDED 2026-07-21 — the operative master is now
> [operation_crush_master_plan.md](operation_crush_master_plan.md).** This file is
> kept as the **D1–D15 discrepancy register** only; its phase framing and status
> claims are stale (see the master plan).
>
> Produced by a full assessment of every open/merged PR, every issue, the July
> handoff docs, the superpowers plans, the EXPERIMENTS ledger, and the north-star
> docs. Original framing: the operative plan for making the project **rigorous,
> correct, and SOTA** before the August 1 north star. GitHub tracking: the
> [Operation Crush milestone](https://github.com/jca225/tracklist_engine/milestone/1).
> Research context: [operation_crush_research_synthesis.md](operation_crush_research_synthesis.md).
> No alignment metrics are typed here — cite [alignment_status.md](alignment_status.md)
> per AGENTS.md §3.

---

## 0. The verdict, in one paragraph

The data-poisoning that cost a week is real, diagnosed, and **mostly already
fixed in code — but the fixes have not landed where they count.** They are
scattered across three unmerged branches (`fix/bb12-inventory-audio-repair`,
`feat/track-audio-id-index`, `fp-hit-decoder-clean`), two green-but-idle PRs
(#34 GT release gate, #37 audio round-trip ruler), and ~33 unpushed commits on
the local `trm-ablation-framework` checkout. Meanwhile the committed BB12 GT
fixture still carries **stale `track_id`s on 14 slots** (a different song's id),
the stem-from-path capture bug (BB12 `42w3` Honest) is undecided, and the
"single source of truth" status doc exists in **two divergent versions** on two
branches. The single biggest rigor risk is no longer any one bug — it is
**divergence between where the truth is fixed and where the truth is read.**

## 1. Discrepancy register

Numbered so issues/PRs can cite them. Evidence paths in parentheses.

### A. Ground-truth poisoning (Ableton ≠ programmatic GT)

- **D1 — Stale fixture ids.** `labeling/fixtures/bb12_ground_truth.yaml` at HEAD
  resolves **3 confirmed slots** to a *different song's* `track_id` — audit-verified
  2026-07-21: **028** Beatles "Can't Buy Me Love" → Garrix "In The Name Of Love";
  **031** CCR "Have You Ever Seen The Rain" → Killers "When You Were Young"; **144**
  Snakehips "All My Friends" → Two Friends "Pacific Coast Highway (Acappella)"
  (`2uq9800f`). *(The earlier "14 slots" was over-counting by the audit's old
  ≥2-token matcher + a header-drop bug; both fixed in `audit_gt_recording_ids.py`.
  Separately: 3 `tlp*` placeholder ids not in the DB, 3 blank-name rows.)*
  Mechanism: manifest `recording_id=None` ×165 + near-total
  `.als`↔manifest path divergence (290/296 clips path-MISS) → `slot_id_map`
  fallback carries stale ids through `export_als_to_gt`
  (`docs/archive/agent_handoff_als_relocation_20260719.md`). Scorer hardened to show
  these as honest misses (`d700e94`, unpushed); **the fixture itself is still
  poisoned.** Re-export is blocked on the operator's `.als` relocation /
  path-normalization decision.
- **D2 — Stem-from-path capture bug.** `labeling/als/identity.py::classify_path`
  stamps `claimed_stem` from the sample *file path*, so an instrumental
  `online_candidate` file under an acappella slot exports as instrumental GT
  (BB12 `42w3` Honest; model + ear agree, label wrong —
  `docs/archive/agent_handoff_spectrogram_review_gt_capture_20260719.md`,
  `eda/alignment/failure_analysis/FOLLOWUPS.md` WS0). The capture rule
  (path-stem vs arranged stem vs tracklist claim) is **undecided**.
- **D3 — `fibers/gt_als.py` bypasses the silence filter.** The deactivated-track
  phantom-span fix (`silence_reason`) landed in the GT exporter, but the fibers
  GT reader still reads `.als` clips directly without filtering — a declared,
  unclosed follow-up (FOLLOWUPS WS0 caveat).
- **D4 — GT slot_label drift.** ~155 BB12 GT rows carry old ALS numbering; the
  remap tool exists only on `fix/bb12-inventory-audio-repair`.

### B. Wrong audio / wrong version

- **D5 — May 6–9 bulk download is pre-QA.** ~17.9k of 18k `track_audio` rows
  predate variant-aware search, chromaprint QA, three-axis identity, and the
  correction ledger; wrong-remix/version rips (incl. the near-semitone detunes
  and the 46 s preview-clip class) trace to it
  (`docs/alignment_objective.md`, FOLLOWUPS WS5). Re-sourcing is on the critical
  path and has no owner issue with a gate.
- **D6 — "Downloaded ≠ matchable."** 11 BB12 GT recordings have *no matchable
  reference at all* (no row / no `is_reference` / no fingerprint / no MERT) —
  the largest identity-residual chunk
  (`docs/superpowers/specs/2026-07-18-acquisition-data-engine-design.md`).
  Phase-1 case machinery (`core/acquisition_case.py`, `--open-cases`) is wired;
  gate 2 (scorer-verified closure) is not.
- **D7 — Pi `wrong_stem` debt.** BB12 acappella claims Demucs-promoted only on
  the repair branch; BB11 has ~26/152 blocking slots still open canonically.
  Mac-side fixes do not fix canon.
- **D8 — work_id grouping suspect.** Identity-verifier eval false positives on
  "I Need Your Love" point at split `work_id` (issue #45); evidence
  (`alignment_status.md` §8, reports) exists **only in unpushed local commits**.

### C. Local ↔ canonical drift

- **D9 — Manifest↔pi divergence** (BB12 `recording_id=None`, local stem
  rewrites not reflected on pi) and **manifest↔`.als` path-convention
  divergence** (D1's mechanism) remain unreconciled on main; the durable
  reconcile CLI + `--strict-inventory` preflight live only on the repair branch.
- **D10 — Machine drift.** Mac-absolute paths in manifests silently starved the
  gpubox agentic runs (0 AUTO); fixed by `aligning_paths.py`, currently
  untracked/uncommitted.

### D. Process / record integrity

- **D11 — Two SSOTs.** `docs/alignment_status.md` regenerated 2026-07-11 on this
  branch vs 2026-07-19 ("after canonical BB11/BB12 Ableton GT regeneration") on
  the repair branch. A doc whose premise is *single* source of truth has two
  versions.
- **D12 — EXPERIMENTS ledger divergence.** The working-tree ledger lacks the two
  2026-07-19 inventory-coherence verdicts (BB12 NEUTRAL) that exist on the
  repair branch — an agent here could re-litigate a closed hypothesis.
- **D13 — 33 unpushed commits, no PR.** Acquisition cascade (#41 territory),
  identity verifier + eval (#45 evidence), DJtransGAN scaffold (#42 territory),
  scorer hardening (D1) — all invisible from origin; violates AGENTS.md push
  discipline; data-loss risk.
- **D14 — Issue/milestone hygiene.** #44/#45/#46 carry the Crush label but were
  not in the milestone; #43's sub-issue list omitted them; milestone due-date
  (07-31) disagreed with issue bodies (08-01); #40 tracks none of its delivered
  acceptance items (PRs #21/#23/#26/#27/#28/#39, open #34/#37); #8's body lists
  findings already fixed in PR #10; #44 doesn't know PR #35 already ran E1 to a
  noise-floor verdict; #2/#3/#4 cite pre-cleanup failure tables.
- **D15 — Un-landed fp SOTA worktree.** The stem-gated belief / `ref_fp_for_span`
  work from `docs/agent_handoff_fp_sota_integration_20260718.md` is partly
  *uncommitted in a worktree* (`fp-hit-decoder-clean`) — at risk of loss; its
  strict prove FAILED, so it must not land without re-proving.

---

## 2. The assault plan

Five phases, strictly ordered. **Nothing in a later phase may consume data a
prior phase has not certified.** The standing rule from the spectrogram handoff
is elevated to doctrine: *never tune, train, or draw conclusions on spans whose
GT has not passed the gates.*

### Phase 0 — Consolidate: land the truth where it is read (1–2 days)

The prerequisite for everything. Order matters.

1. Push the 33 local commits (reconcile with `origin/trm-ablation-framework`,
   which was merged as PR #19 and had main merged back) and open PRs for the
   three coherent streams: acquisition cascade (#41), identity verifier + eval
   (#45), scorer hardening + fixture diagnostics (D1). (D13)
2. Merge `fix/bb12-inventory-audio-repair` and `feat/track-audio-id-index` to
   main (reconcile CLI, Lux proxy, `--strict-inventory` preflight, GT slot
   remap, Demucs promotes, quarantine, `audio_index`). (D4, D7, D9)
3. Land the GT-gate stack: update + merge PR #34 (`make gt-gate` + stamped
   write-back) then PR #37 (audio round-trip law on the stamp), after running
   the two outstanding Mac-side manual checks. (blocks Phase 1)
4. Commit or explicitly discard the `fp-hit-decoder-clean` worktree remnants —
   ledger the FAILED prove in EXPERIMENTS either way. (D15)
5. Unify the EXPERIMENTS ledger and adopt the repair branch's regenerated
   `alignment_status.md` as the one SSOT; delete/mark the stale copy. (D11, D12)
6. Resolve or close PR #35 (conflicting; its experiment sits at noise floor —
   carry the corroboration code only if it survives rebase cleanly).

### Phase 1 — Ground-truth de-poisoning (the centerpiece)

Goal: **the `.als` on disk, the exported GT, and `set_ground_truth` on pi are
provably the same object.**

1. **Operator decision (blocking, ~30 min of John's time):** `.als` relocation +
   path-normalization convention (574 depth-3 relative sample refs), per the
   als-relocation handoff's Step A. Everything below queues behind this.
2. **Capture-rule decision (D2):** decide stem provenance precedence
   (arranged-audio truth > tracklist claim > file path), encode it in
   `labeling/als/identity.py`, and regression-test the Honest 42w3 case.
3. Re-export BB11 + BB12 GT through the full gate: `make gt-gate` (validate →
   anchor check → als_audit → audio round-trip) → fixture regen → id audit
   (`scripts/audit_gt_recording_ids.py`) must report **zero** stale ids →
   transactional write-back to pi (PR #28 path) — coordinated, per AGENTS.md §5.
4. Close D3: filter `silence_reason` in `fibers/gt_als.py` + test.
5. CI fence: the capture-fidelity test suite
   (`tests/labeling/test_export_capture_fidelity.py`, `test_export_id_coverage`,
   round-trip law) becomes part of `make check` so GT poisoning is a
   *class* killed by machine, not a bug killed by hand.

**Exit criterion:** every BB11/BB12 GT row passes id-audit + audio round-trip +
spectrogram spot-review; the gate stamp (sha256-bound) is committed.

### Phase 2 — Audio truth: right song, right version, canonical everywhere

Goal: **the system downloads, stores, and serves the correct recording for the
axes we claim** (`version__stem__variant`), and abstains loudly otherwise.

1. **Acquisition gate 1 — matchable incorporation (D6):** downloaded audio only
   counts when it has a `track_audio` row + `is_reference` + fingerprint + MERT.
   Wire into the cascade executor (`scripts/run_cascade.py`) and `make
   check-inventory`; promote the inventory gate from manual to aligner preflight
   (`--strict-inventory` on by default for GT sets).
2. **Acquisition gate 2 — verified closure:** a case closes only when the scorer
   confirms the residual it was opened for is gone (acquisition-design Phase 2).
3. **Re-source the May 6–9 cohort (D5):** run `scan_wrong_versions.py
   --open-cases` corpus-wide; burn down the worklist through the gates for GT
   sets first (BB10/11/12/13), long tail after. The learned identity verifier +
   heuristic verifiers (local commits) become the gate's verification channel;
   fold in FIGMA-style audio↔text verification per the research synthesis (#41).
4. **Canonical stem debt (D7):** pi-side Demucs promotes for BB11's blocking
   slots (coordinated write), so Mac and pi agree.
5. **work_id hygiene (D8):** resolve #45 on the canonical DB; add a
   sibling-grouping audit query to the weekly audit.

**Exit criterion:** zero GT-set slots resolving to unverified audio; every
`track_audio` consumed by alignment carries a passing gate verdict or an
explicit quarantine/proxy record (Lux Holm pattern).

### Phase 3 — Re-measure everything (the honest re-baseline)

Only after Phases 1–2. One regeneration event, one SSOT.

1. Re-run the full §Regeneration block of `alignment_status.md` on de-poisoned
   GT: scorecard, strict + fiber-aware, per-axis, oracle placement ceiling,
   `make race` (classical / agentic / ml drivers), LOSO. Re-stamp date + SHA.
2. Re-run the TRM diagnostics (#44) on clean GT — the three findings (overfit /
   memorization / sim2real) were measured against contaminated labels and are
   currently *unfalsifiable*.
3. Re-score the evidence base of issues #2 (intro-grab), #3 (instance
   disambiguation), #4 (tempo_ratio): the 2026-07-14 failure tables predate the
   cleanup; each issue proceeds only if its failure class survives clean GT.
4. Re-run the UnmixDB external benchmark unchanged (it is synthetic, unaffected)
   so the synthetic-vs-real contrast in
   [alignment_recharacterization.md](alignment_recharacterization.md) stays valid.
5. Record deltas: every number that moved gets a line in the corrections
   ledger; anything that can't be regenerated gets `⚠ unverified — carried`.

**Exit criterion:** a single dated+SHA-stamped `alignment_status.md` on main
that every other doc cites; no live number older than this event.

### Phase 4 — The SOTA offensive (on certified data only)

Per the recharacterization: three curves, never one scalar. Identity is
~solved; the walls are **placement** and **structure**, and the paper's rigor
claim is the decomposed, real-mix, generalization-aware evaluation itself.

1. **Iteration speed first:** cache the vocal-enhance step (#46) — the measured
   pipeline unlock for the acappella-heavy loop.
2. **Structure axis:** trajectory decoder + pseudo-label flywheel, now against
   clean referees. E1's noise-floor verdict (PR #35) says the current lever is
   AUTO-coverage (cue channel, stem lattice, ACCEPT bar) and pool scale (BB13+),
   not GPU. Third GT set unlocks the learned instance selector and LOSO n=3.
3. **Placement axis:** the heavy tail (mid-song entries over medley beds);
   agentic driver is the placement champion — compose agentic placement +
   gated-ml decode as the default target (per SSOT §4 reading).
4. **Transition model (#42):** DJtransGAN-style differentiable fade/EQ to
   estimate `gain_curve`/`audible_*` — feeds both GT capture (Phase 1's audible
   windows) and scoring realism.
5. **Identity:** frozen except gate duty (verifier in the acquisition gate);
   MERT-330M / learned layer-weighting only if a Phase-3 residual demands it.
6. Every experiment pre-registers kill criteria and lands its verdict in
   EXPERIMENTS.md (the axis-contrast KILL is the model).

### Phase 5 — Keep it correct (standing rigor invariants)

1. **No mutable-string joins** — audio and GT resolve by
   `track_audio_id`/`recording_id` only; fail-closed resolvers stay fenced by
   entropy_audit.
2. **Gates over vigilance** — GT write-back requires the stamp; alignment runs
   require `--strict-inventory`; acquisition requires gate 1+2; `make check`
   carries the capture-fidelity + round-trip suites.
3. **One SSOT** — numbers only in `alignment_status.md`; regeneration is the
   only edit path; corrections ledgered.
4. **One branch discipline** — worktrees per agent, PRs through the gate, push
   daily; no >1-week-old unpushed work (weekly audit checks this).
5. **Ledger before re-litigating** — EXPERIMENTS.md verdict check is part of
   proposing any experiment.

---

## 3. GitHub mapping

| Phase | Tracking |
|---|---|
| 0 Consolidate | new issue: branch/PR consolidation (D11–D13, D15); PRs #34, #37, #35 |
| 1 GT de-poisoning | #40 (keystone), new issue: BB12 fixture re-export + `.als` relocation (D1, D2); FOLLOWUPS WS0 |
| 2 Audio truth | #41 (gate), #45 (work_id), D5 re-source under #41; #4 folds into #40's provenance schema |
| 3 Re-measure | #44 (TRM re-run) + new issue: full SSOT regeneration; #2/#3 re-scored here |
| 4 SOTA | #46 (cache), #42 (DJtransGAN), flywheel (PR #35 successor) |
| 5 Invariants | #8 (weekly audit, now down to one Low finding) |

Milestone: **Operation Crush**, due 2026-08-01, now contains #40–#46 plus the
new issues above.

## 4. Definition of done

Operation Crush closes when: (1) both GT sets pass the full gate with committed
stamps and pi write-back verified; (2) no alignment-consumed audio is ungated;
(3) `alignment_status.md` is regenerated once on clean data and is the only
live number source; (4) the TRM/flywheel/driver verdicts are re-established
against that baseline; (5) the gates run in CI so none of this regresses
silently.
