# Operation Crush — master plan (2026-07-21)

> **This is the operative master.** It supersedes the framing of
> [operation_crush_assault_plan.md](operation_crush_assault_plan.md) (kept as the
> detailed discrepancy register) and folds in everything decided since:
> the intelligence-framework thesis, the Ableton-safety protocol, the canonical
> GT rollback, the full worktree census, the experiment-revival ledger, and the
> SSOT fence. Paper-mapping stays in
> [operation_crush_research_synthesis.md](operation_crush_research_synthesis.md);
> the post-Crush sequence stays in
> [operation_rolling_thunder_proposed.md](operation_rolling_thunder_proposed.md).
> **No alignment metrics are typed here** — cite
> [alignment_status.md](alignment_status.md) per AGENTS.md §3.

---

## 0. Why Crush exists — the intelligence-framework thesis

Operation Crush is not a cleanup chore. It is the act of building a **trustworthy
music-intelligence substrate** — the thing every downstream product draws on
without re-litigating truth. This is the **Tesla data-engine paradigm**: once the
sensors are calibrated and the answer key is certified, the same data engine that
labels one set becomes a *platform* the whole team builds on freely.

The chain, stated plainly:

```
certified data  →  honest metrics  →  trustworthy alignment intelligence  →  free internal app development  →  we are empowered
   (Crush)          (one SSOT)          (Rolling Thunder)                       (mashup compiler, DJ agent,
                                                                                 personalization, audit product, lab)
```

**What "empowered" means concretely.** The repo is step 1 (align) of a five-step
arc that ends in personalized mix generation; `lab/` is the north-north star;
the mashup compiler / Appleseed studio / autonomous-DJ stack already exist as
prototypes. Today they cannot be *trusted* end-to-end because the intelligence
they'd stand on is measured against poisoned labels. The moment the substrate is
certified, those applications can consume the intelligence **freely** — a mashup
tool that trusts "these two spans are the same recording," a DJ agent that trusts
"this transition is real," a personalization layer that trusts the alignment GT —
because the truth was fixed *once*, at the source, behind gates that keep it true.

**The Tesla mapping (the skipped step made explicit):**

| Tesla / Waymo | Our operation | What it is here |
|---|---|---|
| Sensor calibration + GT integrity | **Operation Crush** | Answer key + gates: `track_audio_id`, full axes, Ableton ↔ DB ↔ yaml agree. *Cannot mine failures or pseudo-label on poisoned labels.* |
| Shadow mode + re-benchmark | **Rolling Thunder** | Re-run experiments on clean GT; audit UI; honest SSOT on n≥3 GT sets. |
| Fleet deploy + trigger mining | **Scale op (TBD)** | 20k–40k sets; abstention triggers; pseudo-label pool; active labeling queue. |
| **Product platform on trusted autonomy** | **Empowerment (continuous)** | Internal apps built freely on the certified intelligence — the payoff the whole sequence is *for*. |

The discipline that makes empowerment safe rather than reckless: **applications
consume the intelligence through the same gates that certified it** — no app reads
raw ungated GT, no number ships that isn't from the SSOT. Freedom downstream is
bought by rigor upstream.

---

## 1. State delta since the assault plan (2026-07-21)

The assault plan (2026-07-20) is one day stale in two ways that *reduce* scope —
verify before re-doing work:

- **Already on `main`:** PR #39 (`fix/bb12-inventory-audio-repair`) is merged —
  BB11/BB12 GT slot-label remap, Demucs→acappella promotes, slot-collision
  quarantine, inventory-coherence NEUTRAL ledger. Assault-plan Phase 0.2 is
  **half done**; only `feat/track-audio-id-index` remains unmerged of that pair.
- **`.als` backups now exist** (this session): all 64 `.als` + 7 manifests
  snapshotted in-tree (`~/aligning/_backups/als_snapshot_20260721_091013/`) **and**
  off-tree (`~/als_snapshots/…`), integrity-verified. The single biggest
  irreversible risk in Phase 1 is now insured.

And one way that *increases* scope — the new headline discrepancy:

- **D16 — the consolidation is ~7× bigger than stated.** Not "three branches + one
  worktree." Live state: ~20 local branches, **25 active worktrees** with unmerged
  commits, one **detached-HEAD** worktree (`fail-closed-audio-resolvers` `fa3d4cc`
  — data-loss risk), and `trm-ablation-framework` **34 ahead / 28 behind** origin
  (diverged, not a fast-forward) atop 89 dirty files. Tracked in **#49**.

---

## 2. Augmented discrepancy register (delta only)

The full D1–D15 register lives in the assault plan. New/changed:

- **D16 — worktree/branch sprawl + detached HEAD + diverged trunk** (#49). The #1
  rigor risk restated: truth is fixed in ~25 places and read from none of them.
- **D17 — no canonical GT rollback** (#51). Phase-1 write-back overwrites pi
  `set_ground_truth` with no retained pre-image; a wrong re-export replaces
  poison with poison and loses the diff.
- **D18 — `.als` depth-fragility unguarded** (#50). `.als` files reference samples
  by *relative path at fixed depth*; any move to a new depth or any audio
  rename orphans all references at once (BB12 = 574 refs at depth 3). Mitigated
  now by backups; the durable fix (Collect All & Save) is pending the decision in §5.
- **D19 — SSOT invariant unfenced** (#53). D11 was fixed by hand; nothing
  mechanically stops the next hand-typed metric outside `alignment_status.md`.
- **D20 — Relink-by-name re-injects wrong-version (D5) into GT.** BB12's offline
  clips must be relinked, but the "obvious" successor files differ in *identity*.
  **Proven concretely (2026-07-21):** the clip at `stems/121__Manse - All Around/
  instrumental.flac` has two same-name matches — the renumbered `034__Manse - All
  Around/instrumental.flac` (**14 MB**) and the position-121 original in
  `_orphan_slot_collisions/` (**34 MB**). *Different sizes = different audio.*
  Relinking to the renumbered 034 would silently poison a clip whose warp/offsets
  were tuned against the 34 MB original. (Also: `117 Mode (Remix)` → `033 Mode (Jay
  Hardway Remix)`; `127 …​.m4a` → `035w2 …​.flac` format change ⇒ likely a different
  rip.) **Relink rule:** bind by content (hash/fingerprint) + `recording_id` and
  exact-position original, **never by name/number similarity**. (Folds into Phase-1
  relink and issue #50.)
- **D21 — BB12 `.als` is mixed-provenance** after the partial Collect-All: some
  clips point at `tracks/` (new slot numbering), some at frozen `Samples/Imported/`
  copies carrying *old* numbering + `-1` dedup suffixes — a live D1/D2 (path-stem
  classify, slot-number parse) surface. The re-export must normalize provenance.
- **D22 — Stale doc/memory pointers.** Auto-memory (`project_canonical_gt_sessions`,
  `project_gt_set_status`) and any doc pointing at the old BB12 `_backups/…` path
  now point at an emptied directory; and BB12's canonical home
  (`_labeling/1fsnxchk/…`) matches neither the adopted `<set_dir>/<BBNN> align
  Project/` convention nor BB11's layout — reconcile the convention or the path.

---

## 3. The phased plan (self-contained; [NEW] marks additions to the assault plan)

**Standing doctrine (elevated to law):** *never tune, train, or draw a conclusion
on spans whose GT has not passed the gates.* Nothing in a later phase consumes
data a prior phase has not certified.

### Phase 0 — Consolidate: land the truth where it is read

1. **[NEW] Worktree/branch census first** (#49). Enumerate all 25 worktrees;
   classify each **land / discard / park-with-note**; **rescue the detached HEAD
   onto a named branch before anything else**; then reconcile the 34↑/28↓
   divergence on `trm-ablation-framework` (supervised, not blind).
2. **[NEW] Freeze order** (#49). Pause every tuning/training branch (`e1-*`,
   `cotrain-*`, `acap-oracle-ladder`, `instance-separability`,
   `earliest-instance-tiebreak`) until Phase 3. They may develop *methods*; they
   may not ship a *number* until the referee is clean.
3. Open focused PRs for the coherent streams (acquisition cascade #41, identity
   verifier + eval #45, scorer hardening + fixture diagnostics D1). Merge the
   remaining `feat/track-audio-id-index` (D9; its pair #39 is already on main).
4. Land the GT-gate stack: PR #34 (`make gt-gate` + stamped write-back) then
   PR #37 (audio round-trip law), after the two Mac-side manual checks.
5. Commit or explicitly discard the `fp-hit-decoder-clean` worktree — ledger the
   FAILED strict prove either way (D15). **[NEW] Per-branch merge gate:** each
   consolidated branch must pass `make check` + its own tests + review *before*
   landing — no merge-on-faith, because "the fix already exists" is exactly the
   claim that needs a gate.
6. Unify the EXPERIMENTS ledger; adopt one `alignment_status.md` SSOT (D11, D12).
7. Resolve/close PR #35 (noise-floor experiment; carry corroboration code only if
   it survives a clean rebase).
8. **[NEW] Scaffold the Checked System Contract** — `contract/registry.py` +
   rendered `SYSTEM_CONTRACT.md` + static-plane checks C1–C3, green on day one.
   Design: [specs/2026-07-21-checked-system-contract-design.md](superpowers/specs/2026-07-21-checked-system-contract-design.md).
   It turns Crush's data-truth invariants from "verified once by a human" into
   "verified every build," and its data-plane checks (C4–C7) activate through
   Phases 1–2. This is the structural counter to the Type-II miss that started
   Crush. **Caveat (per Fable review):** only static C1–C3 run in CI day one; the
   data-plane poison-catchers (C5/C6) run Mac-side (need GT + pi DB) and their
   named tooling currently lives only on the *unpushed* branch — so "the poison
   would have failed the build" is the *target*, not yet true. The spec now adds a
   content-binding `.als`-ref check (the actual poison mechanism); see the spec's
   revision note. (#49-adjacent; own issue on land.)

### Phase 1 — Ground-truth de-poisoning (the centerpiece)

Goal: **the `.als` on disk, the exported GT, and `set_ground_truth` on pi are
provably the same object.**

0. **[NEW] Safety pre-gate (partly done).** `.als` snapshot taken this session;
   codify as `scripts/backup_als.sh` + `make backup-als` and require it green
   before any relocation/re-export (§4, #50).
1. **Operator decision (blocking, ~30 min)** — the `.als` relocation + path
   convention (§5). Everything below queues behind it. **Front-loaded** — surfaced
   with options pre-analyzed so it is a *decision*, not a research session.
2. **Capture-rule decision (D2):** stem-provenance precedence. Proposed
   *arranged-audio truth > tracklist claim > file path* — **[NEW] treated as a
   hypothesis**, validated against the Honest `42w3` case **and a counter-example**,
   not asserted; then encoded in `labeling/als/identity.py` + regression test.
3. **[NEW] Canonical rollback (#51):** snapshot pi `set_ground_truth` + the
   fixture *before* overwrite; diff old↔new as a reviewed gate artifact; keep a
   one-command revert.
4. Re-export BB11 + BB12 through `make gt-gate` → fixture regen → id audit
   (`scripts/audit_gt_recording_ids.py` must report **zero** stale ids) →
   transactional write-back (coordinated, AGENTS.md §5).
5. Close D3: filter `silence_reason` in `fibers/gt_als.py` + test.
6. **CI fence:** the capture-fidelity suite joins `make check` so GT poisoning is
   a *class* killed by machine.

**Exit:** every BB11/BB12 row passes id-audit + audio round-trip + spectrogram
spot-review; the sha256-bound gate stamp is committed; a retained pre-image exists.

### Phase 2 — Audio truth: right song, right version, canonical everywhere

Goal: the system serves the correct recording for the axes we claim
(`version__stem__variant`), and abstains loudly otherwise.

1. **Gate 1 — matchable incorporation (D6):** audio counts only with
   `track_audio` row + `is_reference` + fingerprint + MERT. Wire into
   `scripts/run_cascade.py` + `make check-inventory`; `--strict-inventory` on by
   default for GT sets.
2. **Gate 2 — verified closure:** a case closes only when the scorer confirms the
   residual it was opened for is gone.
3. **[NEW — scope seam fixed]** Re-source is **GT-set-bounded in Crush**
   (BB10/11/12/13). The ~17.9k May 6–9 corpus cohort (D5) is a **Scale-op line
   item**, not on Crush's critical path — the assault plan's "corpus-wide" framing
   overreached. Learned + heuristic verifiers become the gate's verification
   channel; fold in FIGMA audio↔text verification (#41).
4. Canonical stem debt (D7): pi-side Demucs promotes for BB11's blocking slots.
5. work_id hygiene (D8 / #45) on the canonical DB.

**Exit:** zero *GT-set* slots resolving to unverified audio; every consumed
`track_audio` carries a passing gate verdict or an explicit quarantine/proxy.

### Phase 3 — Re-measure everything (the honest re-baseline)

One regeneration event, one SSOT. Only after Phases 1–2.

1. Re-run the full regeneration block of `alignment_status.md` on de-poisoned GT.
   **[NEW] Known risk:** BB11 stalls Whisper in `make race` — run `1fsnxchk`
   (BB12) alone with `--reuse-base`, per the race-offline note.
2. Re-run TRM diagnostics (#44) on clean GT — the overfit/memorization/sim2real
   findings were measured against contaminated labels and are currently
   *unfalsifiable*.
3. Re-score #2 (intro-grab), #3 (instance disambiguation), #4 (tempo_ratio):
   each proceeds only if its failure class survives clean GT.
4. Re-run UnmixDB unchanged (synthetic, unaffected) to keep the synthetic-vs-real
   contrast valid.
5. **[NEW] Counterfactual line:** record not just *that* numbers moved but *how
   much* poison moved them — a pre-registered expectation ("14 wrong ids should
   drop identity residual by ~X"). If clean GT barely moves the numbers, that is
   itself a finding.

**Exit:** one dated+SHA `alignment_status.md` on main that every doc cites.

### Phase 4 — The SOTA offensive (on certified data only)

Three curves, never one scalar (identity ~solved; walls are placement + structure).

1. Iteration speed: cache the vocal-enhance step (#46).
2. Structure axis: trajectory decoder + pseudo-label flywheel against clean
   referees; lever is AUTO-coverage + pool scale (BB13+), not GPU.
3. Placement axis: agentic placement + gated-ml decode as the default target.
4. Transition model (#42): DJtransGAN differentiable fade/EQ → `gain_curve` /
   `audible_*`; feeds both GT capture and scoring realism.
5. Identity: frozen except gate duty.
6. **[NEW] Experiment-revival replay (#52):** the dead experiments (TRM, PWS,
   cotraining, labeling functions, all attic KILLs) are re-run here per the §6
   ledger — each Bucket-1 revival pre-registers a kill line before it runs.

### Phase 5 — Keep it correct (standing invariants)

1. No mutable-string joins — resolve by `track_audio_id` / `recording_id` only.
2. Gates over vigilance — GT write-back needs the stamp; alignment needs
   `--strict-inventory`; acquisition needs gate 1+2; `make check` carries
   capture-fidelity + round-trip + **[NEW] `.als` backup + SSOT fence**.
3. **[NEW] One SSOT, fenced (#53):** a guardrail check rejects alignment-metric
   strings typed outside `alignment_status.md` — D11 killed as a *class*.
4. One branch discipline — worktrees per agent, PRs through the gate, push daily;
   no >1-week unpushed work (weekly audit).
5. Ledger before re-litigating — EXPERIMENTS verdict check gates any new experiment.

---

## 4. Ableton `.als` integrity protocol (#50)

**Why this is its own section:** John has been burned by "move audio → ruined
Live set." The cause is structural, so the protection must be structural.

**Mechanism.** An `.als` stores *relative sample paths at a fixed folder depth*,
not the audio itself. BB12's `.als` points three folders up (574 refs). Any of
these orphans every reference at once: moving the `.als` to a different depth,
moving the audio it points to, or renaming that audio.

**Done this session (2026-07-21):**
- Timestamped snapshot of all **64 `.als` + 7 manifests** →
  `~/aligning/_backups/als_snapshot_20260721_091013/` (in-tree) **and**
  `~/als_snapshots/als_snapshot_20260721_091013/` (off-tree mirror, survives an
  `~/aligning` wipe). 16 MB. BB11/BB12 sessions verified as valid gzip.
- **Canonical GT-session naming convention (adopted 2026-07-21):**
  **`bbNN_align.als`** — lowercase, underscore, no spaces (agent- and shell-safe),
  living at `~/aligning/<set_dir>/<BBNN> align Project/bbNN_align.als`. The export's
  `find_default_als` (handoff Step B) must search for this name. Supersedes the
  spaced `BB11 align.als` / `big bootie 12 labeling_fast.als` names.
- **Collect All & Save status (corrected 2026-07-21 after Fable adversarial review —
  the earlier "102 broken / 68 gone" numbers were a parsing bug: refs containing
  `&amp;` were not HTML-unescaped, so every "Artist & Artist" track mis-counted as
  missing).**
  - **BB11 — resolves today but is NOT self-contained.**
    `~/aligning/2nvzlh2k__…/BB11 align Project/bb11_align.als`: **280 external /
    280 local** refs; 560/560 resolve *now*, but renaming anything under the set's
    `tracks/` orphans all 280 — same filename-drift exposure as BB12 had. The prior
    "564/567 local, done" claim was wrong (searched only depth-3 `../../../`; BB11
    uses depth-2 `../../`). Do **not** treat BB11 as immune. (One ref points into
    another user's home `…/nsh/Library/…` — investigate.)
  - **BB12 — relocated (2026-07-21), verified reference-neutral:** now
    `~/aligning/_labeling/1fsnxchk/BB12 align Project/bb12_align.als`. Correct
    counts (HTML-unescaped): **589/613 resolve, 24 broken occurrences, 7 unique
    broken paths.** The move was a **depth-preserving `mv`** (this violated §4's own
    copy-only rule — noted; it happened to be reference-neutral, proven by resolving
    the pre-move snapshot against the new base → identical 7 broken paths). Collect
    All **half-ran** earlier (local `Samples/Imported/` grew ~140 refs incl. Ableton
    `-1` dedup copies) — not "abandoned"; the `.als` is now **mixed-provenance** (some
    clips → `tracks/` new numbering, some → frozen `Samples/Imported` old numbering),
    a fresh D1/D2 drift surface (see D21). **Recovery: broken GT audio survives
    locally — no re-pull needed.** Relink method used (2026-07-21): *restore the
    identity-correct file to the path the `.als` expects* (no GT mutation, fully
    reversible), sourcing from — in priority — (1) the project's own
    `Samples/Imported/` collected copy (same basename = the labeled-against audio by
    construction), (2) the exact-position original, incl. from `_orphan_slot_
    collisions/` quarantine; **never** a renumbered same-name file without a
    content/size check (see D20). **Status: 6 of 7 restored (609/613 refs resolve).
    Pending: `stems/121__Manse - All Around/instrumental.flac`** — restore the 34 MB
    position-121 original from `_orphan_slot_collisions/` (create the dest dir
    first), NOT the 14 MB `034` renumber. Then this becomes a Phase-1 gated
    provenance-normalization (D21) before GT re-export.

**Protocol going forward:**
1. `scripts/backup_als.sh` (codifies the snapshot) runs green before any
   relocation/re-export — a Phase-1 pre-gate.
2. **Relocation is copy-only and depth-preserving.** Never move an `.als` to a
   new depth without a relink (assault-plan Step A).
3. **Durable fix (recommended):** open BB12 in Live → **Collect All and Save**
   into a canonical project folder → the project becomes self-contained and
   permanently immune to future audio moves. Costs ~5 GB disk (267 GB free).
4. **Human gate:** open the relocated/collected `.als` in Live and confirm
   samples resolve **before** any fixture overwrite or pi write-back.

---

## 5. The one decision only John can make (front-loaded)

Everything in Phase 1 queues behind the `.als` relocation + path convention.
Options, safest first:

| Option | What happens | Cost | Trade-off |
|---|---|---|---|
| **A. Collect All & Save** *(recommended)* | Open BB12 in Live, Collect-All into a canonical project folder; re-export from that self-contained copy | ~30 min + ~5 GB | Permanently ends depth-fragility; the durable answer to the recurring pain |
| **B. "go" — copy `.als` only** | Copy the 493 KB `.als` to a depth-3 canonical path, refs preserved, renamed | ~2 min | Lightest; sufficient for export; still depth-fragile if audio later moves |
| **C. "go full" — copy project folder** | Same as B but also copy the whole 5 GB project (Samples/Backup/) | ~5 GB | Ableton opens the copy with its own samples; not self-contained the way Collect-All is |

Recommendation: **A**. It costs the same ~30 min as B/C but removes the failure
*class* instead of dodging it once — consistent with the "gates over vigilance"
doctrine. Backups are already in place, so A is fully reversible.

---

## 6. Experiment-revival replay ledger (#52)

Poisoned GT made prior verdicts **unfalsifiable — not wrong.** "Unfalsifiable" ≠
"was succeeding." Each dead experiment earns **one** honest re-run on clean GT
with a **pre-registered kill line**, or stays dead. Triage into three buckets:

- **Bucket 1 — RE-RUN (verdict depended on labels).** These might change on clean
  GT and are the reason Crush exists: **TRM** sim2real/flywheel (#44),
  **continuous PWS**, **cotraining**, **labeling / weak-supervision functions**,
  instance selection, placement/structure residuals. Each re-runs in Phase 4 with
  a kill criterion written first.
- **Bucket 2 — STAYS DEAD (GT-independent failure).** Failed for signal reasons
  clean labels can't fix: chroma-surprise ranker (AUC ≈ .55), key-change breaks
  chroma, fine-placement chroma fail. Re-running these is wasted time.
- **Bucket 3 — DO NOT TOUCH (double-killed).** **PWS categorical offset-bins —
  refuted twice; the ledger says do not re-test.** Only the *continuous/cotrain*
  PWS flavor is live. Same rule for any attic KILL marked double-sourced.

Deliverable: walk `workspaces/alignment_prototype/attic/EXPERIMENTS.md` + the
closed-experiments ledger and tag every entry 1/2/3 before any revival begins.

---

## 7. GitHub map

| Phase | Existing | New this session |
|---|---|---|
| 0 Consolidate | #48, #35, PR #34/#37 | **#49** census + freeze + reconcile |
| 1 GT de-poison | #40, #47 | **#50** `.als` protocol · **#51** GT rollback |
| 2 Audio truth | #41, #45 | (#41 gets FIGMA verification note) |
| 3 Re-measure | #44, #2/#3/#4 | (BB11 Whisper + counterfactual notes) |
| 4 SOTA | #46, #42 | **#52** experiment-revival ledger |
| 5 Invariants | #8 | **#53** SSOT fence |

Milestone: **Operation Crush**, due 2026-08-01.

## 8. Definition of done

Crush closes when: (1) both GT sets pass the full gate with committed stamps and
verified pi write-back **behind a retained pre-image**; (2) no alignment-consumed
audio is ungated; (3) `alignment_status.md` is regenerated once on clean data and
is the only live number source, **fenced by guardrail**; (4) the TRM / flywheel /
driver verdicts *and the revival ledger's Bucket-1 set* are re-established against
that baseline; (5) the gates — including `.als` backup and the SSOT fence — run in
CI so none of this regresses silently. **Then** the intelligence substrate is
certified and internal application development is empowered to build on it freely.
