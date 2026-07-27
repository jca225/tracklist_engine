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

## START HERE (new implementation session)

**What this is:** Operation Crush fixes ground-truth (GT) data-poisoning *before*
building the aligner — a week was lost to a Type-II error (poisoned GT nobody
detected). This doc is the operative master; PR **#54** carries it + the spec.

**Read order (minimal tokens):** this §START HERE → §0 (thesis) + §3 (phases) →
the contract spec [`docs/superpowers/specs/2026-07-21-checked-system-contract-design.md`](superpowers/specs/2026-07-21-checked-system-contract-design.md)
→ `gh pr view 54` for the diff. Only open the [assault plan](operation_crush_assault_plan.md)
if you need the D1–D15 detail. **Don't reload the planning conversation** — these
artifacts are the state.

**State as of 2026-07-21 (verified):**
- **BB12** relinked **613/613**, canonical at `~/aligning/_labeling/1fsnxchk/BB12 align Project/bb12_align.als`
  (each restore gated on the `.als`'s `OriginalFileSize`/`OriginalCrc`).
- **BB11** resolves 560/560 but is **not self-contained** (280 external refs).
- `.als` backups (in-tree + off-tree) exist; `scripts/backup_als.sh` codifies them.
- Issues **#40–#48** + **#49–#53** in the Operation Crush milestone; contract
  **design approved**, spec written (C0 added), pending your spec sign-off.

**Verification policy (token-calibrated):** self-verify with deterministic
re-checks by default (cheap); an **independent model (Fable) only on
consequential/milestone steps, batched** (~100–150k tokens each — not per micro-step);
Haiku for routine second opinions. **Never trust an `.als` ref count without
`html.unescape` + a re-parse** (a missing unescape produced a wrong "102 broken"
that reached docs before Fable caught it).

**STOP for explicit human go (never auto-run):** the `trm-ablation-framework`
branch reconcile (**61↑/105↓ vs `origin/main`** — 52 genuinely-unique commits +
9 already-landed, measured 2026-07-21; a real divergence, not a fast-forward);
any GT mutation / pi `set_ground_truth` write-back; merges to `main`; corpus-wide
actions.

**Immediate next actions, in order:**
1. **Phase 0** — worktree census + **prune the (clean) detached-HEAD worktree**
   `fail-closed-audio-resolvers` (#49 — `fa3d4cc` is already on `origin/main`,
   working tree clean, so this is *not* a data-loss rescue, just a prune);
   **freeze** the tuning branches; then the branch reconcile *(human-gated)*.
2. Scaffold the contract: `contract/registry.py` + `SYSTEM_CONTRACT.md` + static
   checks **C1–C3** (green day one); wire **C0** (`.als` ref content-binding).
3. **Phase 1** — BB12 durable clean state = **Collect All & Save in Live** (operator);
   capture-rule decision (D2); canonical GT rollback (#51).

**Standing hazards:** `--prune` is **FORBIDDEN** on `1fsnxchk` until Collect-All
(re-orphans the 4 restored `tracks/` files); PRs need `gh auth switch --user jca225`
(the READ account silently drops `--milestone`/`--label`).

**Correction log (2026-07-21, post-Fable adversarial review — read before Phase 1):**
- **GT poison is 3 confirmed cross-song ids, not "14".** The "14" was an artifact
  of `audit_gt_recording_ids.py`'s over-eager matcher (≥2 shared ≥4-char tokens)
  *plus* a header-drop parsing bug. Fixed here: the referee now buckets **poison
  (3: slots 028 Beatles→Garrix, 031 CCR→Killers, 144 Snakehips→Pacific Coast
  Highway) / not-in-DB (3 `tlp*` placeholders) / unverifiable-blank-name (3)**.
  It is a **recall-biased screen, not a certifier** — the authoritative de-poison
  is the `track_audio_id` binding in Phase-1 step 3b, never name overlap.
- **Reject the hand-patch shortcut.** Patching the 3 fixture ids alone cannot even
  go green and leaves the poison *carrier* armed. The poison lives in **three
  derived artifacts that must be regenerated together**: `bb12_ground_truth.yaml`,
  `labeling/fixtures/id_maps/1fsnxchk_slots.json` (the `slot_id_map` bridge —
  `export_als_to_gt.py:170-176`, whose "slot_label is deterministic … exact, not
  a guess" claim is **false** under the 2026-07-15 renumber), and
  `bb12_ground_truth.inventory.json`. De-poison only via step 3b.
- **The capture-fidelity "CI fence" is not yet real.** `test_export_capture_fidelity.py`
  lives only on the unpushed branch (absent on `main`/CI) and self-skips because
  `DEFAULT_ALS` pointed at a now-deleted Desktop path (repointed here to the
  canonical `_labeling/1fsnxchk/BB12 align Project/bb12_align.als`). Even when it
  runs, it re-exports through the same `slot_id_map` and so cannot catch id poison
  — it fences the loop-merge class, not D1. Phase-1 step 6 "class killed by machine"
  is the *target*, not yet true.
- **`make gt-gate` does not exist yet** — PR #34 adds it; nothing may invoke it
  before #34 merges (Phase-0 step 4 lands it first — ordering is consistent).

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
  worktree." Live state: **~30 local branches**, **~27 active worktrees** with unmerged
  commits, one **detached-HEAD** worktree (`fail-closed-audio-resolvers` `fa3d4cc`
  — **not a data-loss risk after all: `fa3d4cc` is already on `origin/main`, working
  tree clean; prune it, don't "rescue" it**), and `trm-ablation-framework`
  **61 ahead / 105 behind `origin/main`** (52 genuinely-unique commits; diverged,
  not a fast-forward). The 89 dirty files were the sharp edge (~2,500 lines of
  never-committed contrast/invariance/flywheel work) — **saved 2026-07-21 in local
  checkpoints**; only the WIP poisoned `bb12_ground_truth.yaml` remains uncommitted.
  Tracked in **#49**.

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
  rename orphans all references at once (BB12 now = 318 external depth-3 refs +
  295 local `Samples/` after a partial Collect-All; was ~574 pre-collect). Mitigated
  now by backups; the durable fix (Collect All & Save) is pending the decision in §5.
- **D19 — SSOT invariant unfenced** (#53). D11 was fixed by hand; nothing
  mechanically stops the next hand-typed metric outside `alignment_status.md`.
- **D20 — Relink-by-name re-injects wrong-version (D5) into GT.** BB12's offline
  clips **were relinked 2026-07-21 (see §4)**; the rule below stands for BB11 + all
  future relinks, because the "obvious" successor files differ in *identity*.
  **Proven concretely (2026-07-21):** the clip at `stems/121__Manse - All Around/
  instrumental.flac` has two same-name matches — the renumbered `034__Manse - All
  Around/instrumental.flac` (**14 MB**) and the position-121 original in
  `_orphan_slot_collisions/` (**34 MB**). *Different sizes = different audio.*
  Relinking to the renumbered 034 would silently poison a clip whose warp/offsets
  were tuned against the 34 MB original. (Also: `117 Mode (Remix)` → `033 Mode (Jay
  Hardway Remix)`; `127 …​.m4a` → `035w2 …​.flac` format change ⇒ likely a different
  rip.) **Relink rule (mechanical, no external data needed):** gate every relink on
  the `.als`'s own recorded **`OriginalFileSize` + `OriginalCrc`** — it is embedded
  ground truth about the exact bytes labeled against, retiring the "same basename =
  same file" assumption entirely; then confirm `recording_id`. **Never** relink by
  name/number similarity. (Folds into Phase-1 relink and issue #50; this is also the
  basis for the contract's C0.)
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

1. **[NEW] Worktree/branch census first** (#49). Enumerate all ~27 worktrees;
   classify each **land / discard / park-with-note**; **prune the clean
   detached-HEAD worktree** (`fa3d4cc` is on `origin/main` — no rescue needed);
   then reconcile the **61↑/105↓ vs `origin/main`** divergence on
   `trm-ablation-framework` (supervised, not blind — 52 unique commits + 9
   already-landed; drop the 9 on rebase).
2. **[NEW] Freeze order** (#49). Pause every tuning/training branch (`e1-*`,
   `cotrain-*`, `acap-oracle-ladder`, `instance-separability`,
   `earliest-instance-tiebreak`) until Phase 3. They may develop *methods*; they
   may not ship a *number* until the referee is clean.
3. Open focused PRs for the coherent streams (acquisition cascade #41, identity
   verifier + eval #45, scorer hardening + fixture diagnostics D1). Merge the
   remaining `feat/track-audio-id-index` (D9; its pair #39 is already on main).
4. Land the GT-gate stack: PR #34 (`make gt-gate` + stamped write-back) then
   PR #37 (audio round-trip law), after the two outstanding Mac-side manual checks
   (**name them from PR #34/#37 before executing** — undefined here).
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
1. **Operator decision (§5)** — relocation is already DONE; what remains is
   **Collect-All (durable form) + the D22 convention/path reconcile**. The re-export
   below queues behind the Collect-All choice (it fixes the final paths).
2. **~~Capture-rule decision (D2)~~ — SUPERSEDED (2026-07-21, see §9).** The
   "arranged-audio > tracklist claim > file path" precedence was the wrong question:
   it re-ranks *strings*. D2 folds into content-binding — `claimed_stem` becomes the
   **resolved `track_audio` row's stem** (content, not path); `classify_path` is
   demoted to a *cross-check that flags disagreement, never stamps*. No precedence
   decision is needed. Do NOT implement the precedence rule.
3. **[NEW] Canonical rollback (#51):** snapshot pi `set_ground_truth` + the
   fixture *before* overwrite; diff old↔new as a reviewed gate artifact; keep a
   one-command revert.
3b. **[REVISED 2026-07-21 — bind by content, delete the guesser; see §9] Content
   catalog binding, not path-matching.** The old "reconcile paths *or* bind via
   `audio_index` — pick one" left the poison **carrier armed**: the `slot_id_map`
   fallback (`export_als_to_gt.py:165-178`, live on `main`) re-injects stale ids on the
   next path drift, and its "slot_label is deterministic — exact, not a guess" claim is
   false under a renumber. **Mandate the binding, and DELETE the guesser:** resolve
   every clip through the content-addressed catalog (`OriginalFileSize`+`Crc` →
   `head_hash` → **else abstain**, evolving `audio_index.json`); **remove `slot_id_map`,
   `_load_slot_id_map`, the `id_maps/*_slots.json` fixtures, and the weak tiers of
   `match_manifest_for_path`.** Abstentions (`recording_id: null` + diagnostic) become
   a human worklist — exactly the 7-mismatch review the BB12 relink already was. Note:
   BB12's manifest carries **152 locally-applied `recording_id`s**
   (`resolve_manifest_recording_ids.py --apply`, uncommitted) — commit or re-derive.
4. Re-export BB11 + BB12 through `make gt-gate` → fixture regen → id audit
   (`scripts/audit_gt_recording_ids.py` must report **zero** stale ids) →
   transactional write-back (coordinated, AGENTS.md §5).
5. Close D3: filter `silence_reason` in `fibers/gt_als.py` + test.
6. **CI fence:** the capture-fidelity suite joins `make check` so GT poisoning is
   a *class* killed by machine.

**Exit:** every BB11/BB12 row passes id-audit + audio round-trip + spectrogram
spot-review; the sha256-bound gate stamp is committed; a retained pre-image exists;
**[NEW, per §9] `slot_id_map` and the guess-tiers are DELETED (grep-clean outside
`attic/`), every GT row carries `id_source: content|abstain`, and the renumber-
metamorphic test (C1b) is green** — otherwise the current exit criteria pass with the
poison carrier still armed.

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
   much* poison moved them — a pre-registered expectation calibrated to the
   **3 confirmed cross-song ids** (not the retracted "14"; see the Correction
   log), e.g. "3 wrong ids on a 172-track set should drop identity residual by
   ~X." If clean GT barely moves the numbers, that is itself a finding — and one
   the small true count makes *likely*, so register the expectation honestly.

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
not the audio itself. BB12's `.als` points three folders up (318 external refs +
295 now-local after a partial Collect-All). Any of
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
    uses **depth-1 `../`** into its set dir). Do **not** treat BB11 as immune.
    (The `…/nsh/Library/…` ref is a **device preset** `.adv` (Simple Delay), not
    audio and not in the 560 audio refs — not an integrity lead.)
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
    locally — no re-pull needed. Relink DONE 2026-07-21: 7/7 restored, 613/613 refs
    resolve** (verified twice, incl. an independent Fable pass). Method: *restore the
    identity-correct file to the path the `.als` expects* (no GT mutation, reversible),
    each match **gated on the `.als`'s own recorded `OriginalFileSize`/`OriginalCrc`**
    (the file identity *at labeling time* — the correct key, embedded in the `.als`;
    698 of them). Manse `121` proved it: the labeled-against original is the 35,441,994-
    byte quarantine file (crc 18999, May 7), **not** the same-named 14,833,697-byte
    `034` renumber (Jul 15).
    - **⚠ Standing hazard — `--prune` is FORBIDDEN on `1fsnxchk` until Collect-All.**
      `prune_orphans` deletes untagged `tracks/` files absent from the manifest; the
      4 restored `tracks/` files (117/127/140/140w1) qualify → a future `pull --prune`
      would silently re-orphan the GT. (Confirmed in `pull_set_for_alignment.py`.)
    - **Durable end-state (needs operator in Live):** file-restore is tactical; do
      **Collect All & Save** to internalize all 613 refs into `Samples/`, making
      prune/renumber permanently irrelevant, then normalize provenance (D21) and
      re-export through the gates. Human gate (open in Live, confirm 613 resolve)
      mandatory before any fixture overwrite / pi write-back.

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

## 5. The decision left for John

The BB12 **relocation is DONE** (§4: canonical at `_labeling/1fsnxchk/…`, relinked
613/613).

**(i) Durable form of BB12 — DONE (2026-07-21, Fable-verified).** Collect All &
Save is **complete**: `bb12_align.als` = **307 active audio refs, all inside
`Samples/`, 0 external, 0 offline** (the earlier "307 collected / 306 external"
was a `<SourceContext>` historical-ghost miscount — see
[[feedback_als_ref_parsing_unescape]]). Depth-fragility and the `--prune` hazard are
now permanently closed **for BB12**. ⚠ Only `bb12_align.als` is the collected session;
the sibling `bb12.als` / `big bootie 12 labeling_fast.als` / `..._slow_ARCHIVED.als`
open with 295–317 OFFLINE clips — never open those.

**(i-bis) [NEW — BB11 is NOT collected].** BB11's `bb11_align.als` still carries
**280 external refs** (a mix of depth-1 into its set dir); it resolves today but is
one `tracks/` rename/renumber from the exact BB12 fire. **Add BB11 Collect All & Save**
(operator, in Live) to §5 as a required Phase-1 durable-state step, then the same
content re-export. Until then, `--prune` is forbidden on BB11 too.

**(ii) Convention vs. path reconcile (D22).** BB12's home `_labeling/1fsnxchk/BB12
align Project/` matches neither the §4-adopted `<set_dir>/<BBNN> align Project/`
convention nor BB11's in-set-folder layout (the depth-3 refs forced it). Decide:
relax the convention to allow `_labeling/<set_id>/`, **or** move BB12 into its set
folder as part of the Collect-All (which frees the depth constraint). Encode the
choice in the canonical-als registry (contract C6).

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

Deliverable: walk `alignment/attic/EXPERIMENTS.md` + the
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

Also open in the milestone: **#43** ("human-readable TLDR") — the START HERE block
partly serves it; keep it as the plain-language digest.

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

---

## 9. Root-cause respec (2026-07-21, post-Fable) — the path/identity disease

A deep adversarial review (Fable) of the whole Ableton/`.als`/file-path bug family
found these fires are not independent — they share a root cause, and Crush as first
written **detects poisoned outputs while leaving the poison *generators* in the code.**
This section supersedes the D2 capture-rule decision and re-specs contract C1. Full
evidence: memory `project_path_identity_root_cause`; enforcement gaps verified on `main`.

**The disease — two coupled failures + one substrate:**
- **A — location-as-identity.** The same string is both a mutable human label
  (rename / renumber / Collect-All flatten / move) *and* a machine join key. Exercise
  the mutability → every join silently breaks or guesses wrong.
- **B — optional identity + guess-shaped recovery (the deeper half).** Stable ids
  (`recording_id` / `track_audio_id`) exist and are clean where used, but are
  **optional at every seam**; on a miss the code **guesses down a fallback ladder**
  (`match_manifest_for_path` = 6 tiers, `slot_id_map` = a 7th) instead of abstaining.
  BB12's GT poisoning needed **both** (drift armed B, a mutable slot number fired A).
- **Substrate — Ableton's relative-path-at-depth model** (unchangeable) — but Live
  *also* embeds per-clip content identity (`OriginalFileSize`+`OriginalCrc`) our code
  ignored until now. The project's own `entropy_reduction_plan.md` quantifies the
  family: id-namespace 18 / stale-artifact 18 / path-resolution 14 / silent-defaults
  14 / identity-axis 12 fixes, vs ~0 in the typed `.als` codec core.

**Enforcement gaps Fable verified (the plan said X ≠ X enforced):** contract C1's old
binding ("extend `entropy_audit` fences, mostly exists") is a **paper tiger** —
`entropy_audit` fences only 3 classes, there is **no resolver fence**; the `slot_id_map`
carrier is **still live on `main`** (Crush never mandated deleting it); `audio_index`
is on `main` but the **GT export path never consults it**; C0's `.als` content-bind has
**zero implementation**; **BB11 is not collected** (280 external refs).

**The fix — one principle:** *paths are locators, never identity; identity binds by
content, once, at one layer; a missing binding is a loud abstention — there is no ladder.*
1. Codec reads `OriginalFileSize`/`Crc` into `ParsedClip` (also dissolves the
   active-ref-vs-SourceContext ambiguity: locator / provenance / identity = three fields).
2. One content-addressed catalog per set (evolve `labeling/audio_index.json`).
3. One resolver, no ladder: `(size,crc)` → `head_hash` → **else abstain(Err)**;
   **delete `slot_id_map` + the weak tiers** (Phase-1 step 3b + exit criterion).
4. **D2 folds in** — `claimed_stem` = resolved row's stem; `classify_path` demoted to a
   disagreement-flag. The precedence decision is **canceled** (Phase-1 step 2).
5. **Class-killer = the renumber-metamorphic test (contract C1b):** rename/renumber
   every file + move the `.als` one depth, re-export, assert GT **byte-identical**.
   Green ⇒ diseases A+B dead in the export path *by construction*. Plus C1a: every GT
   row stamped `id_source: content|abstain`, guardrail-checked.

**Sequencing:** cheap/agent-executable — codec fields, resolver + `slot_id_map`
deletion, `id_source` stamp + guardrail, metamorphic test (fixture matrix exists).
Human-gated — abstention worklists, **BB11 Collect-All** (§5 i-bis), fixture/pi writes.
This *sharpens* Phase 1 and C1; it is not a rival plan. Chromaprint stem/transcode
binding (C0(c) full form) stays a Phase-2 verifier — `size+crc+head_hash` covers this
corridor; don't block on it.
