# Provenance-First Engine — Convergence Plan (Point A → B)

> **Status:** DRAFT (2026-07-25). Proposes the migration of the current engine
> toward the redesign in `~/workspace/alignment_algorithm/dj_engine_pseudocode.md`
> ("Provenance-First DJ Alignment Engine"). Not yet ratified. Supersedes nothing
> until Phase 0 lands.

## 0. Thesis, and what this plan refuses to be

The redesign is **correct and it targets the actual disease.** Every poison
incident in [alignment_state_of_record.md](alignment_state_of_record.md) — #15
through #19, the timeline-provenance gap, the stale-apparatus measurement artifact
(#18) — is one instance of a single class: *identity/coordinates keyed on a mutable
locator (path/slot), no provenance, no abstention discipline.* The pseudocode's §0
rules and §16 `assert_system_laws` make that class **structurally impossible**. The
invariants are a formalization of decisions we already reached empirically (Crush =
content-addressing = rule #4; "abstain, never lie" = a system law; the three-axis
recharacterization = rule #7). That convergence is the strongest evidence the design
is right.

**What this plan refuses to be: a big-bang rewrite.** Taken literally the design is
a multi-month build — content-addressed artifact store, provenance graph, Snorkel-style
Claim/Evidence/Belief/Snapshot/PseudoLabel tables, promotion gates, eventually
Postgres. A stop-the-world version would (a) strand Crush B–E, RT1, and the cotrain
branch, (b) commit months before a single vertical slice proves the abstraction
survives contact with the gain-audible scorer / fibers / odd-ratio warps, and (c)
leave the **actual gate — placement/structure — untouched the entire time.** The
rewrite is worth doing; it is **not the gate.** If we solved placement tomorrow on
the messy substrate we'd have a working aligner with ugly provenance (a good
problem); if we build the perfect substrate and don't solve placement we have honest
proof of failure.

**Method: strangler-fig, first slice chosen for algorithmic payoff.** The design
becomes the canonical target now. New work lands in the new shape. Invariants become
enforced laws immediately. Modules migrate **one vertical slice at a time**, and the
*first* slice is sequenced so it delivers a concrete result (the first honest
identity number = RT1) rather than pure plumbing. Every phase yields a real number;
we can stop after any phase with a strictly better codebase.

## 1. The canonical contract

- **Authority:** `dj_engine_pseudocode.md` — §0 fundamental rules, §16
  `assert_system_laws`, §11 promotion gates. This is a *behavioral contract*; the
  repo is checked against it, not the reverse.
- **In scope:** the alignment DAG — `scrape → tokenize → ingest → analyze →
  {labeling | alignment} → cotraining`. The "honest fork at analyze" (README): both
  labeling and alignment circle back to ingest on wrong-version downloads.
- **Out of scope (unchanged, off the DAG):** `lab/`, `personalization/`,
  `soundcloud/`. Do not entangle them.

## 2. Module map — current → target

| Pseudocode section | Current home | Migration note |
|---|---|---|
| §2 Artifact store + provenance | *(new)* `core/provenance/` | content-addressing partly exists in Crush's content catalog — **reuse, don't rebuild** |
| §4 Observations (source ≠ truth) | `tokenizer/materialize.py`, `web_crawler/` | `set_track_slots.claimed_*` become **Observations**, never identity |
| §5 Claims + Evidence | `pws_aligner/` | the weak-supervision matrix already lives here |
| §6 Audio acquire/analyze | `ingest/`, `analysis/` | wrap downloaders/analyzers as versioned `ProcessSpec` runs |
| §7 Immutable Ableton labels | `labeling/als/`, `ableton_interpreter` | content-bind is **already built** (Crush) — formalize as `HumanLabelAssertion` |
| §8–9 Fitted models + AxisBeliefs | `alignment/` | wrap MERT/fp/chroma probes as belief models per axis |
| §10 Timeline decoder | `alignment_prototype/path_decode.py` etc. | must consume posteriors, emit `None`-not-`0.0` |
| §11 Snapshots + promotion gates | *(new)* | the §16 laws become executable gates |
| §12 Cotraining + pseudo-label lineage | `alignment_prototype/cotrain.py`, cotrain branch | reconcile the branch **into** this shape, don't merge as-is |

**Open decision (Phase 0):** do the new spine primitives (`Artifact`, `Observation`,
`Claim`, `Evidence`, `Belief`, `Snapshot`) live in `core/provenance/` adopted by each
stage, or in a new top-level `engine/`? New top-level requires justification per
CLAUDE.md; a whole engine may qualify. Default: `core/provenance/` until it strains.

## 3. Phases

Each phase ends with a **real number or a shipped invariant**, and a go/no-go gate.

### Phase 0 — Ratify the contract *(days)*
1. Bring `dj_engine_pseudocode.md` into the repo as canonical architecture authority;
   cross-link + demote the ad-hoc plans (`architecture_north_star.md`,
   `alignment_objective.md`, `operation_rolling_thunder_proposed.md`) to "informs".
2. **Law audit:** each of §16's 21 laws → `PASS / VIOLATED / N/A` vs the current repo,
   with `file:line` evidence. This is the prioritized poison worklist.
3. Encode the cheap violated laws as executable fences in `scripts/entropy_audit.py` /
   `make check` (many start `xfail`; each turns green as a phase closes it).
- **Deliverable:** the contract is authority; a regression fence exists; a ranked
  worklist. **Antidote to the degeneration the README is about** — every future PR is
  checked against the contract instead of taking the path of least resistance.

### Phase 1 — Identity/provenance slice = RT1 *(weeks)* — **the load-bearing first slice**
The still-open half of decision #19 *is* the first slice, and it *is* the honest
re-measure. Sub-steps:
1. **1a — Artifact store + content addressing.** Minimal immutable `Artifact`
   (`content_sha256`, `payload_sha256`, derivation edges) over the existing pi object
   store. **Fold in Crush Phase B** — the content-history hash ledger IS the
   `Derivation` store; do not run B–E as a separate track.
2. **1b — Observation layer.** `set_track_slots.claimed_*` materialize as
   `Observation`s (source said, not truth). Directly kills #19's claim-spine-as-identity.
3. **1c — Identity claims/evidence/belief.** Build the **real per-slot candidate pool**
   (claim + siblings + corpus) that #19 flagged as unwired; run the MERT-stem-routed
   (L3 vocal / L22 instr) + pitch-normalized fingerprint evidence sources that
   **already work in eval** (open_set_acappella_identity_findings.md); emit
   support/contradict/abstain; produce `AxisBelief(IDENTITY)` with a margin-based
   decision rule.
4. **1d — Rebuild the bridge id_map generator** (deleted → silently rotted, #18) as a
   provenanced run; regenerate BB11/BB12.
5. **1e — Re-infer identity on clean canonical + score per-set-per-axis** → first honest
   identity number since Crush → `alignment_status.md`.
- **Deliverable:** first trustworthy identity number (RT1); the candidate-pool wiring
  #19 needs; the identity spine in the new shape. Retires the biggest poison **and**
  unblocks measurement in one slice.
- **GATE:** does the new spine pay for itself, or is it friction? Has the abstraction
  survived the gain-audible scorer + fibers? Go/no-go before Phase 2.

### Phase 2 — Placement/structure beliefs + provenanced timeline *(weeks–months)*
1. Wrap `path_decode` / `trajectory` / scorer as `EvidenceSource`s and
   `PLACEMENT`/`STRUCTURE` belief models — **this is mostly *wrapping existing probes*
   in the belief/evidence contract, not solving placement.** The hard research
   continues in parallel *inside* the contract.
2. Timeline decoder consumes **posteriors, not raw margins** (§16 law); emits
   `AlignmentSegment` with `None`-not-`0.0` coordinates — closes the timeline
   provenance-gap wound.
- **Deliverable:** a fully provenanced predicted timeline that round-trips to `.als`;
  honest per-axis numbers for all three axes.

### Phase 3 — Snapshots, promotion gates, cotraining lineage *(months)*
1. Snapshot + promotion machinery (§11); the §16 laws become promotion gates.
2. Cotraining rounds with pseudo-label lineage + round-acyclicity + view-leakage guards
   (§12) — **the flywheel spine, the path to ~40k.**
3. Reconcile the unmerged cotrain branch **into** this shape.
- **Deliverable:** the flywheel runs honestly; scaling path to 40k unblocked.

### Phase 4 — Substrate hardening: Postgres *(months, deferred)*
Migrate SQLite → Postgres so durable invariants are DB-enforced constraints (README's
NB). **Last step, not first** — the durability layer under an already-working design.

## 4. Sequencing against live fronts (nothing stranded)

- **Crush B–E** → **subsumed by Phase 1a** (content-history ledger = the `Derivation`
  store). Absorbed, not stranded.
- **RT1 re-measure** → **is Phase 1's deliverable.** The target, not a casualty.
- **Cotrain branch** → reconciled in Phase 3, not merged as-is.
- **Laws-as-guardrails** → cross-cutting from Phase 0 onward.

## 5. Risks & kill criteria

| Risk | Mitigation |
|---|---|
| Infra-astronomy (months, no accuracy) | every phase ends in a real number; **placement is the gate, not the substrate** — keep the algorithm moving in parallel from Phase 2 |
| Abstraction doesn't survive contact (fibers, gain-audible scorer, odd warps) | Phase 1 slice de-risks; **explicit go/no-go gate** before committing further |
| Solo/agent bandwidth | phases independently valuable; stoppable after any phase with a better codebase |
| Two codebases mid-migration | strangler-fig — old path keeps working until a slice replaces it |
| Poison relocates again | §16 laws as *executable* fences (not docs) catch relocation — this is exactly how #19 slipped past Crush |

## 6. Where I'd be wrong

If the real bottleneck is that the GT/measurement substrate is too rotten to trust any
experiment — i.e. poison is corrupting the *training signal into the algorithm*, not
just the scoreboard — then infra-first gets much stronger, because clean measurement
becomes the precondition for any algorithm work. The record suggests the algorithm
fronts moved on their own merits (so: not that), but this is a judgment call the
operator should confirm.

## 7. Immediate next actions (Phase 0)

1. Copy `dj_engine_pseudocode.md` into `docs/` (or `engine/`) as canonical; index it in
   `docs/design_docs_index.md`.
2. Run the §16 law audit → `PASS/VIOLATED/N/A` table with `file:line` evidence.
3. Stub `tests/laws/` (one test per law, `xfail` until its phase closes it) + wire into
   `make check`.
