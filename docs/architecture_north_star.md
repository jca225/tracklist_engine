# Architecture north star — the aligner as an operating system

Drafted 2026-07-09. Companion to
[alignment_objective.md](alignment_objective.md) (what Point B *is*) and
[entropy_reduction_plan.md](entropy_reduction_plan.md) (the contracts
workstream this builds on). This doc names the architecture the repo has been
converging on, defines "production grade" measurably, and sequences Point A →
Point B. It proposes **no new top-level folders** before the promotion
milestone (P6) — the repo rule stands.

## Point B, stated precisely

`align(set) -> Timeline` such that:

1. **Input** = {tokenized tracklist, track audios, set audio} — nothing else.
2. **Output** = a `PredictedTimeline` that round-trips to a hand-convention
   `.als` (varying master tempo, unwarped mix) via `labeling/als`.
3. **Abstain, never lie** — every span carries confidence + an abstention
   route; the UnmixDB benchmark showed this is our differentiator, not a
   weakness.
4. **SOTA claim** = beats André-2024 NMF on the external benchmark where the
   comparison is fair, AND holds ≥ hand-tuned-stack numbers on held-out BB
   transfer (the axiom test of the low-rank worldview).
5. **Unattended** — corpus-wide runs on the cluster with no flag archaeology
   and no human in the inner loop (humans answer *abstentions*, not runs).

## The OS map

The design already half-exists. Naming it makes the gaps visible:

| OS concept | ours | state | gap to production grade |
|---|---|---|---|
| **ABI / syscall types** | `core/contracts` (typed records, id NewTypes, `join_guard`, slot normal form) + `core/timebase.TimeMap` | built (A1/A2) | manifest/GT records still partial; **no provenance/freshness fields yet** (plan law 4) |
| **Kernel** | the aligner behind `drivers/EndToEndDriver` — `align_set(ctx) -> Timeline` | built as a 3-way race (classical / agentic / ml) | no single *default* kernel entrypoint; winners live as flags in `infer`, not defaults of one command |
| **Device drivers** | `harness/` probes (chroma, fp, HuBERT, continuity, path_decode) + `harness/axes.py` routing | built | lyrics + surprise still enter via `agentic/live_runners`, not the probe contract; no capability registration (which axis/stem a probe serves is implicit) |
| **Page cache** | `mert_store`, `.cache/fp_instr/`, whisper transcript caches, HuBERT feature caches | ad-hoc, per-module | **the biggest gap**: no shared artifact cache keyed by (content hash, extractor version, params); staleness class caused 18 fix-commits |
| **Virtual memory / pages** | time-windowed feature access (probes already window) | implicit | formalize page = (time domain, window, feature id); lazy materialization for hour-long mixes |
| **Scheduler** | `agentic/` ReAct-POMDP loop: belief → probe → update → commit/escalate | built (`--live`) | probe selection is policy-scripted, not budget-aware information-gain; learned precisions (`neuro/`) not fused into scheduling |
| **Page fault → handler** | escalation = active labeling (human answers a span the kernel can't) | designed, not wired | abstentions don't yet enqueue anywhere; John's labeling time is unscheduled |
| **Filesystem** | pi-storage canonical layout + manifest contract | built | A4 asset locator pending (path-resolution class: 14 fixes) |
| **Boot / init** | `pull_set_for_alignment` materializes the working set | built | inventory gate (`make check-inventory`) manual, not preflight in the kernel |
| **Perf counters** | `make scorecard` + failure-analysis attribution + `make race` board | built | not in CI; no golden regression net (F2 in the entropy plan) |
| **Userland** | EDA, review UIs (`review/`), `.als` export, personalization | built | fine — consumers stay above the syscall line |

**Two laws the map implies** (both already repo practice, now stated):

- **Nothing crosses the syscall line untyped.** EDA, review, export consume
  `Timeline`/`TimeMap`/contract records — never raw dicts, never a probe's
  internals. (The zero-fix track record of `labeling/als` is the proof this
  works.)
- **The model is a driver, not a fork.** `drivers/ml.py` swaps one kernel
  stage behind the same interface and races on the same scorecard. When it
  wins, it *becomes* the default — no parallel stack, ever. This is how the
  learned aligner replaces the classical one without a rewrite.

## Production grade = the little things, enumerated

Each is checkable; most extend the entropy ratchet:

1. **Determinism** — same inputs + same commit ⇒ same timeline. Pin seeds,
   stamp versions. Test: run the kernel twice on a fixture set, diff.
2. **Provenance** — every artifact self-describing: `produced_at`, producer
   commit, input fingerprints. (Contracts law 4; not yet implemented.)
3. **Loud staleness** — `is_stale()` on load; delete the silent disk-truth
   fallbacks once it lands.
4. **Config as data** — one defaults table in the kernel; a flag that has been
   default-on for two weeks becomes code, its flag deleted (phase-policy
   enforcement, ratchetable: count argparse flags on the kernel path).
5. **Diagnostics as values** — generalize als `Diagnostic(code, severity,
   location)`; kernel emits per-span *why* (which probe, what margin, why
   abstained). Explainability is what makes review cheap.
6. **Golden regression net** — `make race` on both GT sets is the benchmark;
   a CPU-cheap contract+scorer golden subset runs in CI on every push,
   the full race nightly on the Mac.
7. **Abstention as output** — `span.abstained + reason` in the Timeline
   schema, consumed by the active-labeling queue (P4).
8. **No unnamed conventions** — anything two modules both assume (slot normal
   form, stem routing, GT-axis rule) lives in contracts or harness/axes, not
   in comments.

## Point A → Point B

Phases; each has an exit criterion. Owner key: **[K]** kernel/infra lane
(this doc's lane), **[M]** model lane (parallel agent — trajectory training +
`drivers/ml`, live now), **[J]** John (labeling / listening decisions).

> **Execution detail lives in
> [kernel_data_engine_plan.md](kernel_data_engine_plan.md)** (2026-07-09):
> the estimation-kernel contracts (probe factors, span posteriors,
> calibration), the data-engine mechanics adopted from prior art
> (Snorkel/Tesla/Waymo/FixMatch), and the week-by-week W1–W6 sequencing of
> P1/P2/P4 against Aug 1.

- **P0 — DONE 2026-07-09.** Contracts A1/A2, driver race, taxonomy
  (scripts + prototype satellites), sensor-phase freeze, branch landed.
- **P1 — kernel v1. [K]** `make align SET=<id>` runs the best-known
  composition (classical base + proven winners as *defaults*, not flags)
  through the driver interface, contract-validated, deterministic. Exit:
  fresh clone → one command → current-best timeline; zero flags needed.
- **P2 — the page cache. [K]** One artifact-cache module (likely
  `core/artifacts.py` + a prototype adapter): content-hash keys, extractor
  version, params; provenance fields on read/write; `is_stale()`. Migrate
  mert_store + fp caches + whisper caches onto it. Exit: staleness bug class
  ratcheted to zero silent fallbacks; cache hit = no recompute across drivers.
- **P3 — model becomes the kernel. [M]** TrajectoryDecoder (+ fusion) beats
  classical on the race board held-out; flips to default driver. Exit:
  `make align` runs the learned kernel; classical stays as the racing
  baseline. (Gate is the board, not enthusiasm.)
- **P4 — the data engine. [K+M+J]** Agentic auto-accepted alignments become
  pseudo-labels feeding [M]; abstentions enqueue into an active-labeling list
  (worst-first, like the review UI) that John burns down in minutes, not
  set-at-a-time. BB10/Murph GT happens *this* way. Exit: model retrains from
  the queue without a hand-built dataset; each cycle lifts held-out transfer.
- **P5 — scale-out + open set. [K]** Unattended corpus runs (pi/Vast
  scheduling per the existing loops); the open-set litmus (no tracklist →
  identify + abstain → ACRCloud → human confirm) rides the same kernel with a
  retrieval front-end. Exit: N unseen sets aligned + review-loop-audited with
  zero operator flags.
- **P6 — promotion.** `workspaces/alignment_prototype` → top-level
  `alignment/` (the chain module the root CLAUDE.md always reserved for it);
  attic stays behind; this doc becomes `alignment/CLAUDE.md`'s architecture
  section. Gate: P1–P4 exits held for two consecutive new sets.

**Sequencing note.** P1 and P2 are deliberately [K]-lane and mechanical so
they never block [M]. The race interface is the *coordination protocol*
between the lanes: both improve the same board, neither edits the other's
driver.

## What this doc does NOT license

- No new probes/channels/priors — the sensor-phase freeze stands.
- No top-level folders before P6.
- No re-litigating attic'd experiments; the ledgers are law until new GT
  changes the evidence.
- No "framework" generalization — this architecture serves one goal (the
  aligner) and its one downstream customer (the generation program,
  docs/alignment_objective.md).
