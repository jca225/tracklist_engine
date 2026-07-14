# Design: Three-Regime Lab Platform

**Date:** 2026-07-14
**Status:** Design approved (structure); execution deferred to post-Aug-1 (see Timing).
**Author:** John Abrahams (w/ Claude)
**Companion docs:** [architecture_north_star.md](../../architecture_north_star.md) ·
[alignment_status.md](../../alignment_status.md) ·
[appleseed_empowerment_layer.md](../../appleseed_empowerment_layer.md)

---

## 1. Motivation

`tracklist_engine` today is one flat pile of top-level modules. The next level of
maturity is to re-conceptualize it as **a workshop with three rooms (regimes)** that
material flows through — a lab platform whose end product is *products*, and whose
labs can be operated by agents (human or non-human). The metaphor the project is
converging on: **an assembly line for knowledge workers**, taking inspiration from the
Tesla Data Engine (a self-mining data flywheel) and Mcity (a controlled environment to
test in).

This is a **re-conceptualization + light restructure of what already exists**, not a
greenfield rebuild and not a pivot. The alignment north star (SOTA aligner across ~20k
sets, gate ≈ Aug 1) is **unchanged and still outranks this work**. The engine's own OS
map ([architecture_north_star.md](../../architecture_north_star.md)) previously
forbade "framework generalization"; this doc is the *disciplined* generalization — it
adds three drawers and a shelf, and explicitly refuses to build a speculative
framework (see §4).

---

## 2. The model

Three **labs**, one per regime, with material flowing left→right — but as a **DAG with
feedback**, not a straight line:

```
        ┌───────────── feedback / recirculation (Markov) ─────────────┐
        ▼                                                             │
  ┌────────────┐        ┌───────────────┐        ┌──────────────┐     │
  │ COLLECTION │  ───►  │  UNDERSTANDING │  ───►  │   LEARNING   │ ────┘
  │ (Regime 1) │        │  (Regime 2)    │        │  (Regime 3)  │
  └────────────┘        └───────────────┘        └──────────────┘
        │                      │                        │
        └──────────────┬───────┴────────────┬───────────┘
                       ▼                     ▼
                  ┌─────────────────────────────┐
                  │   PRODUCT SHELF             │  ← any lab may emit a product
                  │ mashup_compiler · cast ·    │
                  │ appleseed studio · findings │
                  └─────────────────────────────┘
```

- **Collection** — bring raw material in (mixes, tracklists, songs; later Spotify /
  SoundCloud sources).
- **Understanding** — study it (audio analysis, EDA, corpus research).
- **Learning** — teach models + build (labeling, the aligner, personalization).

**Four properties of the model:**

1. **Left→right flow, with no-ops.** A regime may pass material straight through if it
   has nothing to add.
2. **DAG with feedback (Markov).** Outputs recirculate as inputs; rooms self-loop and
   reiterate as tools improve. The data engine is the canonical loop: alignments →
   pseudo-labels → re-enter Learning.
3. **Products from any lab, not just the last.** Understanding can ship a product (a
   corpus finding, a report) exactly as Learning ships the aligner or a mashup.
4. **Standard boxes only across boundaries.** The *only* thing that crosses a lab
   boundary is a typed `core/` contract record (a `Recording`, `SetAudio`, `Timeline`,
   `GroundTruth`) — never a raw dict, never another lab's internals.

---

## 3. Compositional growth ("operator algebra")

Inside a lab, tools are **generators**; you *compose and combine* them to make new
capabilities; when a combination stabilizes it becomes a **new basis** — a new
primitive that everything downstream re-expresses itself in.

The important insight is that **this behavior is emergent from the model above, not new
machinery**. An operator algebra needs one thing: **closure** — any two operators
compose into another valid operator. Here, closure *is* the standard-box rule: if every
tool takes standard boxes in and emits standard boxes out, any tool composes with any
other. Closure is a *consequence* of the box rule, not a framework to write.

| Rich idea | What actually produces it (cheap) | Already present as |
|---|---|---|
| Compose / combine operators | standard-box rule = closure under composition | `harness/` probes fused at decode |
| **New bases** (new primitives) | promote a stable combination to a *named new box type* | `workspaces/` → promoted module (P6) |
| Markov "stay & reiterate" | an artifact cache so re-runs don't recompute | P2 page-cache; mert/fp caches |
| Feedback / recirculation | products re-enter as inputs | P4 data engine (alignments → pseudo-labels) |
| Products from any room | shelf accepts a feeder line from any lab | mashup (Learning), corpus findings (Understanding) |

---

## 4. What this design does NOT license

- **No new repo.** The engine stays one git repo, restructured in place. (A separate
  platform repo is justified only if non-DJ verticals ever sit beside the engine — the
  deferred generic case.)
- **No abstract `Regime` / `Operator` / plugin base class, no composition DSL.** The
  algebra lives in plain typed functions + the box rule + the promote-to-new-primitive
  move. The math is intuition, not code. This is the "framework generalization" the OS
  map warns against.
- **No `flow.py` (generic DAG executor) yet.** `make align` and existing targets
  already orchestrate; build the runner only when a *second* collection source actually
  needs flowing.
- **The aligner does not move** out of `workspaces/` before its own P6 promotion.
- **The Aug 1 alignment gate outranks this work** (see §8).

---

## 5. Target structure

```
tracklist_engine/                  (same repo, restructured — NOT a new repo)
│
├── core/                          shared floor — identity, contracts, db, timebase (UNCHANGED)
│
├── labs/
│   ├── collection/                REGIME 1
│   │   ├── scrape/        ← web_crawler
│   │   ├── ingest/        ← ingest
│   │   ├── tokenize/      ← tokenizer
│   │   └── sources/       ← NEW scaffold (spotify/, soundcloud/ later)
│   │
│   ├── analysis/                  REGIME 2
│   │   ├── mir/           ← analysis
│   │   ├── eda/           ← eda
│   │   └── research/      ← lab/   (also resolves the lab/ vs labs/ name clash)
│   │
│   └── learning/                  REGIME 3
│       ├── labeling/     ← labeling
│       ├── personalization/ ← personalization
│       └── alignment/    ← workspaces/alignment_prototype  (moves at ITS P6, not in W1)
│
├── platform/                      NEW · thin glue
│   └── registry.py                declares: each lab's regime, contract types it
│                                   produces/consumes, and feedback edges
│
├── workspaces/                    stays root — incubator for pre-promotion forks
├── scripts/ tests/ docs/ deploy/ data/ …   stay root — infra
```

**Platform root (filesystem, on the Mac).** The engine + product repos are grouped
under one plain directory (not a repo). On AWS this becomes an org/account boundary.

```
~/Desktop/<platform-root>/          ← plain dir, not a repo (the "new environment")
├── engine/   → tracklist_engine         (one repo, three labs inside)
└── products/                            (shelf — each keeps its own .git)
    ├── mashup_compiler/   ← ~/Desktop/mashup_compiler
    ├── mashup_demo/       ← ~/Desktop/mashup_demo  (Appleseed studio)
    └── cast/              ← ~/Desktop/cast         (autonomous DJ)
```

---

## 6. Workstreams

Three separable workstreams; one master picture (this doc), separate implementation
plans.

### W1 — Engine `labs/` restructure

- **Nature:** mechanical `git mv` of folders into `labs/{collection,analysis,learning}`
  + import-path fixes + one small `platform/registry.py`.
- **Deploy wiring updated in the same pass:** dormant systemd unit `ExecStart` module
  paths, `Makefile` entrypoints, `vast_loop`/`mac_analyze_loop` module names,
  `guardrails.py` + `.cursor/rules` path assertions. Verified by `make check`.
- **Why safe:** verified 2026-07-14 that **nothing is running** (pi-storage
  unreachable; pi-worker's only tracklist unit `tracklist-ajax-retry` is `failed`). The
  move perturbs *dormant config + import paths*, not live services — the safest window.
  The one caveat: wiring must be correct *before* services are next started, which is
  why deploy config is fixed in the same commit, not deferred.
- **Tooling:** the `refactor-safety` skill (stale-path scan, `Path(__file__).parents[N]`
  fixes, handoff doc for pi-storage ops).
- **`registry.py` scope:** documentation-only to start (declares regime + I/O types +
  feedback edges). Enforcement ("only declared contract types cross a boundary") is a
  later ratchet, not day one.

### W2 — Platform root + product shelf

- **Nature:** plain `mv` of the independent product repos (`mashup_compiler`,
  `mashup_demo`, `cast`) under `~/Desktop/<platform-root>/products/`, and the engine
  under `.../engine/`. Each repo keeps its own `.git`; nothing internal changes.
- **Risk:** low — does not touch `tracklist_engine`'s imports at all. Only nit: the
  engine's local Mac path changes (affects cwd + absolute paths), trivially fixed.
  pi-storage is untouched (canonical state lives there, not these paths).

### W3 — Versioned, abstracted ground truth

The problem: GT's real source of truth is a pile of `.als` files in `~/aligning/` on
**one laptop**, exported into `set_ground_truth`. No history, no provenance, no diff —
and this has bitten repeatedly (deactivated-clip export, gain-window, phantom clips;
"never bare-re-export — merge `gain_curve`").

The fix: make GT a **typed, git-tracked text artifact**; the `.als` stays the human
authoring surface, but the *truth* is a diffable file.

```
.als  (authoring surface, Ableton)
  │ export ─► labs/learning/labeling/ground_truth/<set_id>.yaml   ← VERSIONED SSOT (git)
  │                 │ load
  │                 └─► set_ground_truth  (DB = cache, not truth)
```

Four properties (each maps to a law already in the OS map):

1. **Version-controlled** — canonical YAML in git; a GT change is a reviewable diff.
   The deactivation/gain-window bugs would have appeared as a *visible diff* instead of
   silent DB corruption.
2. **Provenance-stamped** — each export records source `.als` sha256 + exporter commit
   + date + audibility-rule version.
3. **Loud staleness** — `is_stale()` compares stored `.als` hash vs current; fails loud
   if the YAML is behind its `.als`.
4. **Typed abstraction** — a `GroundTruth` contract type in `core/`; exporter writes it,
   aligner + scorer read it. Round-trip stays honest via existing `als_audit` (~97%),
   wired as a `make` check. `.als` binaries archived to pi-storage/S3 by content-hash
   for reproducibility; the git YAML is the truth. (Also what makes GT AWS-portable and
   what the P4 data engine appends to.)

**W3 split by timing (see §8):**
- **Thin slice — now:** when doing the GT re-export already owed on
  `ws0-scorer-deinflation`, also write GT to a git-tracked YAML with a hash + provenance
  stamp. (~1h, rides existing work, de-risks the numbers being trusted for the gate.)
  Initially lands at today's `labeling/ground_truth/` path; relocates under `labs/` in
  W1.
- **Full W3 — post-Aug-1:** `GroundTruth` contract type, `als_audit` in CI, S3 archival.

---

## 7. Data flow & interfaces

- **Boundary interface = `core/` contract records.** No new interface is built; the
  existing typed records ARE the inter-regime interface (OS-map law: nothing crosses the
  syscall line untyped).
- **Registry** (`platform/registry.py`) declares, per lab: `regime`, `produces: [types]`,
  `consumes: [types]`, `feedback_edges: [...]`. Data declaration (~50 lines), not an OOP
  hierarchy. Documentation-only first; enforcement later.
- **No message bus, no runtime regime polymorphism** — three fixed regimes, one team.

---

## 8. Timing (relationship to the Aug 1 alignment gate)

Decision rule: **does it move the Aug 1 needle? If no, and it touches import paths,
wait.**

| Work | Verdict | Why |
|---|---|---|
| **W1 restructure** | **Defer to post-Aug-1** | Gives the gate nothing (aligner stays in `workspaces/`); every move perturbs the daily alignment tooling (`make check/race/scorecard`, `guardrails.py`). Asymmetric risk. |
| **W2 desktop tidy** | Optional, anytime | Isolated `mv` of separate repos; zero alignment risk. |
| **W3 thin GT slice** | **Do now** | GT is the scorer's fuel; you're re-exporting anyway; versioning makes trusted numbers diffable/safe. |
| **This design doc** | **Write now** | Prose only, zero code risk; captures the vision while fresh. |
| **Full W3** | Post-Aug-1 | Real data-model work; bundle with the restructure pass. |

"Nothing is running" makes W1 *deploy-safe* but not *focus-safe* — the dev-loop import
risk is the reason to wait, not the deploy risk.

---

## 9. Open decisions

1. **Platform-root name.** `~/Desktop/<platform-root>/` needs a name. It should *not* be
   `appleseed` (that's a product line on the shelf, and the engine shouldn't sit under a
   product brand). Provisional: `foundry/`. **John's call** — this is the one naming
   decision left open; the doc uses `<platform-root>` until set.
2. (Resolved) Scope = all three workstreams in one doc, separate implementation plans.
3. (Resolved) `registry.py` = documentation-only first, enforcement as a later ratchet.

---

## 10. Success criteria

- **W1:** fresh clone → `make check` green with all modules under `labs/`; a dormant
  service start (systemd/`make deploy`) resolves to the new module paths with zero flag
  archaeology; the aligner's scorecard/race tooling runs unchanged.
- **W2:** engine + products under one platform root; each product repo intact with its
  own history; engine dev loop works from the new path.
- **W3 (thin slice):** GT for the current sets exists as committed, provenance-stamped
  YAML; re-export produces a reviewable diff; DB `set_ground_truth` loads from it.
- **W3 (full):** `GroundTruth` contract type consumed by aligner + scorer; `als_audit`
  gate in CI; `.als` archived by content-hash.
