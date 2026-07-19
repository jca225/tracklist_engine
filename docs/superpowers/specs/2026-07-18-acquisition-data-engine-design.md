# Acquisition Data Engine — Design

**Date:** 2026-07-18
**Status:** Design (pre-implementation)
**Author:** brainstorming session (John + Claude)

## Problem

The alignment residual is dominated by *acquisition-supply* failures, not model
failures. On BB12, the "identity miss" residual decomposes as:

- **No matchable reference existed** (biggest chunk): 11 GT recordings never
  matched by any predicted span (`FINDINGS.md:116`) — hand-added online-candidate
  acappellas with no id-map, or the Rvmor sided-row gap (SC-only tracklist rows
  with no `data-trackid` never materialize a canonical recording, so no
  fingerprint / MERT / `track_metadata` to match against).
- **Wrong version acquired**: right artist+title, wrong audio — the 46s
  preview-clip → rescue-grabbed-original path (`project_wrong_version_preview_clip`).
- **Genuinely hard core** — thin. Crosstalk is *refuted* as the wall: identity
  holds 83–92% at GT layer-depth ≥3 (`A4:302`).
- **Cross-set slippage** — real but minor (BB11 9% vs BB12 4%); this is a *model*
  issue and is **out of scope** here.

**Consequence:** the biggest lever on identity accuracy is not the aligner — it is
whether a correct, matchable candidate exists in the corpus at all. The remediation
pipeline supplies the aligner's inputs; its failure modes *are* buckets 1–2.

### Current state: rigorous where it writes, ad hoc where it decides

Already formalized (keep, reuse):

- **`track_audio_correction`** (`web_crawler/database/schema.sql:724–747`) —
  append-only DB ledger; every successful fix writes a row with full before/after
  provenance (`old_*`/`new_*` platform/player_id/url, `axis`, `action`, `reason`,
  `source`). Writers: `ingest/corrections.py:log_correction()`, called by
  `scripts/replace_track_audio.py`, `scripts/acquire_variant.py`,
  `scripts/log_acquisition.py`. 1,267-row frozen snapshot fixture exists.
- **`core/acquisition_case.py`** — already a state machine: `CaseStatus`
  (OPEN / RESOLVED / UNRESOLVABLE / HUMAN_REVIEW), a rich extensible `ProblemClass`
  (WRONG_VERSION, MISSING_ASSET, PHANTOM_TRACK, UNRESOLVED_MANIFEST — ≈ the Rvmor
  case — COMMUNITY_TAIL, STRUCTURE, …), an `attempts[]` array with per-attempt
  verdicts, a `resolution` record, and `training` preference pairs.

The gaps (this project):

1. **The case store is only written *after* you act.** Cases materialize already
   RESOLVED (pi-side executors emit `ACQUISITION_CASE` lines when a fix runs).
   Nothing opens an **OPEN** case at *detection* time. The one status that means
   "known-broken, not yet fixed" is never used as intended.
2. **No queryable worklist.** Cases are per-set JSONL
   (`data/acquisition_cases/{set_id}.jsonl`), file-IO only, single-homed on the
   Mac. You cannot ask "what's open across the corpus?" Detection
   (`scripts/scan_wrong_versions.py`) dumps a CSV once and forgets — the "106
   suspects, ~20 fixed" gap lives nowhere and is re-discovered on every scan.
3. **No arm ties a resolution back to the metric.** Nothing re-scores to confirm a
   fix moved identity accuracy. "Fixed" is asserted, not proven — and a downloaded
   file with no fingerprint/MERT is *invisible to the aligner* (bucket 1b), so
   "downloaded" ≠ "matchable".

## Goals

- One **persistent, queryable, deduped worklist** of broken/suspect tracks, each
  attributed to a root-cause `ProblemClass` and carrying an estimated
  metric-impact, that you can burn down and always answer "what's left?"
- Cases are **opened at detection time** (in OPEN), from two co-equal sources.
- A case closes only when the fix is **incorporated in matchable form** *and*
  **re-scoring confirms** the GT span moved unmatched → matched.
- Reuse the existing `acquisition_case.py` state machine and the
  `track_audio_correction` audit tail unchanged in spirit.

## Non-goals

- **Full autonomy** (the self-running detect→decide→act→verify loop). Deferred:
  with n=2 GT sets, human-in-the-loop verification is correct. Automate later.
- Fixing cross-set slippage (bucket 4) — that's a model/GT issue, not acquisition.
- Replacing `track_audio_correction` or GT — those stay authoritative; a case is a
  *trace/worklist entry*, never an override.

## Prior art: Mcity Data Engine (arXiv 2504.21614)

The design borrows the data-engine loop (the Tesla-style closed loop, academic
open-source variant). The mechanisms that shaped this spec:

- **Failures drive selection.** Deployment failures set acquisition priorities →
  our worklist is *downstream of the aligner residual*, not a standalone scan.
- **Long-tail / rare-class bias.** Don't collect uniformly; oversample what's
  underrepresented and hurting you → **prioritize cases by metric impact**.
- **Central registry + dedup + root-cause metadata.** Prevents redundant labeling
  → our deduped worklist tagged with `ProblemClass`; the cure for "106
  re-discovered every scan".
- **Open-vocabulary classes.** No fixed class list → `ProblemClass` stays
  extensible; new failure modes get named, not jammed into an existing tag.
- **Metric-driven closure.** "Handled" = per-scenario metric hits threshold *and
  holds*, not "data collected" → **the scorer-verification gate is the definition
  of done**, not a nice-to-have.

Deliberately NOT taken: full loop automation (their principle "manual bottlenecks
are the limiter") — premature at current GT scale.

## Architecture

```
  ┌─────────────── case sources (co-equal) ───────────────┐
  │  A. residual-driven: aligner scorer nominates          │
  │     - GT spans never matched by any predicted span     │
  │     - low-margin / abstained identity calls            │
  │  B. manual scan: scripts/scan_wrong_versions.py        │
  │     + human files a case by hand                       │
  └───────────────────────┬────────────────────────────────┘
                          │  open_case()  (dedup by case key)
                          ▼
              ┌───────────────────────────┐
              │  acquisition worklist      │  persistent, queryable, global
              │  (OPEN cases, ProblemClass,│  ← promoted from per-set JSONL
              │   estimated metric impact) │
              └───────────┬───────────────┘
                          │  prioritize by impact, burn down
                          ▼
              ┌───────────────────────────┐
              │  fixer                     │  replace_track_audio.py / acquire_variant.py
              │  1. acquire candidate      │  + rescue paths
              │  2. INCORPORATE (gate 1):  │  row + is_reference + fingerprint + MERT
              │     matchable or not done  │
              └───────────┬───────────────┘
                          │  writes track_audio_correction (audit tail, unchanged)
                          ▼
              ┌───────────────────────────┐
              │  verifier (gate 2)         │  re-run set scorer; did the GT span
              │  span moved unmatched→match│  move? → RESOLVED. else stays OPEN.
              └───────────┬───────────────┘
                          │  can't resolve →
                          ▼
              HUMAN_REVIEW / UNRESOLVABLE  (escalate the thin hard slice to John)
```

## Components

### 1. Worklist store (the one new persistent thing)

Promote the case store from per-set JSONL to a **globally queryable** form. The
`acquisition_case.py` dataclass stays the in-memory record; the store gains a
queryable index so "all OPEN cases, by ProblemClass, by impact" is a single query.

**Decision: back the worklist with a DB table** (`acquisition_case`), one row per
case keyed by `case_id` (`{set_id}:{slot_label}:{layer_role}`), columns mirroring
the dataclass head (`status`, `problem_classes`, `impact`, `set_id`, timestamps)
plus a JSON blob for the full trace (attempts/resolution/training). Rationale: the
worklist must be joined against GT and scorer output and deduped globally — JSONL
per-set can't do that. The pi→Mac `ACQUISITION_CASE` emit transport is unchanged;
the Mac-side sink upserts into the table instead of appending JSONL.

**Trade-off named:** this is a schema addition + a one-time migration of existing
JSONL cases. Cost is real but small (the dataclass already defines the shape). The
alternative (an index file over JSONL) is cheaper to build but re-implements a
database badly; rejected.

**New field:** `impact` (estimated metric cost, e.g. "1 guaranteed identity miss
on set X") — used for prioritization. Small; append per the module's "add fields
as real cases demand" rule.

### 2. Case sources → `open_case()`

A find-or-create seam (extends the existing `record_attempt()` at
`acquisition_case.py:360`) that opens a case in OPEN if none exists, deduped by
`case_id`. Two callers:

- **Residual-driven:** a new hook reads the scorer's per-GT-recording match status
  (the same signal behind `FINDINGS.md:116` "11 never matched") and opens/updates
  a case per unmatched or low-margin GT span, tagging the likely `ProblemClass`.
- **Manual scan:** `scripts/scan_wrong_versions.py` gains a `--open-cases` mode
  that writes its suspects into the worklist (deduped) instead of only a CSV; plus
  a tiny `open-case` CLI for filing one by hand.

Dedup is the anti-"106 re-discovered" guarantee: a `case_id` already present
(OPEN, RESOLVED, or UNRESOLVABLE) is updated, never duplicated.

### 3. Fixer + incorporation gate (gate 1)

Existing fixers (`replace_track_audio.py`, `acquire_variant.py`, rescue paths) are
extended so that "resolved" requires **matchable incorporation**, not just a
download:

- `track_audio` row with correct axis + `is_reference` set, **and**
- downstream features the identity channel needs — **fingerprint + MERT** —
  actually computed for the new row.

Today the feature recompute is a separate manual step; the work is *chaining
incorporation onto the fix*. Until gate 1 passes, the case cannot leave OPEN. The
`track_audio_correction` write is unchanged (still the audit tail).

### 4. Verifier + metric gate (gate 2) — the definition of done

A case flips OPEN → RESOLVED only when a fresh scorer run on the case's set shows
its GT span moved unmatched → matched (or identity call corrected). This is the
Mcity "metric-driven closure" bar. Mechanism: re-run the set scorer, diff the
span's match status against the pre-fix baseline stored on the case.

If the fix incorporated cleanly (gate 1) but the span still doesn't match → the
case stays OPEN with the failed attempt recorded (the fix was wrong/insufficient).

### 5. Escalation

Cases the fixer can't resolve (no acquirable asset, ambiguous ears) route to
`UNRESOLVABLE` / `HUMAN_REVIEW` — the thin genuinely-hard slice (bucket 3) plus
phantom/mix-only. These surface as a short human queue, not silent drops.

## Data flow (one case, happy path)

1. Aligner scores BB12 → GT recording R has no matching predicted span.
2. `open_case()` creates `case_id = 1fsnxchk:<slot>:<role>`, status OPEN,
   `problem_classes=(MISSING_ASSET,)`, `impact="1 identity miss @ BB12"`,
   baseline "span unmatched" stored.
3. Prioritized into the worklist (high impact → top).
4. Fixer acquires the correct candidate; gate 1 incorporates it (row +
   is_reference + fingerprint + MERT). `track_audio_correction` row written.
5. Verifier re-scores BB12; span R now matched → case RESOLVED, `resolution`
   records the winning `track_audio_id`.
6. Worklist "OPEN by impact" query shrinks by one; never re-surfaces (deduped).

## Error handling

- **Fix incorporated but metric didn't move** → stays OPEN, attempt logged REJECT;
  no false "done".
- **Re-score unavailable / set not scorable** → case parks in a `PENDING_VERIFY`
  sub-state (or stays OPEN with a note); never auto-closes on assertion alone.
- **Dedup race** (two sources open the same case) → `case_id` upsert is idempotent;
  last-writer merges `problem_classes`.
- **Feature compute fails** (fingerprint/MERT) → gate 1 fails; case stays OPEN,
  surfaced as "acquired-but-not-matchable" (distinguishes bucket 1b explicitly).

## Testing

- **Unit:** `open_case()` dedup (same `case_id` twice → one case, merged classes);
  gate-1 predicate (row present but no MERT → not matchable); gate-2 diff logic
  (baseline unmatched → matched flips to RESOLVED; still unmatched → stays OPEN).
- **Fixture-backed:** replay a known BB12 unmatched recording through
  open→fix→verify against a recorded scorer output; assert RESOLVED only after the
  span moves.
- **Migration test:** existing `data/acquisition_cases/*.jsonl` loads into the new
  table without loss (round-trip the 1,267-row snapshot shape).
- **Regression:** `scan_wrong_versions.py --open-cases` on a fixture opens N cases,
  re-run opens 0 new (dedup holds).

## Phasing

- **Phase 1 — worklist + sources + gate 1:** DB-backed worklist, `open_case()`,
  both sources wiring, incorporation gate chained onto the fixers. Delivers the
  "stop losing the list / stop re-discovering" win and "matchable, not just
  downloaded".
- **Phase 2 — gate 2 (metric closure):** scorer-verification arm; case closes on
  re-score. Delivers "prove it moved the metric".
- **Deferred — autonomy:** the self-running loop; only after Phases 1–2 earn trust.

## Open questions (resolve in planning)

- Exact home of the residual→case hook (in the scorer, or a thin adapter that reads
  its per-GT-recording output).
- Whether `impact` is a free-text string or a small structured estimate (count of
  affected spans × sets).
- Where the DB table lives given the pi-canonical / Mac-local split (cases are
  Mac-single-homed today; keep that or promote to canonical DB?).
