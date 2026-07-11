# Handoff: Alignment Documentation Overhaul — Single Source of Truth + Reconcile

**Date:** 2026-07-11
**For:** a fresh agent, executing cold
**Owner intent:** John flagged "a lot of slippage" in the alignment docs. This
session produced direct evidence (below). The fix is structural, not cosmetic.

---

## 0. Why this exists (the slippage, with evidence)

The alignment project's "current state" is smeared across ~10 prose docs, three
`CLAUDE.md` layers, and ~80 auto-memory entries. Numbers are **hand-typed into
prose**, so they drift the moment code moves. Watched live this session:

1. **Fibers under-credited by ~20pp.** A status synthesis reported the *strict*
   trajectory numbers as the headline and dismissed fibers as a "scoring util,"
   quoting the pre-v4 SALAMI framing (P .88 / R .06) as if it were the whole
   story. Reality (verified against code + `docs/agent_handoff_fibers_20260710.md`):
   fiber-aware scoring lifts BB12 21%→45% and BB11 20%→40%; v4 (2026-07-09)
   fixed the acappella recall hole (vocal coverage 0.06–0.28 → 0.33–0.73);
   phase-cancel clone certificates are wired. The +20pp is the *finding*, not a
   footnote.
2. **Stale external framing.** The SALAMI P .88 / R .06 numbers are *current* but
   were being read as a limitation rather than by-design precision-first
   conservatism on a jam-band pessimistic floor.
3. **Numbers scattered and versioned inconsistently.** `FINDINGS.md` has
   "Re-measure 1" vs "Re-measure 2"; the race board lives in a dated handoff;
   `cotrain_loso_findings.md` holds the LOSO result; no single place says "here
   is the current truth as of <date>."

**Root cause:** no single generated source of truth. **Fix:** create one, reconcile
everything to it, and stop hand-typing metrics.

---

## 1. Deliverable

Primary: **`docs/alignment_status.md`** — one canonical, dated, living status doc
that owns every current alignment number and every method's status. All other
docs cite it or get atticked; none re-state numbers.

Secondary:
- A **corrections ledger** (section inside this handoff's PR description, or a
  short `docs/alignment_status_corrections_20260711.md`) recording every drift
  found and the corrected value, so we learn *what* went stale and *why*.
- A **recurrence guardrail**: numbers live in exactly one generated block; a rule
  added to the relevant `CLAUDE.md` that status metrics belong only in
  `docs/alignment_status.md`, regenerated from the scorer with a date + git SHA
  stamp.

**Out of scope:** changing any alignment code/behavior, re-running expensive
inference on new sets, building the ablation harness (that's a separate,
already-brainstormed effort). This is docs only — but docs *backed by
regenerated numbers*.

---

## 2. Ground rule (non-negotiable)

**Regenerate the numbers from the scorers BEFORE writing a single line of the
canonical doc. Trust no existing prose number.** The entire failure mode is prose
that drifted from code. If you copy prose forward, you have laundered stale
numbers into a nicer-looking doc and made the problem *worse* (now it looks
authoritative).

For any number you genuinely cannot regenerate locally (missing audio/features/
GT — see Risks), carry it forward **explicitly marked** `⚠ unverified — carried
from <file>, not regenerated` so the staleness is visible, not hidden.

---

## 3. Phase 0 — Build the number-truth

Run the actual scorers and capture raw output to the scratchpad. Repo root, use
`venvs/audio/bin/python`. Set ids: **BB11 = `1fsnxchk`**, **BB12 = `2nvzlh2k`**.

```bash
# Corpus failure attribution + per-span table (BB11+BB12)
make scorecard

# Per-set, strict + fiber-aware + oracle decomposition
venvs/audio/bin/python -m workspaces.alignment_prototype.score_timeline_vs_gt \
    --set-id 1fsnxchk --fibers --decompose
venvs/audio/bin/python -m workspaces.alignment_prototype.score_timeline_vs_gt \
    --set-id 2nvzlh2k --fibers --decompose

# Driver comparison board (classical vs agentic vs ml)
make race

# Oracle placement baseline (isolates placement from decode)
venvs/audio/bin/python -m workspaces.alignment_prototype.path_decode --eval \
    --feature hubert --stems acappella --fibers --workers 8
```

Verify each command actually ran to completion (not a partial/cached table).
Record: the command, the date, and the current `git rev-parse --short HEAD`.
These stamps go into the canonical doc's number block.

If a command errors on missing inputs, do NOT silently skip — log it in the
corrections ledger and fall back to the carry-with-warning rule (§2).

---

## 4. Phase 1 — Inventory the doc surface

Enumerate every artifact that states an alignment status or number. Known set
(verify + extend by grepping for `%`, `traj`, `precision`, `BB11`, `BB12`,
`accuracy` under `workspaces/alignment_prototype`, `docs/`, `eda/alignment`):

| Artifact | Role today | Likely fate |
|---|---|---|
| `workspaces/alignment_prototype/CLAUDE.md` | module guide + inline numbers | strip numbers → cite canonical |
| `eda/alignment/failure_analysis/FINDINGS.md` | failure taxonomy + re-measures | keep as deep appendix; canonical owns headline |
| `workspaces/alignment_prototype/looptrace/NOTES.md` | looptrace phase log + dead threads | keep; cite canonical for headline |
| `workspaces/alignment_prototype/cotrain_loso_findings.md` | LOSO result | fold headline into canonical; keep detail |
| `workspaces/alignment_prototype/external/fiber_validation_findings.md` | SALAMI validation | keep; canonical cites it |
| `docs/agent_handoff_fibers_20260710.md` | race board (fiber-aware) | source for canonical fiber rows |
| `docs/agent_handoff_flywheel_select_20260711.md` | flywheel gears 2–3 | keep as active-work pointer |
| `attic/EXPERIMENTS.md` | closed-experiments ledger | keep; canonical links "dead" methods here |
| `MEMORY.md` + `memory/*.md` (~80) | session memory | update stale entries (esp. fibers), point to canonical |
| root `CLAUDE.md`, `docs/architecture_north_star.md`, `docs/alignment_objective.md` | north star | add link to canonical status |

For each, note: does it state numbers? Are they consistent with Phase 0's
regenerated truth? Record every mismatch in the corrections ledger.

---

## 5. Phase 2 — Author the canonical doc (`docs/alignment_status.md`)

Structure (this is the spec — follow it):

1. **Header stamp** — "Numbers regenerated `<date>` at commit `<sha>` via the
   commands in §3." This is the anti-drift contract: a reader always knows how
   fresh the numbers are.
2. **Headline table** — per set (BB11, BB12): identity %, set-start placement
   (median / <15s / p90), per-axis trajectory **both strict and fiber-aware**
   (acappella / regular / instrumental), multiseg+loop. One row = one number
   with a known definition. State the scorer + tolerance (±2s, sec-weighted) once.
3. **The strict-vs-fiber-aware gap** — call out explicitly that fiber-aware −
   strict ≈ +20pp is the "which-instance" residual (right chorus content, wrong
   occurrence), externally precision-validated (SALAMI P .88). This is a named
   contribution, not a caveat.
4. **Method/component registry** — one row per method: `fingerprint`, `HuBERT
   stem-placement`, `MERT identity`, `chroma matched-filter`, `lyrics-align`,
   `looptrace`, `fibers (v4)`, `phase-cancel clone cert`, `agentic driver`,
   `cotrain/LOSO`, `trajectory decoder`. Columns: **status** (wired / frozen /
   in-progress / dead→attic), **role**, **current contribution/number**, **key
   finding**, **deep-doc link**. Fibers: status = wired-for-scoring, contribution
   = +20pp fiber-aware, v4 recall fix, precision-certified.
5. **Binding walls + current lever** — placement (31% of loss) + which-instance
   (45% of loss); lever = learned trajectory decoder + flywheel; LOSO proof that
   identity transfers 100% / placement does not.
6. **Phase policy** — sensor phase frozen (2026-07-09), the 3 sanctioned lanes,
   pointer to `attic/EXPERIMENTS.md` for dead ends.
7. **Paper framing decision (2026-07-11)** — record that "99% accuracy" is an
   **aspirational north star**, not the paper's empirical claim; the paper reports
   real numbers + the methodology, and positions 99% (via trajectory decoder +
   flywheel) as the design target. (Captured here so it stops being re-litigated.)
8. **Appendix: deep-doc index** — FINDINGS / NOTES / cotrain / fiber-validation,
   each described in one line. Canonical owns headline numbers; these own detail.

Every number in this doc must trace to a Phase 0 command or a
carry-with-warning. No orphan numbers.

---

## 6. Phase 3 — Reconcile everything else

For each inventoried artifact, apply exactly one:

- **Correct** — the doc's numbers are wrong/stale and the doc should keep owning
  its topic → fix in place to match Phase 0, add a one-line "headline numbers:
  see `docs/alignment_status.md`."
- **Redirect** — the doc re-states headline numbers it shouldn't own → strip the
  numbers, replace with a pointer to canonical. (Deep docs keep their *detail*
  numbers but not the headline summary.)
- **Attic** — the doc is a superseded handoff/status → move under `attic/` (or the
  repo's existing attic convention) with a one-line tombstone.

**Memory pass:** update stale `memory/*.md` entries and their `MEMORY.md`
pointers. Priority: the fibers entry (`project_fibers.md`) must reflect v4 +
the +20pp fiber-aware result, not the pre-v4 framing. Any memory entry naming a
number should either be corrected or point to canonical. Do NOT delete memory
wholesale — correct in place per the memory rules in the global instructions.

---

## 7. Phase 4 — Prevent recurrence

Lightweight, no big build:

1. **One-place rule.** Add to `workspaces/alignment_prototype/CLAUDE.md` (and a
   one-liner in root `CLAUDE.md`): *"Alignment status numbers live only in
   `docs/alignment_status.md`, regenerated from the scorers (§3 commands) with a
   date+SHA stamp. Do not hand-type metrics into other docs or memory — cite the
   canonical doc."*
2. **Regeneration ritual.** Document the exact §3 command block inside
   `docs/alignment_status.md` itself, so refreshing it is copy-paste, and the
   staleness of the stamp is self-evident.
3. *(Optional, recommended, flag to John before building)* a `make status` target
   that runs the §3 scorers and prints the headline block, so the canonical doc's
   table can be regenerated rather than retyped. Only build if John green-lights —
   it edges into the "stop-the-drift mechanism" option he did not pick as primary.

---

## 8. Known drift to seed the work (from this session)

Start the corrections ledger with these, already established:

- **Fibers:** strict-vs-fiber-aware +20pp (BB12 21→45, BB11 20→40); v4 recall fix
  (0.06–0.28 → 0.33–0.73 vocal coverage, 2026-07-09); phase-cancel clone cert
  wired (BB12 14/414, BB11 2/234 pairs). Old framing called them a scoring util.
- **SALAMI P .88 / R .06:** current, but is precision-first-by-design on a
  jam-band pessimistic floor — not a limitation verdict.
- **FINDINGS.md re-measures:** confirm which re-measure is current; canonical
  should carry only the latest, dated.
- **Paper framing:** 99% = aspirational north star (decided 2026-07-11), not an
  empirical claim.

---

## 9. Definition of done

- [ ] `docs/alignment_status.md` exists, every number traces to a §3 regeneration
      or a visible carry-with-warning, header carries date+SHA stamp.
- [ ] `make scorecard`, both `score_timeline_vs_gt` runs, `make race`, and the
      oracle baseline were actually executed this session (or blockers logged).
- [ ] Every inventoried doc is Corrected / Redirected / Atticked — none silently
      left with contradicting numbers.
- [ ] Fibers memory entry + `MEMORY.md` pointer corrected to v4 reality.
- [ ] One-place rule added to `CLAUDE.md`.
- [ ] Corrections ledger lists every drift found + fix.
- [ ] Committed in reviewable units (canonical doc; reconcile pass; memory pass;
      guardrail) — not one blob. Push per repo git-workflow rule.

---

## 10. Risks & gotchas

- **Scorers may need data not on the dev Mac.** The canonical DB copy is stale;
  audio/features/stems may live only on pi-storage. If a §3 command fails on
  missing inputs, log it, try the pi (`make ssh-storage` / the pi-storage-query
  skill), and if still blocked, carry-with-warning rather than fabricate. Being
  honest about "couldn't regenerate" is the whole point.
- **Don't restructure the deep docs' internals.** FINDINGS/NOTES are working
  logs; strip only their *headline summary* numbers, keep their analysis.
- **Memory edits follow the global memory rules** (one fact per file, correct in
  place, update `MEMORY.md` pointer). Don't mass-delete.
- **Two GT sets only.** The canonical doc must not imply cross-set
  generalization from n=2; report LOSO as the generalization evidence, not a CI.
- **Stay in docs.** If you find a code bug while reconciling, note it in the
  ledger and flag it — do not fix code under this handoff.
```
