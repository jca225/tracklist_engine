# Weekly Audit Process — Design

**Date:** 2026-07-16
**Status:** Approved (design), pending implementation plan
**Author:** alignment session (co-train branch)

## Problem

The repo has a mature *mechanical* drift layer — `make check` (`guardrails.py` +
`guardrails_ratchet.json`), `make docs-gc`, `make check-corpus` (deployed daily
corpus-integrity watcher), `make check-inventory`, `make audit-gt`, plus CI
(`guardrails.yml`). These catch rename drift, stale module names, dead docs, and
corpus integrity.

They do **not** catch *semantic* bugs — the class that produced the 2026-07-16
read-only ingest/pull/encoding audit (missing subprocess timeouts, silent-failure
loops, poison-pill retries, mojibake at the SSH/sqlite Latin-1 boundary,
shell-interpolated SQL). No linter finds these; it takes a reasoning pass.

We want a **recurring (weekly) audit** that (a) runs a semantic bug sweep and
(b) tracks entropy / code-smell metrics over time, surfacing a blocker-first
worklist without crying wolf on known/accepted items.

## Key insight

The semantic and entropy layers are not separate. **Several of the audit's bug
classes are grep-able counts** and can be encoded as *ratchets* in the existing
`guardrails_ratchet.json` mechanism:

- `subprocess.run` / `check_call` / `check_output` without `timeout=` (the B1 hang class)
- `text=True` without explicit `encoding=` on a subprocess boundary (the A2 mojibake class)
- `except: pass` and bare `except:` (the silent-failure class)
- SQL passed as a shell-interpolated f-string over SSH (the A3 injection/quoting class)

Once the current bugs are fixed, the ratchet pins the count at the new floor:
`make check` fails the instant anyone reintroduces the pattern. **The deterministic
layer becomes a regression fence around exactly the bugs the semantic layer finds.**

## Architecture — three components

### 1. `scripts/entropy_audit.py` — deterministic metrics

Reuses the `guardrails.py` ratchet philosophy (`guardrails_ratchet.json` = max
allowed counts; new occurrence fails, removal prompts a baseline bump). Emits:

- a machine snapshot: `docs/audits/metrics/<date>.json`
- a human report section (folded into the weekly report)

**Bug-class fences** (ratcheted, wired into `make check` — see Component 3's
`--check` mode):
| Metric | Pattern (scoped to which roots) | Rationale |
|---|---|---|
| `subprocess_no_timeout` | `subprocess.(run\|check_call\|check_output\|Popen)(` lacking `timeout=` in same call, under `scripts/`, `ingest/`, `analysis/` | B1 hang class |
| `text_true_no_encoding` | `text=True` without `encoding=` in the same call | A2 mojibake class |
| `bare_except` | `except:` and `except Exception: pass` | silent-failure class |
| `shell_sql_fstring` | `sqlite3 ... "{...}"` shell-interpolated SQL | A3 injection class |

**Entropy metrics** (tracked, warn-only at first; ratchet after baseline settles):
| Metric | Definition |
|---|---|
| `max_file_loc` / `files_over_600` | source files exceeding ~600 LOC |
| `todo_debt` | count of `TODO` / `FIXME` / `HACK` / `XXX` |
| `test_module_ratio` | test files ÷ non-test python modules |

Scope: repo python excluding `venvs/`, `cue-detr/`, vendored trees (reuse
`guardrails.py`'s existing exclusion list). Grep-based, no new heavy deps
(no radon/AST unless a metric demands it — YAGNI).

**Modes:**
- `entropy_audit.py --snapshot` → write JSON + print report (weekly use)
- `entropy_audit.py --check` → compare bug-class fences against
  `guardrails_ratchet.json`, exit nonzero on regression (push/CI use, folded into
  `make check`)
- `entropy_audit.py --bump` → rewrite baseline after a legitimate reduction

### 2. Semantic bug-audit agent — the weekly reasoning pass

A fixed prompt (the format the 2026-07-16 report already nails: findings grouped
by class, blocker-first within class, a "confirmed present vs latent" index, a
priority worklist) run over a defined high-risk scope.

- **Scope (v1, fixed):** the loop/driver scripts (`scripts/{vast_loop,
  mac_analyze_loop,set_mert_backfill_loop,loop_prefetch,acquire_variant,
  ingest_stem_url,replace_stem_audio,replace_track_audio,mac_push_acquire}.py`),
  `ingest/` (incl. `main_retry.py` + rescue paths flagged "not fully covered" in
  the first audit), `analysis/{pipeline,persistence,vast_worker}.py`, `core/db.py`.
- **Waivers:** reads `docs/audits/waivers.yaml` — a list of accepted/known-and-won't-fix
  findings with `id`, `path`, `rationale`, `added` — and excludes them from the
  report so the weekly run doesn't re-flag them.
- **Output:** the graded report, appended to the dated audit doc.

The agent is invoked by the cloud routine (Component 3), not hand-run — but the
prompt lives in-repo (`scripts/audit/semantic_audit_prompt.md`) so it's versioned
and improvable.

### 3. `/schedule` weekly cloud routine — wiring

A weekly cloud routine (Mon morning) that:

1. `python scripts/entropy_audit.py --snapshot` → metrics JSON + report section
2. run the semantic bug-audit agent (reads `waivers.yaml`) over the fixed scope
3. write `docs/audits/YYYY-MM-DD-audit.md` (entropy section + semantic findings +
   blocker-first worklist)
4. **dedup delivery:** open/update **one rolling GitHub issue**
   ("Weekly audit — open findings"), only when blocker/high findings exist that
   are not in `waivers.yaml`; edit the existing issue body rather than opening a
   new issue each week
5. commit the dated report + metrics JSON

Runs in the cloud, so it fires whether or not the Mac is on. Findings are advisory
— the routine never edits code, only reports.

## Data flow

```
weekly cloud routine
  ├─ entropy_audit.py --snapshot ──> docs/audits/metrics/<date>.json
  │                                  (bug-class counts + entropy metrics)
  ├─ semantic agent (scope + waivers.yaml) ──> graded findings
  ├─ compose ──> docs/audits/<date>-audit.md
  ├─ if new blocker/high ──> gh issue edit (one rolling issue)
  └─ git commit report + metrics

push / CI (unchanged cadence, new fence)
  └─ make check ──> guardrails.py + entropy_audit.py --check
                    (fails if a bug-class count regresses above baseline)
```

## Error handling / failure modes

- **Routine step fails** (e.g. agent errors): write a partial report noting the
  failed step; do not open a false "all clear" issue. Mirrors the C1 lesson from
  the source audit — a failed audit must not read as a clean audit.
- **Ratchet false positive** (legitimate pattern flagged): add a `# noqa: audit-<metric>`
  inline marker (honored by `entropy_audit.py`) or a waiver entry — same escape
  hatch style as existing guardrails.
- **Issue spam:** guarded by the single-rolling-issue + waivers design.

## Testing

- `tests/test_entropy_audit.py`: fixture files exercising each bug-class pattern
  (positive + negative — e.g. a `subprocess.run(..., timeout=30)` must NOT count),
  `--check` exits nonzero when a count exceeds baseline, `--snapshot` emits valid
  JSON, waivers/`noqa` suppression works.
- Manual: run `--snapshot` on the current tree, confirm the bug-class counts match
  the known offenders from the 2026-07-16 audit (a cross-check that the detector
  actually catches the real bugs).

## Sequencing (companion bug-fix track)

The bug fixes and the auditor are coupled: fixing the bugs sets the ratchet floor.
Order:

1. **Fix the blocker/high bugs** from the 2026-07-16 audit (separate worklist,
   scoped with the user) — establishes the low-water mark.
2. **Build `entropy_audit.py`** with bug-class ratchets pinned at the post-fix floor.
3. **Wire `--check` into `make check`** so regressions fail at push time.
4. **Author the semantic-audit prompt + waivers.yaml.**
5. **Create the `/schedule` weekly routine.**

## Out of scope (YAGNI)

- Architectural-drift and cross-machine-health audits (already covered by
  guardrails / `make check-corpus` / `/pulse`; user deferred).
- AST/complexity analysis beyond grep counts (add only if a metric proves it needs it).
- Auto-fixing findings (the routine reports; humans/sessions fix).
```
