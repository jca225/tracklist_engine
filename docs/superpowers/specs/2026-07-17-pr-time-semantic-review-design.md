# PR-time Semantic Review — Design

**Date:** 2026-07-17
**Status:** Approved (design), pending implementation plan
**Author:** alignment session (post cotrain-grammar-coverage)

## Problem

The repo has two of the three drift layers it wants (numbered by where they sit
on the fast→slow axis; layer 2, the middle, is the gap this spec fills):

- **Layer 1 — deterministic fence** — `entropy_audit.py --check` inside
  `make check` / `guardrails.yml`, runs on **every push**, hard-fails CI when a
  known bug class (missing subprocess/rsync `timeout=`, `text=True` without
  `encoding=`, bare `except`) regresses above baseline.
- **Layer 3 — periodic semantic sweep** — the `weekly-bug-audit` cloud routine
  (`trig_01Ng8C8Tkm1DPYAckRsJLYFC`, now Mon/Wed/Fri 03:00 ET), a full-repo
  reasoning pass that writes rolling GitHub issue #8.

Missing is **layer 2**, the middle: catching a *new* semantic bug **at the moment
it is written**, in the PR that introduces it, with the diff in full context —
instead of up to two days later in a full-repo sweep that mostly re-scans
unchanged code. This is where detection latency is lowest and the fix is
cheapest (the author still has the change in their head).

The cron routine cannot fill this role: the routine system is **cron-only** and
cannot trigger on push/PR events. A separate, event-driven mechanism is
required.

## Goals / non-goals

**Goals**
- Review the **diff of each PR** through the semantic bug-family lens the
  founding 2026-07-16 audit established, plus general correctness.
- Event-driven (fires on PR open), fully automated, no human ceremony.
- **Advisory** — informs via PR comments; never blocks merge.
- Quiet when there is nothing worth saying (no "looks good" noise).
- Cost-bounded and billed against the existing Claude subscription (no new
  billing account).

**Non-goals (YAGNI)**
- No blocking / required-check behavior (the deterministic fence is the hard
  gate; an AI reviewer with false positives must not become a merge blocker
  that trains bypass behavior).
- No waiver file — advisory + quiet-when-clean makes suppression unnecessary at
  this scale.
- No `@claude` interactive mode.
- No separate Anthropic API billing (uses a subscription OAuth token).
- No re-review on every push (see Cost levers).

## Three-layer model (where this fits)

| Layer | Mechanism | Cadence | Output surface | Gate? |
|---|---|---|---|---|
| Deterministic fence | `entropy_audit --check` in `make check` / CI | every push | CI pass/fail | **hard gate** |
| **Semantic PR review (new)** | GitHub Action + Claude | every PR (on open) | inline PR comments | advisory |
| Semantic sweep | `weekly-bug-audit` cron routine | Mon/Wed/Fri | rolling issue #8 | advisory |

Ownership is disjoint: the fence blocks known classes deterministically; the PR
review catches new semantic bugs in the diff at write-time (PR comments); the
cron catches cross-cutting drift across the whole repo (issue #8). No two layers
write the same surface.

## Architecture

A single new workflow file, **`.github/workflows/pr-review.yml`**, separate from
`guardrails.yml` so the review can never affect the required `guardrails` check.

### Trigger

```yaml
on:
  pull_request:
    types: [opened, ready_for_review]
    paths:
      - '**.py'
```

- **`opened` + `ready_for_review`** only — **not** `synchronize`. One review per
  PR at open (and when a draft is marked ready), not one per push. This is the
  primary cost lever (~2–3× fewer runs than including `synchronize`) and is the
  right granularity for an advisory reviewer: the feedback lands when the PR is
  first put up for review.
- **`paths: ['**.py']`** — docs-only / spec-only / non-Python PRs don't run the
  Action at all.
- Draft PRs are skipped implicitly (they don't fire `opened` as
  ready-for-review; the job also guards on `github.event.pull_request.draft ==
  false`).

### Action + auth

- Official **`anthropics/claude-code-action`**.
- Auth via repo secret **`CLAUDE_CODE_OAUTH_TOKEN`** — generated once by the
  maintainer with `claude setup-token`, bills against the existing Claude
  subscription. No `ANTHROPIC_API_KEY`, no separate billing.
- Model **Sonnet** (`claude-sonnet-4-6`) — matches the cron auditor and is the
  correct cost tier for advisory review. (Opus would be ~5× the cost for
  marginal benefit on diff-scoped review.)
- Least-privilege job permissions: `contents: read`, `pull-requests: write`
  (to post review comments), `issues: write` only if comments are posted as
  issue comments rather than review comments.

### Review prompt (versioned in the workflow)

Instructs the reviewer to, over the **full PR diff** (all changed paths):

1. Review through the semantic bug-family lens from the 2026-07-16 audit:
   - **(A)** encoding / mojibake at SSH / sqlite Latin-1 boundaries
   - **(B)** missing subprocess / rsync `timeout=` and no-integrity-check
     transfers
   - **(C)** silent-failure loops (poison-pill retries, dishonest exit codes,
     `except: pass` that swallows real errors)
   plus general correctness bugs in the changed lines.
2. Grade findings Blocker / High / Medium / Low, blocker-first.
3. **Post nothing when there is no High/Medium (or worse) finding** —
   quiet-when-clean. No summary comment on clean PRs.
4. Anchor each comment to the specific changed line.
5. Skip anything the deterministic fence already covers (don't restate a missing
   `timeout=` the CI check will also flag).

## Data flow

```
PR opened / marked ready (touches a .py file, not a draft)
  └─ .github/workflows/pr-review.yml
       └─ anthropics/claude-code-action (Sonnet, CLAUDE_CODE_OAUTH_TOKEN)
            ├─ fetch PR diff + surrounding file context
            ├─ reason over diff (bug families + correctness)
            └─ if High/Medium+ findings: post inline PR review comments
               else: post nothing
```

Merge is never gated on this job. `guardrails.yml` remains the only required
check.

## Error handling / failure modes

- **Token expired / API unavailable / usage limit hit:** the review job shows a
  **red (non-required) check** on the PR. Visible signal, non-blocking. The job
  deliberately does **not** post an "I failed" comment — that would break
  quiet-when-clean and add noise. A failed review can never read as a clean
  review because the red check is on the PR.
- **Large diff:** cost scales with diff size but a single PR is bounded; Sonnet +
  the `.py` path filter keep it in the sub-$2 range even for big PRs.
- **False positive finding:** it's a comment, not a gate — the author reads it,
  disagrees, moves on. No suppression machinery needed at this scale.

## Cost

Billed against the Claude subscription (not new dollars); equivalent-API-cost
figures for magnitude only.

- **Per review (Sonnet):** ~$0.20–0.60 typical (normal diff), ~$1–2 for a large
  multi-file PR.
- **Per month at current velocity** with the two cost levers (no `synchronize`,
  `.py` path filter): roughly **~$15–25/month equivalent** — negligible against
  a Max plan, unlikely to compete meaningfully with interactive use.
- Without the levers (reviewing every push, all PRs) it would be ~$40–60/month
  equivalent. The levers are baked into the design; re-adding `synchronize` later
  is a one-line change if per-push re-review is wanted.

## Setup (one-time, maintainer-run)

Requires the maintainer's auth, so it is not automatable by an agent:

```bash
claude setup-token                              # prints a long-lived OAuth token
gh secret set CLAUDE_CODE_OAUTH_TOKEN \
  --repo jca225/tracklist_engine                # paste the token
```

After the secret exists, the workflow is live on the next PR.

## Testing

- **Positive:** open a throwaway PR that plants a deliberate bug in a `.py` file
  (e.g. `subprocess.run(["ssh", ...], text=True)` with no `timeout=`/`encoding=`)
  → confirm the Action posts an inline comment flagging it.
- **Quiet-when-clean:** open a PR with a trivially correct `.py` change → confirm
  the Action runs and posts **nothing**.
- **Path filter:** open a docs-only PR → confirm the workflow does not run.
- **Non-blocking:** confirm a PR can merge while/after the review job is red
  (required check stays `guardrails` only).

## Sequencing

1. Maintainer adds `CLAUDE_CODE_OAUTH_TOKEN` secret (setup step above).
2. Add `.github/workflows/pr-review.yml` (branch → PR → merge through the gate).
3. Validate with the three test PRs above.
4. Update `AGENTS.md` / `scripts/CLAUDE.md` (or the governance doc) with a
   one-line pointer to the new layer so the three-layer model is documented.

## Out of scope (YAGNI)

- Blocking / required-check semantics.
- Waiver / suppression file.
- `@claude` interactive follow-ups.
- Re-review on `synchronize` (deferred; one-line to enable).
- Separate API-key billing.
