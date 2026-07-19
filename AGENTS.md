# AGENTS.md — operating agreement for agents in this repo

One page on purpose. If it grows into a binder it stops being read. Rules here are
enforced by machines where possible (the gate) and by convention otherwise. This
is the *how we work* layer; the *what the code is* layer lives in
[CLAUDE.md](CLAUDE.md) and the per-module `CLAUDE.md` files.

## 1. Isolate — one agent, one worktree
- Do substantive work in a **git worktree** (`.claude/worktrees/<name>`), not the
  shared main checkout. Multiple agents in one dirty tree collide (a staged edit
  gets swept into another agent's commit — this has happened).
- **Never edit another agent's active files.** Before starting, pull/scan for
  in-flight work; if a file is mid-edit by someone else, leave it.
- A worktree has no `venvs/` (gitignored). Symlink it so the gate can run:
  `ln -s <main-repo>/venvs venvs` (the symlink is gitignored).

## 2. Land through the gate, not around it
- Work on a **branch → PR → CI-gated merge**. Do not push unstable work to `main`.
- **`make check` must pass before every push** — guardrails (stale-name/path +
  dead-flag + the `entropy_audit` bug-class fences) + mypy + pytest. If it's red
  for reasons that aren't yours (another agent's WIP), fix nothing you don't own;
  coordinate.
- Never `--no-verify` to dodge a red gate. If a fence is wrong, fix the fence or
  raise its baseline **with justification** — don't bypass.
- Commit in logical units with conventional prefixes (`feat:`/`fix:`/`docs:`…),
  and push so pi-storage/pi-worker pick changes up via `make deploy`.

## 3. Numbers live in one place
- Every alignment headline number belongs **only** in
  [docs/alignment_status.md](docs/alignment_status.md) (regenerated, dated, SHA-stamped).
  Never hand-type an alignment metric into another doc, a commit message, or memory —
  cite the canonical doc.

## 3b. No GT / status without the release gate
- Do **not** commit a new/changed `labeling/fixtures/*_ground_truth.yaml`, write back
  to `set_ground_truth`, or regenerate `docs/alignment_status.md` from that GT unless
  `make gt-gate SET=… ALS=… YAML=…` is green.
- Gate writes a **committed** stamp under `labeling/fixtures/gt_gate_stamps/` (stage it
  with the YAML — pre-commit checks the sha) and a local cache stamp for write-back.
- Known leftover audit debt: `labeling/fixtures/gt_audit_acks.yaml`. Status regen:
  `make status-preflight` first. Escape: `--force-ungated` / blanket
  `--ack-audio-mismatches` only with an explicit debt note.

## 4. Record dead ends
- A closed experiment goes in the **EXPERIMENTS ledger**
  (`workspaces/alignment_prototype/attic/EXPERIMENTS.md`) with its verdict — so no
  one re-litigates it. Read it before proposing something that smells familiar.

## 5. Canonical state is shared and live — coordinate before mutating
- The canonical DB (`pi-storage:/mnt/storage/.../music_database.db`) and running
  loops are **live and shared**. Don't mutate the canonical DB, restart a running
  driver, or re-run a non-idempotent migration/`--apply` without coordinating.
  Code edits are safe; deploy + restart is the operator's call.
- Vast boxes: **list before you create, destroy only your own** (ownership ledger).

## 6. Keep the record current, not piled up
- End an alignment session with `/align-checkpoint` (updates the living
  state-of-record) rather than adding another dated handoff. Dead docs age out via
  `make docs-gc`.

---

## Branch protection (repo admin, one-time)
Make the gate unbypassable on `main`. **Precondition: confirm the `guardrails`
check is green on `main` first** — marking a currently-red check as required
freezes *all* merges (learned the hard way: the check had been red for days, and
requiring it briefly froze the branch). As a repo admin:
```
gh api -X PUT repos/jca225/tracklist_engine/branches/main/protection \
  -f 'required_status_checks[strict]=true' \
  -f 'required_status_checks[contexts][]=guardrails' \
  -F 'enforce_admins=true' \
  -F 'required_pull_request_reviews[required_approving_review_count]=0' \
  -F 'restrictions='
```
(Requires the `guardrails` CI check from `.github/workflows/guardrails.yml` to be
the required context. Adjust the context name to match the workflow's job.)
