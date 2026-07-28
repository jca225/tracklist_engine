# AGENTS.md — operating agreement for agents in this repo

One page on purpose. If it grows into a binder it stops being read. Rules here are
enforced by machines where possible (the gate) and by convention otherwise. This
is the *how we work* layer; the *what the code is* layer lives in
[CLAUDE.md](CLAUDE.md) and the per-module `CLAUDE.md` files.

## 1. Isolate — one agent, one worktree
- Do substantive work in a **git worktree** (`.claude/worktrees/<name>`), not the
  shared main checkout. Multiple agents in one dirty tree collide (a staged edit
  gets swept into another agent's commit — this has happened). Keep worktrees
  *inside* `.claude/worktrees/` — strays elsewhere on the filesystem are invisible
  to `braid` and to everyone else.
- **Editing files while HEAD is `main` is blocked** by the `guard-worktree`
  PreToolUse hook. That is the one hard stop; everything else here is convention.
  Branch first (`<type>/<slug>`), then edit.
- **Never edit another agent's active files.** Before starting, pull/scan for
  in-flight work; if a file is mid-edit by someone else, leave it. `make collide`
  tells you which files are already contested.
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
- Keep a branch under **10 commits ahead of `origin/main`** — the guardrails WIP
  fence trips above that. Escape hatches (`epic` in the branch name,
  `GUARDRAILS_ALLOW_BIG=1`) are for deliberately long-lived work, not for a
  branch that merely got away from you.

## 2b. Finish before starting
Measured here: landing is *fast* (median 2h at 7 commits, CI 2 min) and still
most open branches sit untouched for days. Fat branches are caused by sitting,
not by difficulty.
- **One open PR per agent at a time.** Land it before opening the next front.
- **Land smallest first** — every commit that reaches `main` makes every other
  open branch worse.
- End a work session with `make gc-branches`: it deletes local branches already
  merged into `main`, and lists idle worktrees and park candidates. Abandon
  explicitly (`braid park <branch>` keeps the commits as a tag, then
  `git worktree remove <path>`) rather than by walking away.

## 3. Numbers live in one place
- Every alignment headline number belongs **only** in
  [docs/alignment_status.md](docs/alignment_status.md) (regenerated, dated, SHA-stamped).
  Never hand-type an alignment metric into another doc, a commit message, or memory —
  cite the canonical doc.

## 4. Record dead ends
- A closed experiment goes in the **EXPERIMENTS ledger**
  (`alignment/attic/EXPERIMENTS.md`) with its verdict — so no
  one re-litigates it. Read it before proposing something that smells familiar.

## 5. Canonical state is shared and live — coordinate before mutating
- The canonical DB (`pi-storage:/mnt/storage/.../music_database.db`) and running
  loops are **live and shared**. Don't mutate the canonical DB, restart a running
  driver, or re-run a non-idempotent migration/`--apply` without coordinating.
  Code edits are safe; deploy + restart is the operator's call.
- GPU boxes: rent + tear down **only via `gpubox`** (`~/workspace/gpubox`) — its guarded teardown proves ownership before destroying, so it won't kill another agent's box. Never raw-`curl` the Vast API or ad-hoc rent.

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
