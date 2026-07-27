# Branch hygiene with braid

This repo is meant to grow under **many parallel agent branches**, same problem
`braid` was built for (`~/workspace/braid`). Fold-in is landing hygiene, not
algorithm logic.

## Prerequisite

`alignment_algorithm` must be a **git repository**. It currently may not be —
run `git init` (and add a remote) before braid forecasts mean anything.

```bash
uv tool install --editable ~/workspace/braid   # once
make ready       # is THIS branch landable?
make collide     # hotspots + landing order
make land-budget # over-budget branches
make land-verify # after merge/rebase
```

## Agent loop

| Moment | Command | Why |
|--------|---------|-----|
| Before starting a worktree | `make collide` | Avoid files already contested |
| While finishing | `braid ready` / `make ready` | Exit 0 only if landable |
| Branch too fat | `braid slice` | Largest clean prefix to land first |
| After merge | `.githooks/post-merge` | Advisory parse check |
| After rebase | `make land-verify` | No post-rebase hook in git |
| Idle branches | `braid stale` / `park` | Finish or park; do not abandon |

**Finish before start.** Fat branches here will come from sitting while `main`
moves, not from slow CI. Land smallest first.

## What braid does vs `make check`

- `braid ready` / `verify` — landability + **parse** (and collision forecast).
- `make check` — behavior: clippy, tests, ruff, pyright, pytest.

Both matter. Declaring victory on `ready` without `make check` is incomplete.

## Rust note

`braid` tracks `.rs` files for **collision hotspots**, but its syntax verifier
does not yet parse Rust (Python/JSON/JS/Go only). `make land-verify` therefore
runs `braid verify` **and** `cargo check --workspace`.

Optional later: extend `~/workspace/braid` `syntax.py` with a crate-aware Rust
check; until then cargo is the Rust gate.

## Hooks (once git exists)

```bash
git config core.hooksPath .githooks
# optional: refuse push unless landable
printf '#!/bin/sh\nexec braid ready\n' > .git/hooks/pre-push && chmod +x .git/hooks/pre-push
```

`.githooks/post-merge` is advisory and never blocks.
