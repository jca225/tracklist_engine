---
name: cluster-deploy
description: Deploy code to pi-storage and pi-worker, check service status, restart services, or tail logs across the Tracklist Engine cluster. Wraps the Makefile targets (make deploy, make status, make queue, make restart-jobqueue, make restart-retry, make logs-*) with knowledge of which service runs where, what each one does, and when restart is safe vs disruptive. Use when the user wants to push code changes to the cluster, check what's running, restart a service, or tail logs. Triggers on phrases like "deploy to the pis", "deploy this", "make deploy", "check cluster status", "is the scraper running", "restart the jobqueue", "tail the retry logs", "stop the scraper".
---

# Cluster Deploy & Ops

The cluster is two Linux ARM Pis reachable via Tailscale MagicDNS, plus the local Mac:

| Host | Role | Key services |
|---|---|---|
| `pi-storage` | canonical state + scraper + CPU analysis | `tracklist-jobqueue.service` (FastAPI), `tracklist-scraper.service` (1001tl scraper) |
| `pi-worker` | AJAX retry drain + spare CPU | `tracklist-ajax-retry.service` |
| Mac (here) | dev driver + alignment + secondary analysis worker | none persistent |

All ops go through the [Makefile](Makefile) at the repo root. Run `make help` to see the full target list.

## Deploy

```bash
make deploy
```

This:
1. SSHes to pi-storage: `git pull --ff-only origin main && pip install -q -r requirements.txt`
2. SSHes to pi-worker: same.

Both Pis run code from `~/tracklist_engine` cloned from `origin/main`. **Services are NOT automatically restarted** — if your change modifies code that a running service has already loaded, you must restart that service yourself (see below). This is deliberate so a `make deploy` doesn't interrupt a long-running retry drain or scraper job.

Deploy a single host if you only changed code that runs there:
```bash
make deploy-storage   # pi-storage only
make deploy-worker    # pi-worker only
```

**Pre-deploy checklist:**
- The change must be committed AND pushed to `origin/main`. The Pis `git pull` from there — uncommitted local changes won't propagate.
- If unsure, `git status && git log origin/main..HEAD` first.

## Check status

```bash
make status
```

Reports systemd state of all three services + the `scrape_failures` queue depth. Use this whenever the user asks "what's running" or "is X up".

For just the failure queue:
```bash
make queue
```

## Restart services

These are deliberate operations — they interrupt the running service. Use only when:

- **`make restart-jobqueue`** — bounce the FastAPI server on pi-storage. Safe; restarts in ~2s. Required after deploying code that the jobqueue uses.

- **`make restart-retry`** — stops + starts the AJAX retry drain on pi-worker. **Disruptive**: kills the in-flight retry pass. The target prints a warning and sleeps 3s before executing — read the warning. Only do this when you actually want to pick up new code or recover a stuck drain.

- **`make start-scraper`** / **`make stop-scraper`** — controls the full-corpus scraper. Starting kicks off a 1001tracklists crawl (heavy, hours long); stopping interrupts it. Only do this when you actually mean to.

## Tail logs

```bash
make logs-jobqueue   # FastAPI access + app log on pi-storage
make logs-scraper    # 1001tl scraper output on pi-storage
make logs-retry      # AJAX retry drain on pi-worker
```

These follow live (`journalctl -f`). Ctrl-C to exit. Use them when:
- Debugging "is the service actually doing anything"
- Watching a deploy take effect after restart
- Investigating an error reported by `make status` or by failure counts

## Quick shells

```bash
make ssh-storage     # ssh pi-storage
make ssh-worker      # ssh pi-worker
```

Use directly when you need to run ad-hoc commands not covered by the Makefile (file inspection, manual DB queries, checking disk space, etc.).

## Typical deploy flow

For a routine code change:

```bash
# 1. Commit + push (Pis pull from origin/main)
git add <files> && git commit -m "..." && git push

# 2. Deploy
make deploy

# 3. Restart anything that has the changed code loaded in memory
make restart-jobqueue   # if you touched FastAPI handlers / shared libs

# 4. Verify
make status
make logs-jobqueue       # briefly, to confirm it came up cleanly
```

For a hot-path change to scraper logic, the scraper has to be restarted too (`make stop-scraper && make start-scraper`) — but only do that if the scrape can safely interrupt.

## Anti-patterns

- ❌ Running `make deploy` without pushing first. The Pis pull from `origin/main`, not from your local working tree.
- ❌ Restarting `tracklist-ajax-retry` casually. It's the long-running drain — `make restart-retry` kills its in-flight work. If you just want to deploy code without disrupting it, deploy without restart and the new code picks up on the next process boundary.
- ❌ Starting the scraper to "test something" — it's a full-corpus crawl, not a unit test. Use a smaller test setup.
- ❌ Treating `make deploy` as auto-restart. It's pull-only by design.
- ❌ Editing files directly on pi-storage over ssh as a shortcut. Always commit → push → deploy so the Pis stay in sync with git history.
