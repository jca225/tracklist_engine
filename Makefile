# Tracklist engine — Mac-side ops Makefile.
#
# Both Pis run code from ~/tracklist_engine cloned from origin/main.
# `make deploy` pulls latest + reinstalls deps on both. Services are
# restarted only on demand (don't interrupt a long-running retry drain
# unless you mean to).
#
# Hosts come from ~/.ssh/config. They go via Tailscale MagicDNS so
# this works whether you're on home WiFi or anywhere else.

PI_STORAGE   := pi-storage
PI_WORKER    := pi-worker
REPO         := ~/tracklist_engine
PIP          := $(REPO)/venvs/web_crawler/bin/pip
DB           := /mnt/storage/data/db/music_database.db

.PHONY: help check check-corpus check-inventory docs-gc docs-gc-apply audit-gt scorecard race align-state align-ablate deploy deploy-storage deploy-worker \
        restart-jobqueue start-scraper stop-scraper restart-retry \
        install-taste-scrape restart-taste-scrape logs-taste-scrape \
        install-corpus-integrity logs-corpus-integrity \
        status logs-jobqueue logs-scraper logs-retry queue ssh-storage ssh-worker

help:
	@echo "Common targets:"
	@echo "  make check            — guardrails script + full pytest suite"
	@echo "  make collide          — which branches will conflict, hotspots, landing order"
	@echo "  make land-budget      — branches grown past the point where landing gets ugly"
	@echo "  make land-verify      — after a rebase: did the merge produce parseable code"
	@echo "  make docs-gc          — classify stale docs (dry run; docs-gc-apply archives)"
	@echo "  make check-inventory SET=<set_id> — slot satisfaction gate (pi-storage)"
	@echo "  make audit-gt SET=<set_id> — audio-verify a labeling .als vs the mix"
	@echo "  make scorecard        — aligner per-span scorecard + failure attribution"
	@echo "  make race             — race classical/agentic/ml drivers on one board (SETS=, DRIVERS=)"
	@echo "  make align-ablate     — run paper ablation matrix → store → print headline + ablation tables"
	@echo "  make deploy           — git pull + pip install on both Pis"
	@echo "  make status           — service states + scrape_failures queue depth"
	@echo "  make queue            — just the scrape_failures count"
	@echo ""
	@echo "Service control (deliberate — won't auto-restart on deploy):"
	@echo "  make restart-jobqueue — bounce the FastAPI server on pi-storage"
	@echo "  make start-scraper    — start tracklist-scraper.service (full corpus)"
	@echo "  make stop-scraper     — stop the scraper"
	@echo "  make restart-retry    — stop + start the retry drain on pi-worker"
	@echo "  make install-taste-scrape — install taste-scrape systemd unit on pi-worker"
	@echo "  make restart-taste-scrape — restart taste scrape loop on pi-worker"
	@echo ""
	@echo "Logs (Ctrl-C to exit):"
	@echo "  make logs-jobqueue logs-scraper logs-retry"
	@echo ""
	@echo "Quick shells:"
	@echo "  make ssh-storage / ssh-worker"

# ---------- local guardrails ------------------------------------------------

check:
	venvs/audio/bin/python scripts/guardrails.py
	bash scripts/typecheck.sh
	venvs/audio/bin/python -m pytest tests/ -q

typecheck:
	bash scripts/typecheck.sh

# ---------- branch hygiene --------------------------------------------------
# Merge cost here tracks commits-since-divergence, not calendar age: branches
# here reach +60 commits in a week. These are read-only (git merge-tree merges
# in memory) and safe to run with worktrees mid-flight. See
# .claude/skills/branch-hygiene/SKILL.md. Requires `braid`
# (uv tool install --editable ~/workspace/braid).

collide:
	@braid status || echo "braid not installed — uv tool install --editable ~/workspace/braid"

land-budget:
	@braid budget || echo "braid not installed — uv tool install --editable ~/workspace/braid"

# Run after a rebase: git has no post-rebase hook, and this repo rebases far
# more than it merges. (Merges are covered by .githooks/post-merge.)
land-verify:
	@braid verify --since 'HEAD@{1}'

# Docs garbage collection — classify docs/ by reachability; archive dead dated
# snapshots. Dry-run by default; `make docs-gc-apply` sweeps COLLECTABLE.
docs-gc:
	venvs/audio/bin/python scripts/docs_gc.py

docs-gc-apply:
	venvs/audio/bin/python scripts/docs_gc.py --apply

# Corpus data-integrity: identity/reference invariants over the CANONICAL DB
# (the data analogue of `make check`, which polices the source tree). ERROR
# rows must be 0; WARN rows are acquisition/routing backlogs.
check-corpus:
	ssh $(PI_STORAGE) 'cd $(REPO) && venvs/audio/bin/python scripts/corpus_integrity.py --db $(DB)'

check-inventory:
	@test -n "$(SET)" || (echo "Usage: make check-inventory SET=<set_id>" && exit 1)
	venvs/audio/bin/python labeling/acquire/pull_set_for_alignment.py $(SET) --check

# Audio-verify a labeling .als against the actual mix (identity / placement /
# ref-offset / pitch per clip). Run after (re-)labeling a set and before
# trusting a GT export — catches silent timestamp drift the XML round-trip
# tests cannot see.
audit-gt:
	@test -n "$(SET)" || (echo "Usage: make audit-gt SET=<set_id> [ALS=<path>]" && exit 1)
	venvs/audio/bin/python -m workspaces.source_detection.als_audit --set-id $(SET) $(if $(ALS),--als $(ALS),)

# One-command aligner scorecard: per-span table + impact-weighted failure
# attribution for BB11+BB12 (reads out/<set>_predicted_timeline_lt.json).
scorecard:
	venvs/audio/bin/python -m eda.alignment.failure_analysis.build_span_table
	venvs/audio/bin/python -m eda.alignment.failure_analysis.analyze

# Race the three end-to-end aligner drivers (classical / agentic / ml) on one
# scorecard board (strict AND fiber-aware always shown). Override SETS/DRIVERS/
# EXTRA (e.g. make race SETS=2nvzlh2k
# EXTRA="--reuse-base 2nvzlh2k=out/2nvzlh2k_predicted_timeline_lt_v2.json").
SETS ?= 1fsnxchk,2nvzlh2k
DRIVERS ?= classical,agentic,ml
race:
	venvs/audio/bin/python -m workspaces.alignment_prototype.drivers.race \
		--sets $(SETS) --drivers $(DRIVERS) $(EXTRA)

# Where is the aligner for a set, and can I trust its timeline? Prints each
# timeline's provenance + FRESH/STALE vs current code/GT/id_map/pi-spine.
# NOPI=1 skips the pi spine check (offline).
align-state:
	@test -n "$(SET)" || { echo "usage: make align-state SET=<set_id> [NOPI=1]"; exit 1; }
	venvs/audio/bin/python scripts/align_state.py --set-id $(SET) $(if $(NOPI),--no-pi,)

# Staged pipeline + ablation framework (docs/pipeline_ablation_framework.md).
# Compose grain reproduces `make race` with auto baseline-injection + a
# reproducible JSONL ledger; CONFIG selects the matrix. Isolate-grain (decoder
# bake-off) ships with the TRM build (docs/trm_decoder_bakeoff.md).
CONFIG ?= workspaces/alignment_prototype/pipeline/configs/race_default.yaml
ablate:
	venvs/audio/bin/python -m workspaces.alignment_prototype.pipeline.cli \
		--config $(CONFIG) $(EXTRA)

# E1 real pseudo-label flywheel (docs/trm_flywheel_design.md §7).
# POOL=unlabeled set, EVAL=hand-GT set, TIMELINE=base predicted timeline,
# SYNTH=generate_v2 root for the synthetic-only control.
POOL ?= w1mgcjt
EVAL ?= 2nvzlh2k
TIMELINE ?= workspaces/alignment_prototype/out/$(POOL)_predicted_timeline.json
SYNTH ?= data/synthetic_mixes_v2
trm-e1:
	venvs/audio/bin/python -m workspaces.alignment_prototype.trajectory.e1 \
		--pool-set $(POOL) --eval-set $(EVAL) \
		--base-timeline $(TIMELINE) --synthetic-root $(SYNTH) $(EXTRA)

align-ablate:
	venvs/audio/bin/python -m workspaces.alignment_prototype.experiments.cli $(EXTRA)

# The kernel entrypoint (P1, docs/architecture_north_star.md): align ONE set
# with the current-best default composition and score it. No flags needed.
# The default driver flips to ml when it wins the race board (P3).
align:
	@test -n "$(SET)" || { echo "usage: make align SET=<set_id>"; exit 1; }
	venvs/audio/bin/python -m workspaces.alignment_prototype.drivers.race \
		--sets $(SET) --drivers classical $(EXTRA)

# W1 determinism check (kernel_data_engine_plan): run the kernel twice, diff.
# Expensive (two full aligns) — nightly-grade, not pre-commit. Byte-identity
# is the bar; if MPS nondeterminism ever breaks it, the span diff below says
# exactly where, and the plan's stated fallback is tolerance-based equality.
PROTO_OUT := workspaces/alignment_prototype/out
determinism:
	@test -n "$(SET)" || { echo "usage: make determinism SET=<set_id>"; exit 1; }
	$(MAKE) align SET=$(SET)
	cp $(PROTO_OUT)/$(SET)_classical_timeline.json $(PROTO_OUT)/$(SET)_determinism_run1.json
	$(MAKE) align SET=$(SET)
	@cmp -s $(PROTO_OUT)/$(SET)_determinism_run1.json $(PROTO_OUT)/$(SET)_classical_timeline.json \
		&& echo "DETERMINISM OK — two runs byte-identical" \
		|| { echo "DETERMINISM FAILED — span-level diff:"; \
		     diff <(venvs/audio/bin/python -m json.tool $(PROTO_OUT)/$(SET)_determinism_run1.json) \
		          <(venvs/audio/bin/python -m json.tool $(PROTO_OUT)/$(SET)_classical_timeline.json) | head -40; exit 1; }

# ---------- deploy ----------------------------------------------------------

deploy: deploy-storage deploy-worker
	@echo ""
	@echo "Done. If you changed code that's loaded by a running service,"
	@echo "restart it: make restart-jobqueue / restart-retry"

deploy-storage:
	@echo "===> pi-storage: pulling + installing"
	ssh $(PI_STORAGE) 'cd $(REPO) && git pull --ff-only origin main && $(PIP) install -q -r requirements.txt'

deploy-worker:
	@echo "===> pi-worker: pulling + installing"
	ssh $(PI_WORKER) 'cd $(REPO) && git pull --ff-only origin main && $(PIP) install -q -r requirements.txt'

# ---------- service control -------------------------------------------------

restart-jobqueue:
	ssh $(PI_STORAGE) 'sudo systemctl restart tracklist-jobqueue.service'
	@sleep 2
	@ssh $(PI_STORAGE) 'sudo systemctl status tracklist-jobqueue.service --no-pager | head -5'

start-scraper:
	@echo "Starting tracklist-scraper.service (limit:0 = full corpus)"
	ssh $(PI_STORAGE) 'sudo systemctl start tracklist-scraper.service'
	@sleep 2
	@ssh $(PI_STORAGE) 'sudo systemctl status tracklist-scraper.service --no-pager | head -5'

stop-scraper:
	ssh $(PI_STORAGE) 'sudo systemctl stop tracklist-scraper.service'

restart-retry:
	@echo "WARNING: this kills the active retry drain. Continue? (Ctrl-C to abort)"
	@sleep 3
	ssh $(PI_WORKER) 'sudo systemctl stop tracklist-ajax-retry.service ; sudo systemctl start tracklist-ajax-retry.service'

# ---------- observability ---------------------------------------------------

status:
	@printf "%-30s " "tracklist-jobqueue (pi-storage):"
	@ssh $(PI_STORAGE) 'systemctl is-active tracklist-jobqueue.service'
	@printf "%-30s " "tracklist-scraper (pi-storage):"
	@ssh $(PI_STORAGE) 'systemctl is-active tracklist-scraper.service' || true
	@printf "%-30s " "tracklist-ajax-retry (pi-worker):"
	@ssh $(PI_WORKER) 'systemctl is-active tracklist-ajax-retry.service' || true
	@printf "%-30s " "scrape_failures queue depth:"
	@ssh $(PI_STORAGE) 'sqlite3 $(DB) "SELECT COUNT(*) FROM scrape_failures"'

queue:
	@ssh $(PI_STORAGE) 'sqlite3 $(DB) "SELECT COUNT(*) FROM scrape_failures"'

logs-jobqueue:
	ssh $(PI_STORAGE) 'sudo journalctl -u tracklist-jobqueue.service -f --no-hostname'

logs-scraper:
	ssh $(PI_STORAGE) 'sudo journalctl -u tracklist-scraper.service -f --no-hostname'

logs-retry:
	ssh $(PI_WORKER) 'sudo journalctl -u tracklist-ajax-retry.service -f --no-hostname'

install-taste-scrape:
	scp deploy/taste-scrape.service $(PI_WORKER):/tmp/taste-scrape.service
	ssh $(PI_WORKER) 'sudo mv /tmp/taste-scrape.service /etc/systemd/system/tracklist-taste-scrape.service && sudo systemctl daemon-reload && sudo systemctl enable tracklist-taste-scrape.service'

restart-taste-scrape:
	ssh $(PI_WORKER) 'sudo systemctl restart tracklist-taste-scrape.service'
	@sleep 2
	@ssh $(PI_WORKER) 'sudo systemctl status tracklist-taste-scrape.service --no-pager | head -8'

logs-taste-scrape:
	ssh $(PI_WORKER) 'sudo journalctl -u tracklist-taste-scrape.service -f --no-hostname'

# Install the daily corpus-integrity watcher on pi-storage (a failed run =
# an ERROR-severity structural violation; it stays failed until the next clean
# run). Requires the code to already be deployed (make deploy).
install-corpus-integrity:
	scp deploy/corpus-integrity.service $(PI_STORAGE):/tmp/corpus-integrity.service
	scp deploy/corpus-integrity.timer $(PI_STORAGE):/tmp/corpus-integrity.timer
	ssh $(PI_STORAGE) 'sudo mv /tmp/corpus-integrity.service /etc/systemd/system/tracklist-corpus-integrity.service && sudo mv /tmp/corpus-integrity.timer /etc/systemd/system/tracklist-corpus-integrity.timer && sudo systemctl daemon-reload && sudo systemctl enable --now tracklist-corpus-integrity.timer && sudo systemctl start tracklist-corpus-integrity.service'
	@ssh $(PI_STORAGE) 'systemctl status tracklist-corpus-integrity.service --no-pager | head -12'

logs-corpus-integrity:
	ssh $(PI_STORAGE) 'sudo journalctl -u tracklist-corpus-integrity.service --no-pager | tail -40'

# ---------- shells ----------------------------------------------------------

ssh-storage:
	ssh $(PI_STORAGE)

ssh-worker:
	ssh $(PI_WORKER)
