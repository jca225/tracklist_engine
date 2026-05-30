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

.PHONY: help check deploy deploy-storage deploy-worker \
        restart-jobqueue start-scraper stop-scraper restart-retry \
        status logs-jobqueue logs-scraper logs-retry queue ssh-storage ssh-worker

help:
	@echo "Common targets:"
	@echo "  make check            — guardrails script + full pytest suite"
	@echo "  make deploy           — git pull + pip install on both Pis"
	@echo "  make status           — service states + scrape_failures queue depth"
	@echo "  make queue            — just the scrape_failures count"
	@echo ""
	@echo "Service control (deliberate — won't auto-restart on deploy):"
	@echo "  make restart-jobqueue — bounce the FastAPI server on pi-storage"
	@echo "  make start-scraper    — start tracklist-scraper.service (full corpus)"
	@echo "  make stop-scraper     — stop the scraper"
	@echo "  make restart-retry    — stop + start the retry drain on pi-worker"
	@echo ""
	@echo "Logs (Ctrl-C to exit):"
	@echo "  make logs-jobqueue logs-scraper logs-retry"
	@echo ""
	@echo "Quick shells:"
	@echo "  make ssh-storage / ssh-worker"

# ---------- local guardrails ------------------------------------------------

check:
	venvs/audio/bin/python scripts/guardrails.py
	venvs/audio/bin/python -m pytest tests/ -q

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

# ---------- shells ----------------------------------------------------------

ssh-storage:
	ssh $(PI_STORAGE)

ssh-worker:
	ssh $(PI_WORKER)
