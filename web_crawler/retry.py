"""AJAX-failure retry for `scrape_failures` rows with `stage='ajax'`.

Hardening (vs. the old corpus-wide retry):
- Stall-page polling (Cloudflare forwarding) up to ~20s with re-checks.
- Captcha solved via the same `handle_captcha` the main scraper uses;
  on solve, the page DOM is re-read before issuing AJAX calls.
- Per-track exponential backoff with reset-on-success (`--base-delay`,
  `--max-backoff`).
- Circuit breaker (`--abort-after`) so we don't grind the queue when the
  site rate-limits us or the captcha solver stalls.
- `--dry-run` reports the queue without touching the live site.
- `--limit N` caps how many failures we attempt — useful for cautious
  first passes.

Scope filters (mutually exclusive with each other; default is corpus-wide):
- `--big-bootie`             only retry sets whose title contains 'Big Bootie'
- `--title-like '%pattern%'` arbitrary SQL LIKE filter on `dj_sets.title`

Runs inside the micromamba `tracklist_engine` env (has playwright + the
rest of the scraper stack):

    $HOME/micromamba/envs/tracklist_engine/bin/python \
        web_crawler/retry.py --dry-run                    # preview corpus-wide
    $HOME/micromamba/envs/tracklist_engine/bin/python \
        web_crawler/retry.py --big-bootie --limit 2       # cautious BB pass
    $HOME/micromamba/envs/tracklist_engine/bin/python \
        web_crawler/retry.py                              # run everything
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from itertools import groupby
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

from browser import create_browser_context, safe_close_context
from config import AppConfig, load_config
from data_models import DJSetTrackMediaLink
from database import MusicDatabase
from logging_setup import setup_logging
from scraper import request_ajax_media_link
from workers import handle_captcha, is_forwarding_stall_page


BIG_BOOTIE_TITLE_LIKE = "%Big Bootie%"


# ---------- DB query --------------------------------------------------------
# Moved to MusicDatabase.fetch_ajax_failures so JobQueueClient can mirror
# the same interface for RPC-mode runs from pi-worker. Keep this thin
# wrapper for backward-compat; new callers should use the method directly.

def fetch_ajax_failures(
    db,
    max_retries: int,
    title_like: str | None,
) -> list[dict]:
    return db.fetch_ajax_failures(max_retries, title_like)


# ---------- retry runner ----------------------------------------------------

def _ajax_retry_one(page, failure: dict) -> tuple[bool, str | None]:
    """Attempt to resolve a single AJAX failure.
    Returns (success, error_detail_or_platforms_json)."""
    try:
        params = json.loads(failure["params_json"])
    except Exception as e:
        return False, f"bad_params_json: {e}"

    try:
        res = request_ajax_media_link(params, page)
    except Exception as e:
        return False, f"ajax_exception: {e}"

    if not res.is_success:
        return False, f"ajax_error: {res.error}"
    if not res.value:
        return False, "ajax_empty"
    return True, json.dumps(res.value)


def run_retries(
    cfg: AppConfig,
    db: MusicDatabase,
    *,
    title_like: str | None,
    limit_failures: int,
    base_delay_s: float,
    max_backoff_s: float,
    abort_after_consecutive: int,
    dry_run: bool,
) -> None:
    log = logging.getLogger("Retry")

    failures = fetch_ajax_failures(db, cfg.retry.max_retries, title_like)
    if not failures:
        log.info("No AJAX failures to retry (title_like=%s).", title_like)
        return

    if limit_failures > 0:
        failures = failures[:limit_failures]

    grouped = {sid: list(items) for sid, items in groupby(failures, key=lambda f: f["set_id"])}
    total = sum(len(v) for v in grouped.values())
    log.info(
        "Scope: %d AJAX failures across %d sets (title_like=%s, limit=%d, dry_run=%s)",
        total, len(grouped), title_like, limit_failures, dry_run,
    )

    for sid, fs in list(grouped.items())[:25]:
        sample = [f["track_title"] or f["track_id"] for f in fs[:3]] + (["…"] if len(fs) > 3 else [])
        log.info("  set=%s  failures=%d  tracks=%s", sid, len(fs), sample)
    if len(grouped) > 25:
        log.info("  … %d more sets", len(grouped) - 25)

    if dry_run:
        log.info("DRY RUN — exiting before touching the live site.")
        return

    resolved_total = 0
    failed_total = 0
    consecutive_fail = 0
    current_delay = base_delay_s

    with sync_playwright() as p:
        ctx_res = create_browser_context(p, cfg, 0)
        if not ctx_res.is_success:
            log.error("Failed to create browser context: %s", ctx_res.error)
            return

        context = ctx_res.value
        try:
            for sid, set_failures in grouped.items():
                if consecutive_fail >= abort_after_consecutive:
                    log.error(
                        "Aborting: %d consecutive failures (likely a captcha wall or block).",
                        consecutive_fail,
                    )
                    break

                set_url = next((f["set_url"] for f in set_failures if f["set_url"]), None)
                if not set_url:
                    log.warning("No URL for set=%s, skipping %d failures", sid, len(set_failures))
                    for f in set_failures:
                        db.increment_failure_retries(f["failure_id"])
                    continue

                page = context.new_page()
                try:
                    log.info(
                        "→ opening set=%s url=%s  (queued %d failures)",
                        sid, set_url, len(set_failures),
                    )
                    page.goto(set_url, wait_until="domcontentloaded")
                    time.sleep(2)
                    html = page.content()
                    soup = BeautifulSoup(html, "html.parser")

                    # Cloudflare / forwarding stall — try the "click here" bypass,
                    # then wait up to ~20s for the real page to land.
                    if is_forwarding_stall_page(soup):
                        log.info("  stall page detected; attempting click-through")
                        try:
                            page.locator('form input[type="submit"][value="here"]').first.click(timeout=2000)
                        except Exception:
                            pass
                        for _ in range(10):
                            time.sleep(2)
                            html = page.content()
                            soup = BeautifulSoup(html, "html.parser")
                            if not is_forwarding_stall_page(soup):
                                log.info("  cleared stall page")
                                break
                        else:
                            log.warning("still on stall page for set=%s; skipping", sid)
                            for f in set_failures:
                                db.increment_failure_retries(f["failure_id"])
                            consecutive_fail += 1
                            current_delay = min(current_delay * 2, max_backoff_s)
                            continue

                    # Captcha — local OCR via ddddocr (no API key required) with
                    # an optional email-solver fallback for hard images.
                    captcha_found, captcha_solved = handle_captcha(page, cfg, log, sid)
                    if captcha_found and not captcha_solved:
                        log.warning("captcha on set=%s not solved by OCR; skipping", sid)
                        for f in set_failures:
                            db.increment_failure_retries(f["failure_id"])
                        consecutive_fail += 1
                        current_delay = min(current_delay * 2, max_backoff_s)
                        continue
                    if captcha_found and captcha_solved:
                        log.info("  captcha solved via OCR")
                        # Refresh the DOM after solve; Playwright page is still live.
                        html = page.content()
                        soup = BeautifulSoup(html, "html.parser")

                    resolved_in_set = 0
                    for f in set_failures:
                        ok, detail = _ajax_retry_one(page, f)
                        if ok:
                            platforms = json.loads(detail)
                            params_map = json.loads(f["params_json"])
                            for platform, player_id in platforms.items():
                                db.insert_track_media_links([DJSetTrackMediaLink(
                                    set_id=sid,
                                    tlp_id=f["tlp_id"],
                                    track_id=f["track_id"],
                                    platform=platform,
                                    player_id=player_id,
                                    id_object=params_map.get("idObject"),
                                    id_item=params_map.get("idItem"),
                                    id_source=params_map.get("idSource"),
                                    view_source=params_map.get("viewSource"),
                                    view_item=params_map.get("viewItem"),
                                )])
                            db.delete_failure(f["failure_id"])
                            resolved_in_set += 1
                            resolved_total += 1
                            consecutive_fail = 0
                            current_delay = base_delay_s  # reset backoff on success
                            log.info(
                                "  ✓ track=%s → platforms=%s",
                                f["track_title"] or f["track_id"], list(platforms.keys()),
                            )
                        else:
                            db.increment_failure_retries(f["failure_id"])
                            failed_total += 1
                            consecutive_fail += 1
                            current_delay = min(current_delay * 2, max_backoff_s)
                            log.warning(
                                "  ✗ track=%s: %s (backoff → %ds)",
                                f["track_title"] or f["track_id"], detail, int(current_delay),
                            )
                            if consecutive_fail >= abort_after_consecutive:
                                break

                        # Per-track delay with jitter + backoff.
                        jitter = random.uniform(0, 2.0)
                        time.sleep(current_delay + jitter)

                    log.info(
                        "set=%s: resolved %d/%d in this pass",
                        sid, resolved_in_set, len(set_failures),
                    )

                except PlaywrightTimeoutError as e:
                    log.warning("timeout loading set=%s: %s", sid, e)
                    for f in set_failures:
                        db.increment_failure_retries(f["failure_id"])
                    consecutive_fail += 1
                    current_delay = min(current_delay * 2, max_backoff_s)
                except Exception:
                    log.exception("error retrying set=%s", sid)
                    for f in set_failures:
                        db.increment_failure_retries(f["failure_id"])
                    consecutive_fail += 1
                    current_delay = min(current_delay * 2, max_backoff_s)
                finally:
                    try:
                        page.close()
                    except Exception:
                        pass

                # Between-set delay (separate from per-track delay)
                inter_set = cfg.timing.crawl_delay_s + random.uniform(0, cfg.timing.random_jitter_s)
                time.sleep(inter_set)

        finally:
            safe_close_context(context)

    log.info(
        "Done. resolved=%d failed=%d consecutive_fail=%d",
        resolved_total, failed_total, consecutive_fail,
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be retried without touching the site.")
    p.add_argument("--limit", type=int, default=0,
                   help="Max failures to attempt (0 = all queued).")
    p.add_argument("--base-delay", type=float, default=8.0,
                   help="Seconds between per-track AJAX calls on success.")
    p.add_argument("--max-backoff", type=float, default=120.0,
                   help="Cap on exponential backoff between calls after failures.")
    p.add_argument("--abort-after", type=int, default=3,
                   help="Abort the whole run if this many consecutive failures happen.")

    scope = p.add_mutually_exclusive_group()
    scope.add_argument("--big-bootie", action="store_true",
                       help="Restrict retries to Big Bootie sets (title LIKE '%%Big Bootie%%').")
    scope.add_argument("--title-like", type=str, default=None,
                       help="Restrict retries to sets whose dj_sets.title matches this SQL LIKE pattern.")

    p.add_argument("--rpc-url", type=str, default=None,
                   help="When set, talk to a remote jobqueue server instead of opening "
                        "the SQLite DB directly. Used by pi-worker to retry against pi-storage's DB. "
                        "Token comes from $JOBQUEUE_TOKEN.")

    args = p.parse_args()
    title_like = BIG_BOOTIE_TITLE_LIKE if args.big_bootie else args.title_like

    current_src_dir = Path(__file__).resolve().parent
    project_root = current_src_dir.parent
    config_file_path = project_root / "config.yaml"
    if not config_file_path.exists():
        print(f"CRITICAL: Config file not found at {config_file_path}", file=sys.stderr)
        return 2

    try:
        cfg = load_config(config_file_path, project_root)
    except Exception as e:
        print(f"CRITICAL: Could not load config: {e}", file=sys.stderr)
        return 2

    setup_logging(cfg)
    log = logging.getLogger("Retry-Main")
    log.info(
        "AJAX retry — title_like=%s limit=%d dry_run=%s rpc=%s",
        title_like, args.limit, args.dry_run, args.rpc_url or "off",
    )

    db: object
    if args.rpc_url:
        from jobqueue.client import JobQueueClient
        db = JobQueueClient(args.rpc_url)
    else:
        db = MusicDatabase(str(cfg.paths.db_path), str(cfg.paths.schema_path))

    try:
        run_retries(
            cfg, db,
            title_like=title_like,
            limit_failures=args.limit,
            base_delay_s=args.base_delay,
            max_backoff_s=args.max_backoff,
            abort_after_consecutive=args.abort_after,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
    finally:
        db.close()
        log.info("exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
