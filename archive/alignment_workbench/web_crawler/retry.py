from __future__ import annotations

import json
import logging
import random
import sys
import time
from itertools import groupby
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

from archive.alignment_workbench.web_crawler.browser import create_browser_context, safe_close_context
from archive.alignment_workbench.web_crawler.config import AppConfig, load_config
from archive.alignment_workbench.web_crawler.data_models import DJSetTrackMediaLink
from archive.alignment_workbench.web_crawler.database import MusicDatabase
from archive.alignment_workbench.web_crawler.logging_setup import setup_logging
from archive.alignment_workbench.web_crawler.scraper import SOURCE_TYPE, request_ajax_media_link
from archive.alignment_workbench.web_crawler.workers import handle_captcha, is_forwarding_stall_page


def run_retries(cfg: AppConfig, db: MusicDatabase) -> None:
    log = logging.getLogger("RetryWorker")

    failures = db.get_ajax_failures(cfg.retry.max_retries)
    if not failures:
        log.info("No AJAX failures to retry.")
        return

    # Group by set_id so we only open each page once.
    grouped = {
        set_id: list(items)
        for set_id, items in groupby(failures, key=lambda f: f["set_id"])
    }
    log.info("Retrying AJAX failures: %d failures across %d sets", len(failures), len(grouped))

    with sync_playwright() as p:
        ctx_res = create_browser_context(p, cfg, 0)
        if not ctx_res.is_success:
            log.error("Failed to create browser context: %s", ctx_res.error)
            return

        context = ctx_res.value
        try:
            for set_id, set_failures in grouped.items():
                set_url = next((f["set_url"] for f in set_failures if f["set_url"]), None)
                if not set_url:
                    log.warning("No URL for set=%s, skipping %d failures", set_id, len(set_failures))
                    for f in set_failures:
                        db.increment_failure_retries(f["failure_id"])
                    continue

                page = context.new_page()
                try:
                    log.info("Opening set=%s url=%s (%d failures to retry)", set_id, set_url, len(set_failures))
                    page.goto(set_url, wait_until="domcontentloaded")
                    html = page.content()
                    soup = BeautifulSoup(html, "html.parser")

                    if is_forwarding_stall_page(soup):
                        try:
                            page.locator('form input[type="submit"][value="here"]').first.click(timeout=2000)
                            time.sleep(2)
                        except Exception:
                            pass

                    captcha_found, captcha_solved = handle_captcha(page, cfg, log, set_id)
                    if captcha_found and not captcha_solved:
                        log.warning("Captcha unresolved for set=%s, skipping", set_id)
                        for f in set_failures:
                            db.increment_failure_retries(f["failure_id"])
                        continue

                    resolved = 0
                    for f in set_failures:
                        params = json.loads(f["params_json"])
                        res = request_ajax_media_link(params, page)

                        if res.is_success and res.value:
                            for platform, player_id in res.value.items():
                                source_id = None
                                for key, name in SOURCE_TYPE.items():
                                    if name == platform:
                                        source_id = key
                                        break
                                p_map = json.loads(f["params_json"])
                                db.insert_track_media_links([DJSetTrackMediaLink(
                                    set_id=set_id,
                                    tlp_id=f["tlp_id"],
                                    track_id=f["track_id"],
                                    platform=platform,
                                    player_id=player_id,
                                    id_object=p_map.get("idObject"),
                                    id_item=p_map.get("idItem"),
                                    id_source=p_map.get("idSource"),
                                    view_source=p_map.get("viewSource"),
                                    view_item=p_map.get("viewItem"),
                                )])
                            db.delete_failure(f["failure_id"])
                            resolved += 1
                        else:
                            db.increment_failure_retries(f["failure_id"])
                            log.warning(
                                "Retry failed for track=%s (attempt %d/%d): %s",
                                f["track_title"],
                                f["retries"] + 1,
                                cfg.retry.max_retries,
                                res.error if not res.is_success else "empty response",
                            )

                        delay = cfg.retry.retry_delay_s + random.uniform(0, cfg.retry.retry_jitter_s)
                        time.sleep(delay)

                    log.info("Set=%s: resolved %d/%d failures", set_id, resolved, len(set_failures))

                except PlaywrightTimeoutError as e:
                    log.warning("Timeout loading set=%s: %s", set_id, e)
                    for f in set_failures:
                        db.increment_failure_retries(f["failure_id"])
                except Exception:
                    log.exception("Error retrying set=%s", set_id)
                    for f in set_failures:
                        db.increment_failure_retries(f["failure_id"])
                finally:
                    try:
                        page.close()
                    except Exception:
                        pass

                delay = cfg.timing.crawl_delay_s + random.uniform(0, cfg.timing.random_jitter_s)
                time.sleep(delay)

        finally:
            safe_close_context(context)


def main() -> None:
    current_src_dir = Path(__file__).resolve().parent
    project_root = current_src_dir.parent
    config_file_path = project_root / "config.yaml"

    if not config_file_path.exists():
        print(f"CRITICAL: Config file not found at {config_file_path}")
        return

    try:
        cfg = load_config(config_file_path, project_root)
    except Exception as e:
        print(f"CRITICAL: Could not load config: {e}")
        return

    setup_logging(cfg)
    log = logging.getLogger("RetryMain")
    log.info("Starting AJAX retry mechanism...")

    db = MusicDatabase(str(cfg.paths.db_path), str(cfg.paths.schema_path))

    try:
        run_retries(cfg, db)
    except KeyboardInterrupt:
        log.warning("KeyboardInterrupt received! Stopping retry run...")
    finally:
        db.close()
        log.info("Retry run finished. Exiting.")


if __name__ == "__main__":
    main()
