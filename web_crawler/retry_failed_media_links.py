from __future__ import annotations
import pandas as pd
import hashlib
import logging
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
import re
import base64

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

from browser import create_browser_context, safe_close_context
from captcha_solver import EmailCaptchaSolver, solve_captcha_ocr
from config import AppConfig
from data_models import DJSet, DJSetCrawl, ScrapeFailure
from database import MusicDatabase
from scraper import ScrapedSetData, scrape_dj_set

DEFAULT_KILL_AFTER_CONSECUTIVE_FAILURES = 5


def extract_tracklist_id(soup: BeautifulSoup) -> str | None:
    """The point of this method is to assert we have actually ended up at the correct page
       (In some extraneous cases we are redirected to a different page)"""
    tag = soup.find("meta", attrs={"property": "og:url"})
    url = tag.get("content") if tag else None

    if not url:
        link = soup.find("link", rel="canonical")
        url = link.get("href") if link else None

    if not url:
        return None

    path = urlparse(url).path
    m = re.search(r"^/tracklist/([^/]+)/", path)
    return m.group(1) if m else None


def is_forwarding_stall_page(soup: BeautifulSoup) -> bool:
    body = soup.find("body", id="body")
    if not body:
        return False
    text = body.get_text(" ", strip=True).lower()
    if "please wait, you will be forwarded" in text:
        return True
    if "turnstile" in text or "cf-turnstile-response" in str(body):
        return True
    return False


def _get_captcha_image_bytes(page, log: logging.Logger) -> bytes | None:
    img = page.query_selector("img[alt='Captcha']")
    if not img:
        return None
    src = img.get_attribute("src")
    if src and "base64," in src:
        try:
            b64 = src.split("base64,", 1)[1]
            return base64.b64decode(b64)
        except Exception:
            log.warning("Failed to decode base64 captcha image.")
            return None
    try:
        return img.screenshot(type="png")
    except Exception:
        log.warning("Failed to screenshot captcha image.")
        return None


def _submit_captcha_solution(page, solution: str, log: logging.Logger) -> bool:
    input_el = page.query_selector("#captcha")
    if not input_el:
        log.warning("Could not find captcha input field.")
        return False
    input_el.fill(solution)

    submit_btn = page.query_selector("button[type='submit']")
    if not submit_btn:
        log.warning("Could not find captcha submit button.")
        return False
    submit_btn.click()
    page.wait_for_timeout(3000)
    return True


def _persist_captcha_image(img_bytes: bytes, set_id: str, base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = base_dir / f"{set_id}_{stamp}.png"
    path.write_bytes(img_bytes)


def handle_captcha(page, cfg: AppConfig, log: logging.Logger, set_id: str) -> tuple[bool, bool]:
    img = page.query_selector("img[alt='Captcha']")
    if not img:
        return False, True

    log.warning("Captcha found, attempting to solve...")
    email_solver = EmailCaptchaSolver()

    while img:
        solved = False
        for attempt in range(cfg.captcha.solver_max_attempts):
            img_bytes = _get_captcha_image_bytes(page, log)
            if not img_bytes:
                log.warning("Captcha image missing or not base64.")
                return True, False

            _persist_captcha_image(img_bytes, set_id, cfg.paths.captcha_imgs_dir)

            solution = solve_captcha_ocr(img_bytes)
            if not solution:
                log.warning(f"Captcha API solver failed on attempt {attempt + 1}.")
                continue

            log.info(f"Captcha solution guessed (attempt {attempt + 1}).")
            if not _submit_captcha_solution(page, solution, log):
                return True, False

            img = page.query_selector("img[alt='Captcha']")
            if not img:
                solved = True
                break

        if img:
            if not email_solver.is_configured():
                log.warning("Email captcha solver not configured; cannot continue.")
                return True, False

            log.warning("Captcha still present; falling back to email solver.")
            img_bytes = _get_captcha_image_bytes(page, log)
            if not img_bytes:
                return True, False

            _persist_captcha_image(img_bytes, set_id, cfg.paths.captcha_imgs_dir)

            # wait only on email fallback; max_wait_s <= 0 means wait indefinitely
            solution = email_solver.solve_captcha(img_bytes, max_wait_s=cfg.captcha.captcha_wait_s)
            if not solution:
                log.warning("Email captcha solver did not return a solution.")
                return True, False

            if not _submit_captcha_solution(page, solution, log):
                return True, False

            img = page.query_selector("img[alt='Captcha']")
            if not img:
                solved = True

        if not solved:
            return True, False

    return True, True


def _persist_html(html: str, set_id: str, base_dir: Path) -> tuple[str, str]:
    base_dir.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(html.encode("utf-8")).hexdigest()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{set_id}_{stamp}.html"
    path = base_dir / filename
    path.write_text(html, encoding="utf-8")
    return sha, str(path)




def retry_failed_media_links(
    df: pd.DataFrame,
    cfg: AppConfig,
    db: MusicDatabase,
) -> None:
    log = logging.getLogger("SerialWorker")
    sets_processed_total = 0
    sets_since_restart = 0
    consecutive_failures = 0
    kill_after_failures = max(
        1,
        int(
            getattr(
                cfg.failure,
                "kill_process_after_consecutive_failures",
                DEFAULT_KILL_AFTER_CONSECUTIVE_FAILURES,
            )
        ),
    )

    def send_failure_threshold_email() -> None:
        email_solver = EmailCaptchaSolver()
        if not email_solver.is_configured():
            log.warning("Failure-threshold email not sent because captcha email is not configured.")
            return

        body = (
            "The web crawler stopped after hitting the consecutive failure threshold.\n\n"
            f"Consecutive failures: {consecutive_failures}\n"
            f"Threshold: {kill_after_failures}\n"
            f"UTC time: {datetime.now(timezone.utc).isoformat()}\n"
            f"PID: {os.getpid()}\n"
        )
        try:
            email_solver.send_notification_email(
                subject="Web Crawler Stopped: Failure Threshold Reached",
                body=body,
            )
            log.info("Failure-threshold notification email sent.")
        except Exception:
            log.exception("Failed to send failure-threshold notification email.")

    def record_failure(reason: str) -> None:
        nonlocal consecutive_failures
        consecutive_failures += 1
        log.warning(
            "Consecutive crawl/scrape failures: %d/%d (%s)",
            consecutive_failures,
            kill_after_failures,
            reason,
        )
        if consecutive_failures >= kill_after_failures:
            log.critical(
                "Failure threshold reached (%d consecutive failures). Killing Python process.",
                consecutive_failures,
            )
            send_failure_threshold_email()
            os._exit(1)

    def reset_failure_streak() -> None:
        nonlocal consecutive_failures
        if consecutive_failures:
            log.info("Resetting failure streak after successful scrape.")
        consecutive_failures = 0

    with sync_playwright() as p:
        ctx_res = create_browser_context(p, cfg, 0)
        if not ctx_res.is_success:
            log.error(f"Failed to create initial context: {ctx_res.error}")
            record_failure("initial_context_creation_failed")
            return

        context = ctx_res.value
        try:
            for _, row in df.iterrows():
                tracklist_id = row["set_id"]
                url = row["set_url"]

                if sets_since_restart >= cfg.execution.restart_every_n_sets:
                    log.info(f"Hit restart limit ({sets_since_restart}). Refreshing browser context...")
                    safe_close_context(context)
                    ctx_res = create_browser_context(p, cfg, 0)
                    if not ctx_res.is_success:
                        log.critical(f"Failed to restart context: {ctx_res.error}")
                        record_failure("context_restart_failed")
                        return
                    context = ctx_res.value
                    sets_since_restart = 0

                page = context.new_page()
                try:
                    if sets_processed_total % cfg.execution.log_every_n_sets == 0:
                        log.info(f"Stats: Processed {sets_processed_total} sets so far.")

                    log.info(f"Scraping set={tracklist_id} url={url}")

                    response = page.goto(url, wait_until="domcontentloaded")
                    html = page.content()
                    soup = BeautifulSoup(html, "html.parser")

                    if is_forwarding_stall_page(soup):
                        log.warning(f"Forwarding/captcha stall page for set={tracklist_id} - attempting click")
                        try:
                            page.locator('form input[type=\"submit\"][value=\"here\"]').first.click(timeout=2000)
                            time.sleep(2)
                            html = page.content()
                            soup = BeautifulSoup(html, "html.parser")
                        except Exception:
                            pass

                    captcha_found, captcha_solved = handle_captcha(page, cfg, log, tracklist_id)
                    if captcha_found and not captcha_solved:
                        log.warning(f"Captcha unresolved for set={tracklist_id}")
                        db.insert_failures([ScrapeFailure(
                            set_id=tracklist_id,
                            set_url=url,
                            stage="captcha",
                            error="captcha_unsolved",
                        )])
                        record_failure("captcha_unsolved")
                        if cfg.failure.fail_fast:
                            break
                        else:
                            continue

                    html = page.content()
                    soup = BeautifulSoup(html, "html.parser")

                    if (soup_tracklist := extract_tracklist_id(soup)) != tracklist_id:
                        page.close()
                        log.warning(
                            f"Tracklist {url} did not have the same tracklist ID "
                            f"{tracklist_id} as the html retrieved {soup_tracklist}"
                        )
                        record_failure("tracklist_id_mismatch")
                        continue

                    res = scrape_dj_set(soup, tracklist_id, page)
                    if not res.is_success:
                        log.warning(f"Scrape failed for {tracklist_id}: {res.error}")
                        db.insert_failures([ScrapeFailure(
                            set_id=tracklist_id,
                            set_url=url,
                            stage="scrape",
                            error=res.error,
                        )])
                        record_failure("scrape_failed")
                        if cfg.failure.fail_fast:
                            break
                        else:
                            continue

                    data: ScrapedSetData = res.value

                    html_dir = Path(__file__).resolve().parent.parent / "data" / "html"
                    html_sha, html_path = _persist_html(html, tracklist_id, html_dir)

                    status = response.status if response else None
                    headers = response.headers if response else {}
                    crawl = DJSetCrawl(
                        set_id=tracklist_id,
                        set_url=url,
                        http_status=status,
                        etag=headers.get("etag"),
                        last_modified=headers.get("last-modified"),
                        html_sha256=html_sha,
                        html_path=html_path,
                    )

                    log.info(f"Done set={tracklist_id} media_links={len(data.set_media_links)}")
                    reset_failure_streak()

                    sets_processed_total += 1
                    sets_since_restart += 1

                except PlaywrightTimeoutError as e:
                    log.warning(f"Timeout set={tracklist_id}: {e}")
                    record_failure("playwright_timeout")
                    if cfg.failure.fail_fast:
                        break
                except Exception:
                    log.exception(f"Serial worker crashed on set={tracklist_id}")
                    record_failure("worker_exception")
                    if cfg.failure.fail_fast:
                        break
                finally:
                    try:
                        page.close()
                    except Exception:
                        pass

                delay = cfg.timing.crawl_delay_s
                jitter = random.uniform(0, cfg.timing.random_jitter_s)
                time.sleep(delay + jitter)

        finally:
            db.close_thread()
            safe_close_context(context)


def main():
    DB_PATH = Path("/home/ubuntu/tracklist_engine/data/db/music_database.db")
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
    log = logging.getLogger("Main")
    log.info("Starting Scraper Application...")

    db = MusicDatabase(str(cfg.paths.db_path), str(cfg.paths.schema_path))

    # Count DJ sets in DB (and html files if you want)
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    scrape_failures_with_dj_links_query = """
        SELECT
            f.failure_id,
            f.set_id,
            s.set_url,
            f.track_title,
            f.track_id,
            f.tlp_id,
            f.params_json,
            f.error,
            f.created_at
        FROM scrape_failures f
        INNER JOIN dj_sets s
            ON f.set_id = s.set_id
        WHERE EXISTS (
            SELECT 1
            FROM dj_set_media_links l
            WHERE l.set_id = f.set_id
        );
    """
    scrape_failures_with_dj_links_df = pd.read_sql_query(scrape_failures_with_dj_links_query, conn)
    retry_failed_media_links(scrape_failures_with_dj_links_df, cfg, db)
    cur.close()



if __name__ == "__main__":
    main()