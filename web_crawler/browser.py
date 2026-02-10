from __future__ import annotations

import logging
from dataclasses import asdict

from playwright.sync_api import Playwright, BrowserContext

from config import AppConfig, Result


def safe_close_context(context: BrowserContext) -> None:
    """Safely closes the browser context, suppressing errors if already closed."""
    try:
        context.close()
    except Exception:
        pass


def create_browser_context(
    p: Playwright,
    config: AppConfig,
    worker_id: int
) -> Result[BrowserContext, str]:
    """
    Launches a persistent browser context for a specific worker.
    """
    logger = logging.getLogger("BrowserContext")

    # Example: ./profiles/worker_0
    profile_dir = config.profiles.base_dir / f"worker_{worker_id}"

    try:
        # Ensure parent folder exists (./profiles)
        if not config.profiles.base_dir.exists():
            config.profiles.base_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created profiles base directory: {config.profiles.base_dir}")

        logger.info(f"[Worker {worker_id}] Launching browser with profile: {profile_dir}")

        # Convert viewport dataclass to dict
        viewport_dict = asdict(config.browser.viewport)

        context = p.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=config.browser.headless,
            viewport=viewport_dict,
            user_agent=config.browser.user_agent,
            locale=config.browser.locale,
            timezone_id=config.browser.timezone,

            args=config.browser.args,
            timeout=config.browser.nav_timeout_ms,

            # Reduce shared memory usage issues in Docker/Linux
            ignore_default_args=["--enable-automation"],
        )

        # Selector timeout
        context.set_default_timeout(config.browser.selector_timeout_ms)

        # Mask webdriver
        context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)

        return Result.success(context)

    except Exception as e:
        error_msg = f"Failed to launch browser for Worker {worker_id}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return Result.fail(error_msg)
