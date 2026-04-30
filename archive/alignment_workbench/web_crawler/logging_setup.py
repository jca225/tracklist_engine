from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from archive.alignment_workbench.web_crawler.config import AppConfig


def setup_logging(config: AppConfig) -> None:
    """Sets up basic logging configuration."""
    # Ensure log directories exist
    if not config.paths.log_dir.exists():
        config.paths.log_dir.mkdir(parents=True, exist_ok=True)

    log_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    log_file = config.paths.log_dir / "latest.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )

    handlers = [
        logging.StreamHandler(),
        file_handler,
    ]

    logging.basicConfig(level=logging.INFO, format=log_fmt, handlers=handlers, force=True)
