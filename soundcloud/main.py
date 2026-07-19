"""CLI — SoundCloud data lake: sync-user / crawl / stats."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from typing import Any

from soundcloud import client, crawl, store
from soundcloud.config import load_settings
from soundcloud.records import CrawlPolicy

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("soundcloud")

_TABLES = (
    "sc_users",
    "sc_tracks",
    "sc_playlists",
    "sc_likes",
    "sc_reposts",
    "sc_follows",
    "sc_playlist_tracks",
)


def _now(args: argparse.Namespace) -> str:
    return args.now or datetime.now(timezone.utc).isoformat()


def _bootstrap_client(settings) -> tuple[Any, Any, str]:
    c = client.sc_client()
    rl = client.RateLimiter(settings.rpm)
    cid = client.extract_client_id(c, rl)
    return c, rl, cid


def _run_crawl(policy: CrawlPolicy, now: str) -> int:
    settings = load_settings(rpm=policy.rpm)
    store.init_db(settings.db_path)
    c, rl, cid = _bootstrap_client(settings)
    with store.connect(settings.db_path) as conn:
        counts = crawl.crawl(conn, settings, policy, c, rl, cid, now)
        conn.commit()
    logger.info("crawl counts: %s", counts)
    return 0


def cmd_sync_user(args: argparse.Namespace) -> int:
    settings = load_settings()
    target = args.target
    if target.isdigit():
        uid = int(target)
    else:
        c, rl, cid = _bootstrap_client(settings)
        resolved = client.resolve(c, rl, cid, target)
        if resolved.get("kind") != "user":
            logger.error("resolved kind=%s, expected user", resolved.get("kind"))
            return 2
        uid = int(resolved["id"])
    return _run_crawl(
        CrawlPolicy(seed_user_ids=(uid,), depth=1, rpm=settings.rpm), _now(args)
    )


def cmd_crawl(args: argparse.Namespace) -> int:
    settings = load_settings(rpm=args.rpm)
    policy = CrawlPolicy(seed_user_ids=(args.seed,), depth=args.depth, rpm=settings.rpm)
    return _run_crawl(policy, _now(args))


def cmd_stats(args: argparse.Namespace) -> int:
    settings = load_settings()
    store.init_db(settings.db_path)
    with store.connect(settings.db_path) as conn:
        for t in _TABLES:
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"{t:20s} {n}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SoundCloud data lake")
    p.add_argument("--now", default=None, help=argparse.SUPPRESS)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_sync = sub.add_parser(
        "sync-user", help="Sync one user's public library (depth-1)"
    )
    p_sync.add_argument("target", help="profile URL or numeric sc_user_id")
    p_sync.add_argument("--now", default=None, help=argparse.SUPPRESS)
    p_sync.set_defaults(func=cmd_sync_user)

    p_crawl = sub.add_parser("crawl", help="Frontier crawl from a seed user id")
    p_crawl.add_argument("--seed", type=int, required=True)
    p_crawl.add_argument("--depth", type=int, default=1)
    p_crawl.add_argument("--rpm", type=int, default=None)
    p_crawl.add_argument("--now", default=None, help=argparse.SUPPRESS)
    p_crawl.set_defaults(func=cmd_crawl)

    p_stats = sub.add_parser("stats", help="Print node/edge coverage")
    p_stats.add_argument("--now", default=None, help=argparse.SUPPRESS)
    p_stats.set_defaults(func=cmd_stats)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
