"""Deepen a focused list of mix cohorts to a target enriched-user count, in ONE process.

Lesson from v1 (round-robin, unbounded collect): on mega-cohorts (28k-178k likers)
collect re-paged likers every pass while enrich crawled, and one stalled mix starved
the rest. Fix: enrich is the bottleneck and the thing we actually need, so

  - collect only to a LISTENER FLOOR (enough pool to enrich the target), then stop;
  - then hammer enrich_batch until the cohort hits --target enriched users;
  - process mixes SEQUENTIALLY to target (a stall is visible + isolated, not silent).

Sustained ~45 rpm (single RateLimiter, sequential). Resumable (warehouse checkpoints).

  venvs/audio/bin/python -m personalization.cohort_driver --target 500 \
      --mixes rufus_mayan hardwell_tml kygo_ultra jauz_hard dom_dolla_edc
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import time

from personalization.collect import collect_tick
from personalization.config import load_settings, mix_by_id
from personalization.enrich import enrich_batch
from personalization.persistence import init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cohort_driver")


def _count(db, sql, mix_id):
    c = sqlite3.connect(db)
    try:
        return c.execute(sql, (mix_id,)).fetchone()[0]
    finally:
        c.close()


def enriched(db, mix_id):
    return _count(db, "SELECT COUNT(DISTINCT user_id) FROM sc_likes WHERE mix_id=?", mix_id)


def collected(db, mix_id):
    return _count(db, "SELECT COUNT(*) FROM listeners WHERE mix_id=?", mix_id)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixes", nargs="+", required=True)
    ap.add_argument("--target", type=int, default=500, help="enriched-user target per cohort")
    ap.add_argument("--batch", type=int, default=30)
    ap.add_argument("--stall-passes", type=int, default=6,
                    help="give up on a cohort after this many enrich passes with no growth")
    args = ap.parse_args()

    settings = load_settings()
    init_db(settings.db_path)

    for mid in args.mixes:
        mix = mix_by_id(mid)
        floor = args.target * 3                      # listener pool >> target (skips, dupes, deferrals)
        while collected(settings.db_path, mid) < floor:
            try:
                n = collect_tick(settings, mix)
            except Exception:
                log.exception("collect failed %s", mid); time.sleep(5); break
            log.info("%-16s collect: listeners=%d (floor %d)", mid, collected(settings.db_path, mid), floor)
            if n == 0:
                break                                # likers exhausted

        stall, last = 0, enriched(settings.db_path, mid)
        while enriched(settings.db_path, mid) < args.target and stall < args.stall_passes:
            try:
                enrich_batch(settings, mix, batch_size=args.batch)
            except Exception:
                log.exception("enrich failed %s", mid); time.sleep(5)
            now = enriched(settings.db_path, mid)
            stall = stall + 1 if now <= last else 0
            last = now
            log.info("%-16s enriched=%d/%d  (stall %d/%d)", mid, now, args.target, stall, args.stall_passes)
        log.info("DONE %-16s enriched=%d/%d", mid, enriched(settings.db_path, mid), args.target)

    log.info("all cohorts processed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
