"""Round-robin collect+enrich across a focused list of mix cohorts, in ONE process.

Why not N parallel `loop` processes: N clients bursting from one IP trips SoundCloud
rate limits. One process doing ticks sequentially keeps the instantaneous request
rate at a single RateLimiter's budget (~45 rpm) while still advancing every cohort
each pass — the safe way to scrape many cohorts at once.

Each pass, per mix: collect a tick (until likers exhausted), then enrich a batch of
listeners' like-histories. Runs until all cohorts hit the target enriched-user count
or you Ctrl-C. Resumable (checkpoints in the warehouse).

  venvs/audio/bin/python -m personalization.cohort_driver --target 1200 \
      --mixes rlgrime_hw5 porter_essential hardwell_tml kygo_ultra jauz_hard dom_dolla_edc rufus_mayan
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


def enriched_count(db, mix_id: str) -> int:
    c = sqlite3.connect(db)
    try:
        return c.execute("SELECT COUNT(DISTINCT user_id) FROM sc_likes WHERE mix_id=?", (mix_id,)).fetchone()[0]
    finally:
        c.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixes", nargs="+", required=True, help="mix_ids to deepen")
    ap.add_argument("--target", type=int, default=1200, help="stop a cohort at this many enriched users")
    ap.add_argument("--batch", type=int, default=40)
    args = ap.parse_args()

    settings = load_settings()
    init_db(settings.db_path)
    mixes = [mix_by_id(m) for m in args.mixes]

    while True:
        done = 0
        for mix in mixes:
            n = enriched_count(settings.db_path, mix.mix_id)
            if n >= args.target:
                done += 1
                continue
            try:
                collect_tick(settings, mix)
                enrich_batch(settings, mix, batch_size=args.batch)
            except Exception:
                log.exception("tick failed for %s", mix.mix_id)
                time.sleep(5)
            log.info("%-16s enriched=%d/%d", mix.mix_id, enriched_count(settings.db_path, mix.mix_id), args.target)
        if done == len(mixes):
            log.info("all %d cohorts reached target %d", len(mixes), args.target)
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
