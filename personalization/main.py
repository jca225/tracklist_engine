"""CLI — taste prior scrape loop for pi-worker."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from core.result import Err, Ok
from personalization.bot_heuristics import score_mix_listeners
from personalization.collect import collect_tick
from personalization.comment_heatmap import run_heatmap_analysis
from personalization.config import load_mixes, load_settings, mix_by_id
from personalization.enrich import enrich_batch
from personalization.enrich_playlists import enrich_playlists_batch
from personalization.import_archive import import_archive_dir
from personalization.persistence import connect, init_db, migrate_db, status_counts
from personalization.pipeline import run_analysis_pipeline
from personalization.prior_mert import run_prior_pipeline
from personalization.taste_cluster import cluster_mix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("taste_prior")


def cmd_init_db(_: argparse.Namespace) -> int:
    settings = load_settings()
    init_db(settings.db_path)
    print(f"initialized {settings.db_path}")
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    settings = load_settings()
    if not settings.db_path.is_file():
        print(json.dumps({"error": "db missing — run init-db"}, indent=2))
        return 1
    with connect(settings.db_path) as conn:
        print(json.dumps(status_counts(conn), indent=2))
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    settings = load_settings()
    init_db(settings.db_path)
    mix = mix_by_id(args.mix)
    n = collect_tick(settings, mix)
    print(f"collect tick: {n} listener upserts")
    return 0


def cmd_enrich(args: argparse.Namespace) -> int:
    settings = load_settings()
    init_db(settings.db_path)
    mix = mix_by_id(args.mix)
    n = enrich_batch(settings, mix, batch_size=args.batch)
    print(f"enrich tick: {n} likes inserted")
    return 0


def cmd_score_bots(args: argparse.Namespace) -> int:
    settings = load_settings()
    init_db(settings.db_path)
    mix = mix_by_id(args.mix)
    with connect(settings.db_path) as conn:
        summary = score_mix_listeners(conn, mix.mix_id)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_comment_heatmap(args: argparse.Namespace) -> int:
    settings = load_settings()
    init_db(settings.db_path)
    mix = mix_by_id(args.mix)
    gt = args.gt
    measures = args.measure_times
    if measures is None and args.set_id:
        candidate = Path(f"data/analysis/{args.set_id}_measure_times.json")
        if candidate.is_file():
            measures = candidate
    result = run_heatmap_analysis(
        settings.db_path,
        mix.mix_id,
        gt_yaml=gt,
        measure_times_json=measures,
        bin_width_s=args.bin_width,
        out_path=args.out,
    )
    match result:
        case Ok(payload):
            print(json.dumps({k: payload[k] for k in payload if k not in ("bin_counts", "bin_centers_s")}, indent=2))
            if args.out:
                print(f"wrote {args.out}")
            return 0
        case Err(msg):
            print(json.dumps({"error": msg}, indent=2))
            return 1


def cmd_cluster(args: argparse.Namespace) -> int:
    settings = load_settings()
    init_db(settings.db_path)
    mix = mix_by_id(args.mix)
    with connect(settings.db_path) as conn:
        migrate_db(conn)
        summary = cluster_mix(
            conn,
            mix.mix_id,
            exclude_bots=not args.include_bots,
            min_tracks=args.min_tracks,
            max_users=args.max_users,
            n_clusters=args.clusters,
        )
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_prior_mert(args: argparse.Namespace) -> int:
    settings = load_settings()
    init_db(settings.db_path)
    mix = mix_by_id(args.mix)
    with connect(settings.db_path) as conn:
        migrate_db(conn)
        summary = run_prior_pipeline(
            conn,
            mix.mix_id,
            settings.data_dir / "prior_cache" / mix.mix_id,
            max_tracks=args.max_tracks,
            max_users=args.max_users,
            device=args.device,
        )
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_run_analysis(args: argparse.Namespace) -> int:
    settings = load_settings()
    init_db(settings.db_path)
    mix = mix_by_id(args.mix)
    summary = run_analysis_pipeline(
        settings,
        mix.mix_id,
        set_id=mix.set_id,
        gt_yaml=args.gt,
        exclude_bots=not args.include_bots,
        max_tracks=args.max_tracks,
        max_users=args.max_users,
        cluster_users=args.max_users_cluster,
        device=args.device,
        out_dir=args.out_dir,
        with_mert=args.with_mert,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_enrich_playlists(args: argparse.Namespace) -> int:
    settings = load_settings()
    init_db(settings.db_path)
    mix = mix_by_id(args.mix)
    n = enrich_playlists_batch(settings, mix, batch_size=args.batch)
    print(f"enrich-playlists tick: {n} playlists inserted")
    return 0


def cmd_import_archive(args: argparse.Namespace) -> int:
    settings = load_settings()
    init_db(settings.db_path)
    mix = mix_by_id(args.mix)
    stats = import_archive_dir(settings, mix, args.archive_dir)
    print(json.dumps(stats, indent=2))
    return 0


def _collect_done(settings, mix_id: str) -> bool:
    from personalization.persistence import load_checkpoint

    with connect(settings.db_path) as conn:
        ck = load_checkpoint(conn, mix_id, "collect")
    return bool(ck.get("likers_done") and ck.get("reposters_done") and ck.get("comments_done"))


def _enrich_likes_done(settings, mix_id: str) -> bool:
    from personalization.persistence import listener_sc_ids, load_checkpoint

    with connect(settings.db_path) as conn:
        ck = load_checkpoint(conn, mix_id, "enrich_likes")
        completed = set(ck.get("completed_sc_user_ids") or [])
        in_progress = ck.get("in_progress") or {}
        if in_progress:
            return False
        return len(completed) >= len(listener_sc_ids(conn, mix_id))


def cmd_loop(args: argparse.Namespace) -> int:
    settings = load_settings()
    init_db(settings.db_path)
    mixes = load_mixes() if args.all_mixes else (mix_by_id(args.mix),)

    def one_pass() -> None:
        for mix in mixes:
            if not _collect_done(settings, mix.mix_id):
                collect_tick(settings, mix)
            # Enrich runs in parallel with likers collect — don't wait for collect done.
            enrich_batch(settings, mix, batch_size=args.batch)
            if _enrich_likes_done(settings, mix.mix_id):
                enrich_playlists_batch(settings, mix, batch_size=args.batch)

    if args.once:
        one_pass()
        return 0

    logger.info("loop started sleep=%ds mixes=%d", args.sleep, len(mixes))
    while True:
        try:
            one_pass()
        except Exception:
            logger.exception("loop pass failed")
        time.sleep(args.sleep)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Taste prior — SoundCloud scrape (pi-worker)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init-db", help="Create taste_warehouse.db").set_defaults(func=cmd_init_db)
    sub.add_parser("status", help="Row counts").set_defaults(func=cmd_status)

    p_col = sub.add_parser("collect", help="One collect tick")
    p_col.add_argument("--mix", required=True)
    p_col.set_defaults(func=cmd_collect)

    p_en = sub.add_parser("enrich", help="One enrich batch")
    p_en.add_argument("--mix", required=True)
    p_en.add_argument("--batch", type=int, default=20)
    p_en.set_defaults(func=cmd_enrich)

    p_imp = sub.add_parser("import-archive", help="Import legacy dj-listener-pipeline JSONL")
    p_imp.add_argument("--mix", required=True)
    p_imp.add_argument("--archive-dir", type=Path, required=True)
    p_imp.set_defaults(func=cmd_import_archive)

    p_bot = sub.add_parser("score-bots", help="Heuristic bot scoring for cohort listeners")
    p_bot.add_argument("--mix", required=True)
    p_bot.set_defaults(func=cmd_score_bots)

    p_hm = sub.add_parser("comment-heatmap", help="Comment density vs GT section starts")
    p_hm.add_argument("--mix", required=True)
    p_hm.add_argument("--gt", type=Path, default=None, help="Ground truth YAML (optional)")
    p_hm.add_argument("--set-id", default=None, help="Set id for measure_times lookup")
    p_hm.add_argument("--measure-times", type=Path, default=None)
    p_hm.add_argument("--bin-width", type=float, default=30.0)
    p_hm.add_argument("--out", type=Path, default=None)
    p_hm.set_defaults(func=cmd_comment_heatmap)

    p_cl = sub.add_parser("cluster", help="Cluster clean cohort by track overlap")
    p_cl.add_argument("--mix", required=True)
    p_cl.add_argument("--include-bots", action="store_true")
    p_cl.add_argument("--min-tracks", type=int, default=15)
    p_cl.add_argument("--max-users", type=int, default=5000)
    p_cl.add_argument("--clusters", type=int, default=12)
    p_cl.set_defaults(func=cmd_cluster)

    p_pm = sub.add_parser("prior-mert", help="Cache SC track MERT + build user priors")
    p_pm.add_argument("--mix", required=True)
    p_pm.add_argument("--max-tracks", type=int, default=150)
    p_pm.add_argument("--max-users", type=int, default=100)
    p_pm.add_argument("--device", default="auto")
    p_pm.set_defaults(func=cmd_prior_mert)

    p_ra = sub.add_parser("run-analysis", help="Bots + cluster + comment heatmap (MERT optional)")
    p_ra.add_argument("--mix", required=True)
    p_ra.add_argument("--gt", type=Path, default=None)
    p_ra.add_argument("--include-bots", action="store_true")
    p_ra.add_argument("--max-tracks", type=int, default=150)
    p_ra.add_argument("--max-users", type=int, default=100)
    p_ra.add_argument("--max-users-cluster", type=int, default=5000)
    p_ra.add_argument("--device", default="auto")
    p_ra.add_argument("--with-mert", action="store_true", help="Also build MERT user priors (post-pretrain)")
    p_ra.add_argument("--out-dir", type=Path, default=Path("data/analysis"))
    p_ra.set_defaults(func=cmd_run_analysis)

    p_epl = sub.add_parser("enrich-playlists", help="One playlist enrich batch")
    p_epl.add_argument("--mix", required=True)
    p_epl.add_argument("--batch", type=int, default=20)
    p_epl.set_defaults(func=cmd_enrich_playlists)

    p_loop = sub.add_parser("loop", help="Collect then enrich loop")
    p_loop.add_argument("--mix", default="1fsnxchk")
    p_loop.add_argument("--all-mixes", action="store_true")
    p_loop.add_argument("--batch", type=int, default=15)
    p_loop.add_argument("--sleep", type=int, default=300)
    p_loop.add_argument("--once", action="store_true")
    p_loop.set_defaults(func=cmd_loop)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
