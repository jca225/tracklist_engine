"""One-shot taste prior analysis pipeline."""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from core.result import Err, Ok
from workspaces.taste_prior.bot_heuristics import score_mix_listeners
from workspaces.taste_prior.comment_heatmap import run_heatmap_analysis
from workspaces.taste_prior.config import TasteSettings
from workspaces.taste_prior.persistence import connect, migrate_db
from workspaces.taste_prior.prior_mert import run_prior_pipeline
from workspaces.taste_prior.taste_cluster import cluster_mix

logger = logging.getLogger(__name__)


def run_analysis_pipeline(
    settings: TasteSettings,
    mix_id: str,
    *,
    set_id: str | None = None,
    gt_yaml: Path | None = None,
    exclude_bots: bool = True,
    max_tracks: int = 150,
    max_users: int = 100,
    cluster_users: int = 5000,
    device: str = "auto",
    out_dir: Path | None = None,
    with_mert: bool = False,
) -> dict[str, object]:
    conn = connect(settings.db_path)
    migrate_db(conn)

    summary: dict[str, object] = {"mix_id": mix_id}
    summary["bots"] = score_mix_listeners(conn, mix_id)
    summary["clusters"] = cluster_mix(
        conn, mix_id, exclude_bots=exclude_bots, max_users=cluster_users
    )
    if with_mert:
        summary["mert"] = run_prior_pipeline(
            conn,
            mix_id,
            settings.data_dir / "prior_cache" / mix_id,
            max_tracks=max_tracks,
            max_users=max_users,
            device=device,
        )

    sid = set_id or mix_id
    measure_path = Path(f"data/analysis/{sid}_measure_times.json")
    heat_out = (out_dir or Path("data/analysis")) / f"{mix_id}_comment_heatmap.json"
    hm = run_heatmap_analysis(
        settings.db_path,
        mix_id,
        gt_yaml=gt_yaml,
        measure_times_json=measure_path if measure_path.is_file() else None,
        out_path=heat_out,
    )
    match hm:
        case Ok(payload):
            summary["comment_heatmap"] = payload
        case Err(msg):
            summary["comment_heatmap"] = {"error": msg}

    conn.close()

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{mix_id}_pipeline_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
    return summary
