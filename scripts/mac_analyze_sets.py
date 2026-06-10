"""One-shot: run beat_this + Demucs on full DJ-set mixes via Mac MPS.

Sibling of mac_analyze_loop.py but for set_audio rows instead of track_audio.
Pulls each set audio from pi-storage, runs analyze_set on MPS, rsyncs stems
back to pi-storage, writes set_analysis + set_stems via ssh sqlite3.

By default targets every set_audio row that has no set_analysis row yet.
Pass --set-audio-ids to scope.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Set analysis writes set_audio_id-keyed rows; canonical has FKs to set_audio
# already so no scratch-DB indirection needed. We still set
# TRACKLIST_DISABLE_FK because we're going to write rows via raw INSERTs.
os.environ["TRACKLIST_DISABLE_FK"] = "1"

from analysis.pipeline import load_analyzers
from analysis.set_analysis import analyze_set
from core.models import SetAudioAsset
from core.result import Err, Ok

PI_HOST = "pi-storage"
CANONICAL_DB = "/mnt/storage/data/db/music_database.db"
PI_STEMS_ROOT = "/mnt/storage/stems"

SCRATCH_DIR = REPO / "_mac_scratch"
LOCAL_SETS = SCRATCH_DIR / "sets"
LOCAL_SET_STEMS = SCRATCH_DIR / "set_stems"

DEVICE = "mps"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mac_analyze_sets")


def ssh_pi(sql: str) -> str:
    full = f'sqlite3 -separator "|" {CANONICAL_DB} "{sql}"'
    r = subprocess.run(["ssh", PI_HOST, full],
                       capture_output=True, text=True, check=True)
    return r.stdout.strip()


def fetch_pending_sets(only_ids: tuple[int, ...] | None) -> list[SetAudioAsset]:
    where = ""
    if only_ids:
        ids_csv = ",".join(str(i) for i in only_ids)
        where = f"AND sa.set_audio_id IN ({ids_csv})"
    sql = (
        "SELECT sa.set_audio_id, sa.set_id, sa.platform, sa.source_url, "
        "sa.path, COALESCE(sa.sha256,''), COALESCE(sa.duration_s,0), "
        "COALESCE(sa.sample_rate,0), COALESCE(sa.codec,''), "
        "COALESCE(sa.bitrate_kbps,0) "
        "FROM set_audio sa "
        "LEFT JOIN set_analysis san ON san.set_audio_id=sa.set_audio_id "
        f"WHERE san.set_audio_id IS NULL {where} "
        "ORDER BY sa.set_audio_id"
    )
    out = ssh_pi(sql)
    if not out:
        return []
    rows = []
    for line in out.splitlines():
        p = line.split("|")
        rows.append(SetAudioAsset(
            set_audio_id=int(p[0]),
            set_id=p[1],
            platform=p[2],
            source_url=p[3],
            path=p[4],
            sha256=p[5] or None,
            duration_s=float(p[6]) if p[6] else None,
            sample_rate=int(p[7]) if p[7] else None,
            codec=p[8] or None,
            bitrate_kbps=int(p[9]) if p[9] else None,
        ))
    return rows


def rsync_in(remote_path: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["rsync", "-q", f"{PI_HOST}:{remote_path}", str(local)])


def rsync_stems_out(local_dir: Path, set_audio_id: int) -> None:
    src = f"{local_dir}/"
    dst = f"{PI_HOST}:{PI_STEMS_ROOT}/set/{set_audio_id}/"
    subprocess.check_call(
        ["ssh", PI_HOST, f"mkdir -p {PI_STEMS_ROOT}/set/{set_audio_id}"]
    )
    subprocess.check_call(["rsync", "-aq", src, dst])


def _sql_lit(v):
    if v is None: return "NULL"
    if isinstance(v, (int, float)): return repr(v) if not isinstance(v, bool) else str(int(v))
    return "'" + str(v).replace("'", "''") + "'"


def push_set_rows(result, mix_local_path: Path) -> None:
    """Push set_analysis + set_stems to canonical DB on pi-storage.

    set_stems schema uses (set_audio_id, stem_name) as natural key and UPSERTs,
    so no PK-collision issue. set_analysis is keyed by set_audio_id.
    """
    import json
    sid = result.set_audio_id
    # Rewrite local stem paths -> canonical pi-storage paths.
    stems = result.stems.stems
    sql_lines = [".bail on", "BEGIN;"]
    sql_lines.append(f"DELETE FROM set_analysis WHERE set_audio_id={sid};")
    if stems:
        # Beat-grid-only runs (--skip-stems) must not clobber existing
        # set_stems rows written by another host.
        sql_lines.append(f"DELETE FROM set_stems WHERE set_audio_id={sid};")
    for stem in stems:
        canonical_path = str(stem.path).replace(
            str(LOCAL_SET_STEMS), f"{PI_STEMS_ROOT}/set",
        )
        sql_lines.append(
            "INSERT INTO set_stems (set_audio_id, stem_name, path, codec) "
            f"VALUES ({sid}, {_sql_lit(stem.stem_name)}, "
            f"{_sql_lit(canonical_path)}, {_sql_lit(stem.codec)});"
        )
    beats_json = json.dumps(list(result.beats.beat_times))
    downbeats_json = json.dumps(list(result.beats.downbeat_times))
    measures_json = json.dumps(list(result.beats.measure_times))
    versions_json = json.dumps(result.analyzer_versions)
    sql_lines.append(
        "INSERT INTO set_analysis (set_audio_id, beat_times_json, "
        "downbeat_times_json, measure_times_json, analyzer_versions_json) "
        f"VALUES ({sid}, {_sql_lit(beats_json)}, {_sql_lit(downbeats_json)}, "
        f"{_sql_lit(measures_json)}, {_sql_lit(versions_json)});"
    )
    sql_lines.append("COMMIT;")
    full = "\n".join(sql_lines)
    subprocess.run(["ssh", PI_HOST, f"sqlite3 {CANONICAL_DB}"],
                   input=full, text=True, check=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--set-audio-ids", type=str, default=None,
                   help="Comma-separated list to scope (default: all pending)")
    p.add_argument("--separator", choices=["demucs", "uvr", "roformer"], default="demucs",
                   help="Stem-separation backend (default: demucs).")
    p.add_argument("--skip-stems", action="store_true",
                   help="Beat grid only (set_analysis row, no set_stems) — "
                        "use when separation runs on another host.")
    args = p.parse_args()
    only_ids = None
    if args.set_audio_ids:
        only_ids = tuple(int(x) for x in args.set_audio_ids.split(","))

    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_SETS.mkdir(parents=True, exist_ok=True)
    LOCAL_SET_STEMS.mkdir(parents=True, exist_ok=True)

    log.info("loading analyzers (device=%s, separator=%s)…", DEVICE, args.separator)
    ar = load_analyzers(device=DEVICE, separator=args.separator)
    if not ar.is_ok():
        log.error("load_analyzers failed: %s", ar.error)
        return 1
    analyzers = ar.value
    log.info("analyzers ready")

    pending = fetch_pending_sets(only_ids)
    log.info("%d set_audio rows pending analysis", len(pending))
    if not pending:
        return 0

    for asset in pending:
        sid = asset.set_audio_id
        log.info("[%d] %s (%s, %.0fs)", sid, asset.set_id, asset.platform,
                 asset.duration_s or 0)

        # 1) Pull audio
        local_audio = LOCAL_SETS / f"{sid}.m4a"
        t0 = time.monotonic()
        log.info("[%d] pulling %s", sid, asset.path)
        rsync_in(asset.path, local_audio)
        log.info("[%d] pulled in %.0fs", sid, time.monotonic() - t0)

        # 2) Analyze (analyze_set takes a SetAudioAsset; rewrite path to local)
        local_asset = SetAudioAsset(
            set_audio_id=sid, set_id=asset.set_id, platform=asset.platform,
            source_url=asset.source_url, path=str(local_audio),
            sha256=asset.sha256, duration_s=asset.duration_s,
            sample_rate=asset.sample_rate, codec=asset.codec,
            bitrate_kbps=asset.bitrate_kbps,
        )
        t1 = time.monotonic()
        result = analyze_set(analyzers, local_asset, LOCAL_SET_STEMS,
                             skip_stems=args.skip_stems)
        elapsed = time.monotonic() - t1
        match result:
            case Ok(r):
                log.info("[%d] analyzed in %.0fs (bpm=%.1f, %d beats, %d measures)",
                         sid, elapsed, r.beats.bpm or 0,
                         len(r.beats.beat_times), len(r.beats.measure_times))
            case Err(e):
                log.error("[%d] analyze_set failed: %s — %s", sid, e.kind, e.detail)
                continue

        # 3) Push stems to pi-storage
        local_set_dir = LOCAL_SET_STEMS / "set" / str(sid)
        if local_set_dir.exists():
            log.info("[%d] pushing stems to pi-storage", sid)
            rsync_stems_out(local_set_dir, sid)

        # 4) Push DB rows
        log.info("[%d] pushing DB rows to canonical", sid)
        push_set_rows(r, local_audio)
        log.info("[%d] DONE", sid)

        # Cleanup local audio (stems already pushed)
        if local_audio.exists():
            local_audio.unlink()

    log.info("queue drained")
    return 0


if __name__ == "__main__":
    sys.exit(main())
