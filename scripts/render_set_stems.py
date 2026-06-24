"""Render a continuous acappella (vocals) + instrumental for a full DJ-set mix.

Why this exists separately from `mac_analyze_sets.py`: that script runs
beat_this on the whole mix first, and beat_this is a transformer with no
internal chunking — a 62-min waveform OOMs MPS (~26 GiB) before separation
even starts. This script skips the beat grid entirely and does only what the
"full-set acappella/instrumental" goal needs: stem separation.

Separation models (audio-separator MDX/VR + demucs) *do* segment internally,
but to keep an unattended overnight run safe on MPS we slice the mix into
fixed-length chunks, separate each, and concatenate. The loop is **resumable**:
a chunk whose part files already exist is skipped, so a crash/restart resumes.

    venvs/audio/bin/python scripts/render_set_stems.py --set-audio-id 5 --separator uvr

Output: _mac_scratch/set_stems/set/<id>/{vocals,instrumental}.flac, rsynced to
pi-storage /mnt/storage/stems/set/<id>/ and written into set_stems (unless
--no-push).
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

os.environ.setdefault("TRACKLIST_DISABLE_FK", "1")

from analysis.adapters import demucs_adapter, roformer_chain_adapter, uvr_chain_adapter  # noqa: E402
from analysis.roformer_config import RoformerChainConfig  # noqa: E402
from analysis.separation_config import ChainConfig  # noqa: E402

PI_HOST = "pi-storage"
CANONICAL_DB = "/mnt/storage/data/db/music_database.db"
PI_STEMS_ROOT = "/mnt/storage/stems"

SCRATCH = REPO / "_mac_scratch"
LOCAL_SETS = SCRATCH / "sets"
RENDER_ROOT = SCRATCH / "set_render"
OUT_ROOT = SCRATCH / "set_stems" / "set"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("render_set_stems")


def ssh_pi_sql(sql: str) -> str:
    r = subprocess.run(["ssh", PI_HOST, f'sqlite3 -separator "|" {CANONICAL_DB} "{sql}"'],
                       capture_output=True, text=True, check=True)
    return r.stdout.strip()


def fetch_set_audio(set_audio_id: int) -> tuple[str, str, float]:
    out = ssh_pi_sql(
        f"SELECT set_id, path, COALESCE(duration_s,0) FROM set_audio "
        f"WHERE set_audio_id={set_audio_id}")
    if not out:
        log.error("no set_audio row for id=%d", set_audio_id)
        sys.exit(1)
    set_id, path, dur = out.split("|")
    return set_id, path, float(dur)


def pull_mix(remote_path: str, set_audio_id: int) -> Path:
    LOCAL_SETS.mkdir(parents=True, exist_ok=True)
    local = LOCAL_SETS / f"{set_audio_id}{Path(remote_path).suffix}"
    if local.exists() and local.stat().st_size > 0:
        log.info("mix already local: %s", local)
        return local
    log.info("pulling mix %s", remote_path)
    subprocess.check_call(["rsync", "-q", f"{PI_HOST}:{remote_path}", str(local)])
    return local


def split_chunks(mix: Path, chunks_dir: Path, chunk_sec: int) -> list[Path]:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(chunks_dir.glob("chunk_*.wav"))
    if existing:
        log.info("reusing %d existing chunks", len(existing))
        return existing
    log.info("splitting %s into %ds chunks", mix.name, chunk_sec)
    subprocess.check_call([
        "ffmpeg", "-v", "error", "-y", "-i", str(mix),
        "-f", "segment", "-segment_time", str(chunk_sec),
        "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le",
        str(chunks_dir / "chunk_%04d.wav"),
    ])
    chunks = sorted(chunks_dir.glob("chunk_*.wav"))
    log.info("created %d chunks", len(chunks))
    return chunks


def separate_chunk(separator, handle, chunk: Path, tmp_dir: Path):
    """Return (vocals_path, instrumental_path) for one chunk."""
    if separator == "uvr":
        r = uvr_chain_adapter.separate(handle, chunk, tmp_dir, track_audio_id=0)
    elif separator == "roformer":
        r = roformer_chain_adapter.separate(handle, chunk, tmp_dir, track_audio_id=0)
    else:
        r = demucs_adapter.separate(handle, chunk, tmp_dir, 0)
    if not r.is_ok():
        raise RuntimeError(f"{separator} separate failed: {r.error.kind} — {r.error.detail}")
    leaf = tmp_dir / "0"
    return leaf / "vocals.flac", leaf / "instrumental.flac"


def concat_flac(parts: list[Path], dest: Path) -> None:
    """Concatenate same-format flac parts into one flac (re-encode for safety)."""
    listfile = dest.parent / f"{dest.stem}_concat.txt"
    listfile.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    subprocess.check_call([
        "ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
        "-i", str(listfile), "-c:a", "flac", str(dest),
    ])
    listfile.unlink(missing_ok=True)


def push_to_canonical(set_audio_id: int, out_dir: Path) -> None:
    dst = f"{PI_HOST}:{PI_STEMS_ROOT}/set/{set_audio_id}/"
    subprocess.check_call(["ssh", PI_HOST, f"mkdir -p {PI_STEMS_ROOT}/set/{set_audio_id}"])
    subprocess.check_call(["rsync", "-aq", f"{out_dir}/", dst])
    sql = "\n".join([
        ".bail on", "BEGIN;",
        f"DELETE FROM set_stems WHERE set_audio_id={set_audio_id};",
        f"INSERT INTO set_stems (set_audio_id, stem_name, path, codec) VALUES "
        f"({set_audio_id}, 'vocals', "
        f"'{PI_STEMS_ROOT}/set/{set_audio_id}/vocals.flac', 'flac');",
        f"INSERT INTO set_stems (set_audio_id, stem_name, path, codec) VALUES "
        f"({set_audio_id}, 'instrumental', "
        f"'{PI_STEMS_ROOT}/set/{set_audio_id}/instrumental.flac', 'flac');",
        "COMMIT;",
    ])
    subprocess.run(["ssh", PI_HOST, f"sqlite3 {CANONICAL_DB}"],
                   input=sql, text=True, check=True)
    log.info("wrote 2 set_stems rows + rsynced stems for set_audio_id=%d", set_audio_id)


def fetch_set_audio_local(db: Path, set_audio_id: int) -> tuple[str, str, float]:
    out = subprocess.run(
        [
            "sqlite3", "-separator", "|", str(db),
            f"SELECT set_id, path, COALESCE(duration_s,0) FROM set_audio "
            f"WHERE set_audio_id={set_audio_id}",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not out:
        log.error("no set_audio row for id=%d", set_audio_id)
        sys.exit(1)
    set_id, path, dur = out.split("|")
    return set_id, path, float(dur)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set-audio-id", type=int, required=True)
    ap.add_argument("--separator", choices=["uvr", "demucs", "roformer"], default="uvr")
    ap.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    ap.add_argument("--chunk-sec", type=int, default=360)
    ap.add_argument("--mix", type=Path, default=None,
                    help="local mix file (default: pull from pi-storage)")
    ap.add_argument("--db", type=Path, default=None,
                    help="local SQLite for set_audio lookup when --mix is set")
    ap.add_argument("--no-push", action="store_true",
                    help="render locally only; don't write to canonical DB/stems")
    args = ap.parse_args()
    sid = args.set_audio_id

    if args.mix is not None:
        if args.db is not None:
            set_id, _remote_path, dur = fetch_set_audio_local(args.db, sid)
        else:
            set_id, dur = f"id={sid}", 0.0
        mix = args.mix
    else:
        set_id, remote_path, dur = fetch_set_audio(sid)
        mix = pull_mix(remote_path, sid)
    log.info("set_audio_id=%d set_id=%s duration=%.0fs", sid, set_id, dur)

    work = RENDER_ROOT / str(sid)
    chunks_dir = work / "chunks"
    parts_dir = work / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    chunks = split_chunks(mix, chunks_dir, args.chunk_sec)

    log.info("loading %s analyzers (device=%s)…", args.separator, args.device)
    if args.separator == "uvr":
        lr = uvr_chain_adapter.load(ChainConfig.default(), device=args.device)
    elif args.separator == "roformer":
        lr = roformer_chain_adapter.load(RoformerChainConfig.default(), device=args.device)
    else:
        lr = demucs_adapter.load(device=args.device)
    if not lr.is_ok():
        log.error("load failed: %s — %s", lr.error.kind, lr.error.detail)
        return 1
    handle = lr.value
    log.info("analyzers ready")

    voc_parts: list[Path] = []
    inst_parts: list[Path] = []
    for i, chunk in enumerate(chunks):
        voc_part = parts_dir / f"vocals_{i:04d}.flac"
        inst_part = parts_dir / f"instrumental_{i:04d}.flac"
        if voc_part.exists() and inst_part.exists():
            log.info("[%d/%d] %s — already done, skipping", i + 1, len(chunks), chunk.name)
        else:
            t0 = time.monotonic()
            tmp = work / "tmp" / f"{i:04d}"
            v, ins = separate_chunk(args.separator, handle, chunk, tmp)
            v.replace(voc_part)
            ins.replace(inst_part)
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
            log.info("[%d/%d] %s separated in %.0fs", i + 1, len(chunks),
                     chunk.name, time.monotonic() - t0)
        voc_parts.append(voc_part)
        inst_parts.append(inst_part)

    out_dir = OUT_ROOT / str(sid)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("concatenating %d parts → vocals.flac + instrumental.flac", len(voc_parts))
    concat_flac(voc_parts, out_dir / "vocals.flac")
    concat_flac(inst_parts, out_dir / "instrumental.flac")
    log.info("rendered: %s", out_dir)

    if args.no_push:
        log.info("--no-push set; leaving canonical DB untouched")
    else:
        push_to_canonical(sid, out_dir)

    log.info("DONE set_audio_id=%d", sid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
