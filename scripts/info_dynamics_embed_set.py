"""MERT-embed a set's mix + stems and run the info-dynamics significance test.

Prerequisites on pi-storage: ``set_analysis.measure_times_json``, mix audio,
``set_stems`` vocals/instrumental (or local ``_mac_scratch/set_stems/set/<id>/``).

    venvs/audio/bin/python scripts/info_dynamics_embed_set.py --set-id w1mgcjt
    venvs/audio/bin/python scripts/info_dynamics_embed_set.py --set-id 2nvzlh2k --skip-test
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ANALYSIS = REPO / "data/analysis"
PI_HOST = "pi-storage"
CANONICAL_DB = "/mnt/storage/data/db/music_database.db"
PYTHON = REPO / "venvs/audio/bin/python"


def ssh_sql(sql: str) -> str:
    r = subprocess.run(
        ["ssh", PI_HOST, f"sqlite3 -noheader {CANONICAL_DB} \"{sql}\""],
        capture_output=True,
        text=True,
        check=True,
    )
    return r.stdout.strip()


def pull(remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists() and local.stat().st_size > 0:
        return
    subprocess.check_call(["rsync", "-q", f"{PI_HOST}:{remote}", str(local)])


def embed(set_id: str, audio: Path, measure_json: Path, out: Path) -> None:
    subprocess.check_call([
        str(PYTHON), "-m", "eda.alignment.prepare_mix_artifact",
        "--set-id", set_id,
        "--audio", str(audio),
        "--measure-times-json", str(measure_json),
        "--out", str(out),
    ])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set-id", required=True)
    p.add_argument("--skip-test", action="store_true")
    args = p.parse_args(argv)
    sid = args.set_id

    boundaries = ANALYSIS / f"{sid}_tracklist_boundaries.json"
    if not boundaries.is_file():
        print(f"missing {boundaries} — run cache_tracklist_boundaries.py first", file=sys.stderr)
        return 1

    row = ssh_sql(
        f"SELECT sa.set_audio_id, sa.path FROM set_audio sa "
        f"JOIN set_analysis san ON san.set_audio_id=sa.set_audio_id "
        f"WHERE sa.set_id='{sid}' LIMIT 1"
    )
    if not row:
        print(f"{sid}: no set_analysis row yet (run beat grid first)", file=sys.stderr)
        return 1
    set_audio_id_s, mix_path = row.split("|", 1)
    set_audio_id = int(set_audio_id_s)

    measures_json = ANALYSIS / f"{sid}_measure_times.json"
    if not measures_json.is_file():
        raw = ssh_sql(
            f"SELECT measure_times_json FROM set_analysis "
            f"WHERE set_audio_id={set_audio_id}"
        )
        measures_json.write_text(raw if raw.startswith("[") else json.dumps(json.loads(raw)))

    scratch = REPO / "_mac_scratch" / "info_dynamics" / sid
    scratch.mkdir(parents=True, exist_ok=True)

    mix_local = scratch / f"mix{Path(mix_path).suffix}"
    pull(mix_path, mix_local)

    stem_root = REPO / "_mac_scratch" / "set_stems" / "set" / set_audio_id_s
    voc_remote = f"/mnt/storage/stems/set/{set_audio_id}/vocals.flac"
    inst_remote = f"/mnt/storage/stems/set/{set_audio_id}/instrumental.flac"
    voc_local = stem_root / "vocals.flac" if stem_root.is_dir() else scratch / "vocals.flac"
    inst_local = stem_root / "instrumental.flac" if stem_root.is_dir() else scratch / "instrumental.flac"
    if not voc_local.is_file():
        pull(voc_remote, voc_local)
    if not inst_local.is_file():
        pull(inst_remote, inst_local)

    streams = [
        (mix_local, ANALYSIS / f"{sid}_mix_mert.npz"),
        (voc_local, ANALYSIS / f"{sid}_mix_vocals_mert.npz"),
        (inst_local, ANALYSIS / f"{sid}_mix_instrumental_mert.npz"),
    ]
    for audio, out in streams:
        if not audio.is_file():
            print(f"[skip] {audio.name}: not found", file=sys.stderr)
            continue
        print(f"embedding {audio.name} -> {out.name}")
        embed(sid, audio, measures_json, out)

    if args.skip_test:
        return 0

    subprocess.check_call([
        str(PYTHON), "-m", "eda.alignment.info_dynamics.run_set", "--set-id", sid,
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
