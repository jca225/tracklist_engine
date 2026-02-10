from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from downloader.commands import run_command

_CUE_DETR_PREDICTOR_CACHE: dict[str, Callable[..., dict[str, list[float]]]] = {}


def _load_cue_detr_predictor(cue_detr_dir: Path) -> Callable[..., dict[str, list[float]]]:
    cue_detr_dir = cue_detr_dir.expanduser().resolve()
    cache_key = str(cue_detr_dir)
    cached = _CUE_DETR_PREDICTOR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    cue_points_script = cue_detr_dir / "cue_points.py"
    if not cue_points_script.exists():
        raise FileNotFoundError(f"cue-detr entrypoint not found: {cue_points_script}")

    module_name = f"cue_detr_api_{hashlib.sha1(cache_key.encode('utf-8')).hexdigest()[:10]}"
    spec = importlib.util.spec_from_file_location(module_name, str(cue_points_script))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load cue-detr module from {cue_points_script}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    predictor = getattr(module, "predict_cue_points_for_dir", None)
    if not callable(predictor):
        raise RuntimeError(
            "cue-detr API function `predict_cue_points_for_dir` is missing. "
            "Update cue-detr/cue_points.py to expose that function."
        )

    _CUE_DETR_PREDICTOR_CACHE[cache_key] = predictor
    return predictor


def get_cue_points_with_cue_detr(
    audio_file: Path,
    *,
    cue_detr_dir: Path,
    checkpoint: str = "disco-eth/cue-detr",
    sensitivity: float = 0.9,
    radius: int = 16,
) -> list[float]:
    """Predict cue points using the local cue-detr repository."""
    if not cue_detr_dir.exists():
        raise FileNotFoundError(f"cue-detr directory not found: {cue_detr_dir}")

    predictor = _load_cue_detr_predictor(cue_detr_dir)

    with tempfile.TemporaryDirectory(prefix="cue_detr_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        tmp_track = tmpdir / f"{audio_file.stem}.mp3"

        if audio_file.suffix.lower() == ".mp3":
            try:
                os.link(audio_file, tmp_track)
            except OSError:
                shutil.copy2(audio_file, tmp_track)
        else:
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(audio_file),
                "-vn",
                "-ar",
                "44100",
                "-ac",
                "2",
                "-c:a",
                "libmp3lame",
                "-q:a",
                "2",
                str(tmp_track),
            ]
            ok, err = run_command(cmd)
            if not ok:
                raise RuntimeError(f"Failed to prepare mp3 for cue-detr: {err}")

        cue_points = predictor(
            tmpdir,
            checkpoint=checkpoint,
            sensitivity=sensitivity,
            radius=radius,
            print_points=False,
            write_output=False,
        )
        if not isinstance(cue_points, dict):
            raise RuntimeError("cue-detr API returned an unexpected result")

        parsed = cue_points.get(tmp_track.name)
        if parsed is None:
            parsed = cue_points.get(tmp_track.stem)
        if isinstance(parsed, list):
            return [float(v) for v in parsed]

    return []
