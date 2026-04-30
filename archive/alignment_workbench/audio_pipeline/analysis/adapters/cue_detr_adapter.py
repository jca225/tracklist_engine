"""cue-detr adapter: wraps the in-repo `cue-detr/cue_points.py` script.

Because `cue-detr/` uses a hyphen (not an importable package name), we
insert its parent on sys.path and import `cue_points` directly. The
underlying module exposes a directory-scanning function; we reuse its
inner per-file logic by calling it on a 1-file temp directory.
"""
from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from ...result import Err, Ok, Result
from ..errors import CueError

_CUE_DETR_DIR: Path = Path(__file__).resolve().parents[3] / "cue-detr"


@dataclass(frozen=True)
class CueDetrHandle:
    checkpoint: str
    sensitivity: float
    radius: int


def load(
    checkpoint: str = "disco-eth/cue-detr",
    sensitivity: float = 0.9,
    radius: int = 16,
) -> Result[CueDetrHandle, CueError]:
    if not _CUE_DETR_DIR.exists():
        return Err(CueError(kind="model_load", detail=f"cue-detr dir missing: {_CUE_DETR_DIR}"))
    if str(_CUE_DETR_DIR) not in sys.path:
        sys.path.insert(0, str(_CUE_DETR_DIR))
    try:
        import cue_points  # noqa: F401 — import-probe only
    except ImportError as e:
        return Err(CueError(kind="model_load", detail=f"cue_points import: {e}"))
    return Ok(CueDetrHandle(checkpoint=checkpoint, sensitivity=sensitivity, radius=radius))


def predict(h: CueDetrHandle, audio_path: Path) -> Result[tuple[float, ...], CueError]:
    """Run cue-detr on a single file. Returns cue timestamps in seconds.

    The upstream API operates on directories; we stage the file into a
    temp dir per-call. Model weights are cached by transformers, so the
    repeated load is cheap after the first call.
    """
    if str(_CUE_DETR_DIR) not in sys.path:
        sys.path.insert(0, str(_CUE_DETR_DIR))
    try:
        import cue_points  # type: ignore
    except ImportError as e:
        return Err(CueError(kind="model_load", detail=f"cue_points import: {e}"))

    src = Path(audio_path)
    if not src.exists():
        return Err(CueError(kind="inference", detail=f"audio missing: {src}"))

    with TemporaryDirectory() as td:
        dest_dir = Path(td)
        staged = dest_dir / (src.stem + src.suffix.lower())
        try:
            shutil.copy2(src, staged)
            result = cue_points.predict_cue_points_for_dir(
                str(dest_dir),
                checkpoint=h.checkpoint,
                sensitivity=h.sensitivity,
                radius=h.radius,
                print_points=False,
                write_output=False,
            )
        except (FileNotFoundError, RuntimeError, OSError, ValueError) as e:
            return Err(CueError(kind="inference", detail=str(e)))

    cues = result.get(staged.name, [])
    return Ok(tuple(float(t) for t in cues))
