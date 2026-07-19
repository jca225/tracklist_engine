"""Cut mix/source windows to playable mp3 clips for the review player."""

from __future__ import annotations

import subprocess
from pathlib import Path


def cut_mp3(src: Path, start_s: float, end_s: float, dst: Path) -> bool:
    """Extract [start_s, end_s) from ``src`` into ``dst`` as loudness-normalized mp3."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.05, end_s - start_s)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{max(0.0, start_s):.3f}",
        "-t",
        f"{dur:.3f}",
        "-i",
        str(src),
        "-af",
        "loudnorm=I=-16:TP=-1.5,afade=t=in:d=0.03,afade=t=out:d=0.05",
        "-ac",
        "2",
        "-ar",
        "44100",
        "-b:a",
        "192k",
        str(dst),
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0
