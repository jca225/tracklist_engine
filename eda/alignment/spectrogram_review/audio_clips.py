"""Cut mix/source windows to playable clips for the review player.

Uses AAC/m4a when the destination suffix is ``.m4a`` (Safari-friendly), else mp3.

Critical ffmpeg footgun: ``afade=t=out:d=0.05`` without ``st=`` starts the fade
at t=0 and leaves the rest of the file silent — that is why the player had
"working" Web Audio with no audible music.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def cut_clip(src: Path, start_s: float, end_s: float, dst: Path) -> bool:
    """Extract [start_s, end_s) from ``src`` into ``dst`` (aac/m4a or mp3)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.05, end_s - start_s)
    fade_out_st = max(0.0, dur - 0.05)
    ext = dst.suffix.lower()
    if ext == ".m4a":
        codec = ["-c:a", "aac", "-b:a", "192k"]
    else:
        codec = ["-c:a", "libmp3lame", "-b:a", "192k"]
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
        f"afade=t=in:d=0.03,afade=t=out:st={fade_out_st:.3f}:d=0.05",
        "-ac",
        "2",
        "-ar",
        "44100",
        *codec,
        str(dst),
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def cut_mp3(src: Path, start_s: float, end_s: float, dst: Path) -> bool:
    """Back-compat alias."""
    return cut_clip(src, start_s, end_s, dst)
