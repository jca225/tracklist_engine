from __future__ import annotations

import subprocess
from pathlib import Path


def run_command(cmd: list[str], *, cwd: Path | None = None) -> tuple[bool, str]:
    try:
        subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        return True, ""
    except FileNotFoundError as exc:
        return False, str(exc)
    except subprocess.CalledProcessError as exc:
        return False, exc.stderr.strip()
