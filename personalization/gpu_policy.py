"""GPU placement policy: batch deep-learning inference runs on Vast, not the Mac.

Mac is fine for I/O (SoundCloud scrape, DB), CPU work, and tiny smoke tests
(`--limit` + `--allow-local-gpu`). Anything that loads MERT/transformers on
GPU for a real batch → `scripts/vast_taste_embed.sh`.
"""
from __future__ import annotations

import platform
import sys


def enforce_vast_for_gpu(device: str, *, allow_local_gpu: bool, limit: int) -> None:
    """Exit unless this is an explicit local smoke test or we're on a CUDA host (Vast)."""
    if allow_local_gpu and limit > 0 and limit <= 5:
        return
    if platform.system() != "Darwin":
        return                                      # Linux box (Vast / pi) — OK
    if device == "cpu":
        return
    print(
        "GPU deep-learning jobs run on Vast only (not Mac MPS).\n"
        "  Launch:  scripts/vast_taste_embed.sh\n"
        "  Smoke:   --limit 3 --allow-local-gpu --device mps\n",
        file=sys.stderr,
    )
    raise SystemExit(2)
