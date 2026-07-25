"""GPU placement policy: batch deep-learning inference runs on a GPU box, not the Mac.

Mac is fine for I/O (SoundCloud scrape, DB), CPU work, and tiny smoke tests
(`--limit` + `--allow-local-gpu`). Anything that loads MERT/transformers on
GPU for a real batch runs on a `gpubox`-rented GPU box (the only sanctioned way
to rent GPUs here; gpubox rents from Vast.ai under the hood).
"""

from __future__ import annotations

import platform
import sys


def enforce_gpu_host(device: str, *, allow_local_gpu: bool, limit: int) -> None:
    """Exit unless this is an explicit local smoke test or we're on a CUDA host
    (a gpubox-rented GPU box)."""
    if allow_local_gpu and limit > 0 and limit <= 5:
        return
    if platform.system() != "Darwin":
        return  # Linux box (gpubox GPU / pi) — OK
    if device == "cpu":
        return
    print(
        "GPU deep-learning jobs run on a gpubox-rented GPU box only (not Mac MPS).\n"
        "  Launch:  rent a box with gpubox, then run the batch on it\n"
        "  Smoke:   --limit 3 --allow-local-gpu --device mps\n",
        file=sys.stderr,
    )
    raise SystemExit(2)
