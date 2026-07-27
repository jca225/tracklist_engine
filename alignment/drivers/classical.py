"""Driver A — classical / deterministic: the shipped `infer` -> `joint_ref_decode`
pipeline behind the driver interface.

No learning at decision time: MERT `predict_sequence` for identity, axis-routed
placement (fp for regular/instrumental, lyrics+HuBERT for acappella, all gated to
the anchored prior), then the piecewise-linear Viterbi (`looptrace`) for
ref_segments. This is the baseline the other two drivers must beat.

The two stages run as subprocesses (not in-process): infer loads MPS torch + a
MERT ensemble and SSHes pi-storage; joint_ref_decode forks a ProcessPoolExecutor
(MPS is not fork-safe). Separate processes keep those from colliding — the same
isolation `scripts/reinfer_driver.sh` relies on. Set `--reuse-base` in the race
runner to skip the expensive infer stage and start from an existing timeline.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .base import _REPO, SetContext, finalize


def _run(argv: list[str]) -> None:
    """Run a pipeline stage, streaming its output; raise on non-zero exit."""
    print(f"  $ {' '.join(argv)}", flush=True)
    proc = subprocess.run(argv, cwd=str(_REPO))
    if proc.returncode != 0:
        raise RuntimeError(f"stage failed (exit {proc.returncode}): {argv[:4]} …")


class ClassicalDriver:
    name = "classical"

    def __init__(
        self,
        *,
        decoder: str = "looptrace",
        band_s: float = 45.0,
        python: str | None = None,
    ) -> None:
        self.decoder = decoder
        self.band_s = band_s
        # the interpreter running us IS venvs/audio when invoked correctly
        self.python = python or sys.executable

    def align_set(self, ctx: SetContext) -> Path:
        py = self.python
        # infer writes the FIXED out/<set>_predicted_timeline.json (placement +
        # identity); all placement channels default-on (fp/stem/lyrics).
        infer_out = ctx.out_dir / f"{ctx.set_id}_predicted_timeline.json"
        _run(
            [
                py,
                "-m",
                "alignment.infer",
                "--set-id",
                ctx.set_id,
                "--train-yaml",
                str(ctx.source_yaml),
                "--band-s",
                str(self.band_s),
            ]
        )
        # joint_ref_decode adds ref_segments; write to the driver-named artifact
        # (never rewrite the shared infer output — agentic/ml read it as base).
        out = ctx.timeline_path(self.name)
        _run(
            [
                py,
                "-m",
                "alignment.joint_ref_decode",
                "--set-id",
                ctx.set_id,
                "--decoder",
                self.decoder,
                "--timeline",
                str(infer_out),
                "--out",
                str(out),
            ]
        )
        # already schema-valid (both stages write through the contract shape);
        # re-validate + return the canonical path.
        import json

        return finalize(json.loads(out.read_text()), out)
