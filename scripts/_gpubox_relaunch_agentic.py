"""One-shot: fix aligning symlinks + relaunch agentic-both on existing gpubox run."""

from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.home() / "workspace/gpubox/src"))

from gpubox.api import Box  # noqa: E402

CODE = "/workspace/tracklist_engine"
PY = "/opt/conda/envs/align/bin/python"
RUN = "align-agentic-both"
EXPECT_ID = 45324301

JOB = f"""#!/bin/bash
set -euo pipefail
cd {CODE}
export PYTHONPATH={CODE}
export HF_HOME=/workspace/hf
export TRANSFORMERS_CACHE=/workspace/hf
export PYTHONUNBUFFERED=1
mkdir -p /workspace/hf workspaces/alignment_prototype/out

echo "=== BB11 structure_cue agentic ==="
{PY} -u - <<'PY'
from pathlib import Path
from workspaces.alignment_prototype.drivers.base import SetContext
from workspaces.alignment_prototype.drivers.agentic import AgenticDriver
ctx = SetContext.for_set("2nvzlh2k", preflight=True)
base = Path("workspaces/alignment_prototype/out/2nvzlh2k_predicted_timeline.json")
if not base.is_file():
    raise SystemExit("no BB11 base timeline")
print("base", base)
out = AgenticDriver(base, live=True).align_set(ctx)
print("BB11_DONE", out)
PY

echo "=== BB10 pool agentic (unlabeled) ==="
{PY} -u -m workspaces.alignment_prototype.drivers.flywheel \\
  --set-id w1mgcjt \\
  --reuse-base workspaces/alignment_prototype/out/w1mgcjt_predicted_timeline.json

echo "=== materialize BB10 pseudo-GT ==="
{PY} -u -m workspaces.alignment_prototype.trajectory.materialize_pseudo_gt \\
  --timeline workspaces/alignment_prototype/out/w1mgcjt_agentic_timeline.json \\
  --out workspaces/alignment_prototype/out/w1mgcjt_pseudo_gt.yaml \\
  --set-id w1mgcjt

echo "ALL_DONE"
date
"""


def main() -> int:
    box = Box.attach(RUN)
    if box is None or box.instance_id != EXPECT_ID:
        raise SystemExit(f"expected {RUN}@{EXPECT_ID}, got {box}")
    print(f"attached {box.instance_id}")

    rc, out, err = box.session.exec(
        "rm -rf /root/aligning; mkdir -p /root/aligning; "
        "ln -sfn /workspace/aligning/2nvzlh2k__Two /root/aligning/2nvzlh2k__Two; "
        "ln -sfn /workspace/aligning/w1mgcjt__Two /root/aligning/w1mgcjt__Two; "
        "ls -la /root/aligning",
        timeout=60,
    )
    print(out or err)
    if rc != 0:
        raise SystemExit(f"symlink failed rc={rc}")

    b64 = base64.b64encode(JOB.encode()).decode()
    box.session.exec(
        f"echo {b64} | base64 -d > /workspace/agentic_both.sh && "
        "chmod +x /workspace/agentic_both.sh",
        timeout=30,
    )
    box.session.exec(
        "tmux kill-session -t agentic-both 2>/dev/null || true", timeout=15
    )
    box.run_cmd(
        "bash /workspace/agentic_both.sh",
        name="agentic-both",
        log="/workspace/agentic_both.log",
    )
    box.heartbeat()
    time.sleep(8)
    running = box.is_running("agentic-both")
    log = box.session.logs("agentic-both", tail=80, log="/workspace/agentic_both.log")
    print(f"running={running}")
    print("--- log ---")
    print(log)
    if not running:
        raise SystemExit("agentic-both not running after launch")
    if "Traceback" in log and "BB11_DONE" not in log:
        raise SystemExit("agentic-both hit Traceback during startup")
    print(f"\nOK run={RUN} instance={box.instance_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
