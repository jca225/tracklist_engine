#!/usr/bin/env python3
"""Bedtime watchdog for align-bb10-flywheel.

Polls until ALL_DONE (pull + destroy) or DEADLINE_S from start (destroy anyway).
Runs standalone so billing stops even if the Cursor agent session sleeps.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.home() / "workspace/gpubox/src"))

from gpubox.api import Box  # noqa: E402

RUN = "align-bb10-flywheel"
EXPECT_ID = 45344153
DEADLINE_S = 90 * 60  # 90 minutes from watchdog start
POLL_S = 10 * 60  # check every 10 min
LOCAL_OUT = Path(
    "/Users/johnnycabrahams/Desktop/tracklist_engine/workspaces/"
    "alignment_prototype/out/vast_agentic"
)
LOG = Path("/tmp/gpubox_bb10_watchdog.log")


def _log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def _pull_and_destroy(box: Box) -> None:
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    for name in (
        "w1mgcjt_agentic_timeline.json",
        "w1mgcjt_pseudo_gt.yaml",
    ):
        try:
            box.session.pull(
                f"/workspace/tracklist_engine/workspaces/alignment_prototype/out/{name}",
                str(LOCAL_OUT / name),
                timeout=300,
                delete=False,
            )
            _log(f"pulled {name}")
        except Exception as e:  # noqa: BLE001
            _log(f"pull miss {name}: {type(e).__name__}: {e}")
    try:
        box.session.pull(
            "/workspace/agentic_both.log",
            str(LOCAL_OUT / "agentic_both_bb10_hubert.log"),
            timeout=60,
            delete=False,
        )
    except Exception as e:  # noqa: BLE001
        _log(f"pull log miss: {e}")
    ok = box.destroy()
    _log(f"destroyed={ok}")


def main() -> int:
    t0 = time.time()
    deadline = t0 + DEADLINE_S
    _log(
        f"WATCHDOG start run={RUN} expect={EXPECT_ID} "
        f"deadline_in={DEADLINE_S // 60}m poll={POLL_S // 60}m"
    )
    while True:
        now = time.time()
        left = deadline - now
        box = Box.attach(RUN)
        if box is None:
            _log("box gone (already destroyed) — exit")
            print(
                'AGENT_LOOP_TICK_bb10_watchdog {"prompt":"BB10 watchdog: box already gone"}'
            )
            return 0
        if box.instance_id != EXPECT_ID:
            _log(f"REFUSE unexpected id={box.instance_id} — exit without destroy")
            return 2

        log = ""
        try:
            log = (
                box.session.logs(
                    "agentic-both", tail=40, log="/workspace/agentic_both.log"
                )
                or ""
            )
        except Exception as e:  # noqa: BLE001
            _log(f"logs failed: {type(e).__name__}: {e}")

        running = False
        try:
            running = box.is_running("agentic-both")
        except Exception as e:  # noqa: BLE001
            _log(f"is_running failed: {type(e).__name__}: {e}")

        _log(
            f"poll running={running} left={left / 60:.0f}m all_done={'ALL_DONE' in log}"
        )

        if "ALL_DONE" in log:
            _log("ALL_DONE — pull + destroy")
            _pull_and_destroy(box)
            print(
                'AGENT_LOOP_TICK_bb10_watchdog {"prompt":"BB10 watchdog: ALL_DONE pulled and destroyed"}'
            )
            return 0

        if left <= 0:
            _log("DEADLINE — destroy without waiting (bedtime cost cap)")
            try:
                # best-effort pull of partial log
                LOCAL_OUT.mkdir(parents=True, exist_ok=True)
                box.session.pull(
                    "/workspace/agentic_both.log",
                    str(LOCAL_OUT / "agentic_both_bb10_hubert_deadline.log"),
                    timeout=60,
                    delete=False,
                )
            except Exception:  # noqa: BLE001
                pass
            ok = box.destroy()
            _log(f"deadline destroyed={ok}")
            print(
                'AGENT_LOOP_TICK_bb10_watchdog {"prompt":"BB10 watchdog: DEADLINE destroyed box"}'
            )
            return 1

        # sleep until next poll or deadline, whichever sooner
        time.sleep(min(POLL_S, max(30, left)))


if __name__ == "__main__":
    raise SystemExit(main())
