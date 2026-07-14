"""One-slot input prefetch for the analysis driver loops (streaming_mir WS1).

While the GPU analyzes track N, a single background thread picks and pulls
track N+1 so its audio is already local when the main loop needs it. This is
the input-side mirror of vast_loop's single-slot persist thread.

Imported, not run (like rescue_common.py). Deliberately torch-free and
I/O-free: the three I/O actions are injected callables, so vast_loop wires
ssh/rsync in and tests wire lambdas in. Keep it that way — tests import this
module directly.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class PrefetchItem:
    """A track whose audio is already local, ready to analyze."""

    tid: int
    local_audio: Path
    asset: Any  # core.models.AudioAsset — typed Any to stay import-light
    pull_s: float


@dataclass(frozen=True)
class PrefetchFailure:
    """pull/hydrate raised for this tid; caller treats it like today's
    per-track subprocess failure (log + failed_tids.add + continue)."""

    tid: int
    detail: str


class PrefetchSlot:
    """Single-slot prefetch thread: start() spawns, take() joins + returns.

    take() returns:
      - PrefetchItem     — picked, pulled, hydrated; ready to analyze
      - PrefetchFailure  — picked, but pull/hydrate raised
      - None             — pick() found nothing (queue drained for the given
                           skip set; the caller decides whether that's final)

    The single-slot join discipline (take() before the next start()) is the
    same happens-before contract the persist thread uses, so the plain
    attribute write in _run is safe to read after join.
    """

    def __init__(
        self,
        pick: Callable[[frozenset[int]], tuple[int, str] | None],
        pull: Callable[[int, str], Path],
        hydrate: Callable[[int, Path], Any],
    ) -> None:
        self._pick = pick
        self._pull = pull
        self._hydrate = hydrate
        self._thread: threading.Thread | None = None
        self._result: PrefetchItem | PrefetchFailure | None = None

    @property
    def pending(self) -> bool:
        return self._thread is not None

    def start(self, skip_tids: frozenset[int]) -> None:
        assert self._thread is None, "single slot: take() before next start()"

        def _run() -> None:
            picked = self._pick(skip_tids)
            if picked is None:
                self._result = None
                return
            tid, remote_path = picked
            try:
                t0 = time.monotonic()
                local = self._pull(tid, remote_path)
                pull_s = time.monotonic() - t0
                asset = self._hydrate(tid, local)
            except Exception as exc:  # rsync/ssh CalledProcessError et al.
                self._result = PrefetchFailure(tid=tid, detail=str(exc))
                return
            self._result = PrefetchItem(
                tid=tid, local_audio=local, asset=asset, pull_s=pull_s
            )

        self._result = None
        self._thread = threading.Thread(target=_run, daemon=False)
        self._thread.start()

    def take(self) -> PrefetchItem | PrefetchFailure | None:
        assert self._thread is not None, "take() without start()"
        self._thread.join()
        self._thread = None
        result = self._result
        self._result = None
        return result
