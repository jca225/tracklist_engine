"""Generic retry-with-backoff over Result-returning calls.

Lives in core/ (substrate rule: no upward imports — it knows only Result). It is
*not* a place to merge error types across layers; it composes on the Result
algebra. Retry only what is genuinely transient (``retry_on``); deterministic
failures (bot-detection, missing JS runtime) must not be retried — they need the
preflight remedy, not another attempt.
"""

from __future__ import annotations

import time
from typing import Callable

from core.result import Ok, Result


def retry[T, E](
    fn: Callable[[], Result[T, E]],
    *,
    attempts: int = 3,
    base_delay_s: float = 0.5,
    retry_on: Callable[[E], bool] = lambda _e: True,
    sleep: Callable[[float], None] = time.sleep,
) -> Result[T, E]:
    """Call ``fn`` until it returns Ok or retries are exhausted.

    Backoff is exponential (base_delay_s, 2x, 4x, ...). A retry happens only when
    the call errs AND ``retry_on(err)`` is true; otherwise the Err is returned
    immediately. Returns the first Ok, or the last Err. ``sleep`` is injectable so
    tests don't actually wait. ``attempts`` is clamped to >= 1 (always one call).
    """
    delay = base_delay_s
    result: Result[T, E] = fn()
    for _ in range(1, max(1, attempts)):
        if isinstance(result, Ok) or not retry_on(result.error):
            return result
        sleep(delay)
        delay *= 2
        result = fn()
    return result
