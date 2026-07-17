"""Shared hardening primitives for the analysis/backfill driver loops
(`vast_loop`, `mac_analyze_loop`, `set_mert_backfill_loop`).

Imported, never run — a sibling of `rescue_common.py`. Consolidates the
subprocess/SSH hardening constants and the failure-accounting logic so the same
bug classes can't be fixed in one loop and left latent in the others:

- **A2 (mojibake)** / **B1 (hang)** — `SSH_OPTS` + timeout constants: bound every
  ssh/rsync with a wall-clock timeout, keepalive-probe the connection, and pin
  UTF-8 at the Mac/pi locale boundary. Call sites pass these into `subprocess`.
- **C1 (escalation)** — `exit_status`: a drained loop with dominant failures must
  return nonzero + ERROR, never a cheerful INFO "drained" that reads as success.
- **C2 (transient vs poison)** — `register_transient`: bounded retry-with-backoff
  for transient failures (rsync/ssh timeout, truncated pull, DB lock), distinct
  from poison pills (decode errors) that are quarantined on first sight.
- **C3 (honest counting)** — `Counts`: thread-safe tally where `done` is bumped
  only after the canonical push confirms, so the headline can't over-report.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path

# --- A2 / B1: subprocess hardening --------------------------------------------
# ssh keepalive so a dead connection (e.g. a broken Tailscale SOCKS pipe) is
# detected in ~45s instead of hanging forever; BatchMode so a missing key fails
# fast instead of blocking on a password prompt.
SSH_OPTS: list[str] = [
    "-o",
    "ConnectTimeout=15",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=3",
    "-o",
    "BatchMode=yes",
]
SSH_QUERY_TIMEOUT = 120  # wall-clock for a sqlite3-over-ssh query / mkdir
SSH_PUSH_TIMEOUT = 300  # wall-clock for the canonical row push
RSYNC_TIMEOUT = 600  # wall-clock backstop for a single file transfer
RSYNC_IO_TIMEOUT = 300  # rsync --timeout: abort if no I/O flows for this long

# Pass to subprocess text calls that carry DB text (paths) across the SSH
# boundary, so a non-ASCII byte can't latin-1-mangle between Mac(UTF-8) and
# pi(Latin-1). surrogateescape keeps any already-corrupt byte round-trippable.
TEXT_KW = {"text": True, "encoding": "utf-8", "errors": "surrogateescape"}

# --- C2: transient retry policy -----------------------------------------------
MAX_TRANSIENT_ATTEMPTS = 3


def exit_status(
    n_done: int,
    n_failed: int,
    n_persist_failed: int = 0,
    fail_ratio_limit: float = 0.5,
) -> tuple[int, int, str]:
    """Decide a drained loop's outcome so a failing run can't read as success (C1).

    Returns ``(exit_code, log_level, message)``. Escalate to nonzero + ERROR when
    nothing succeeded, when failures exceed ``fail_ratio_limit``, or when
    handed-off tracks never persisted.
    """
    total = n_done + n_failed
    parts = [f"analyzed {n_done}", f"failed {n_failed}"]
    if n_persist_failed:
        parts.append(f"persist-failed {n_persist_failed}")
    msg = "queue drained — " + ", ".join(parts)
    if n_failed and n_done == 0:
        return 1, logging.ERROR, msg + " — ALL work failed"
    if total and n_failed / total > fail_ratio_limit:
        return (
            1,
            logging.ERROR,
            msg
            + f" — failure ratio {n_failed / total:.0%} exceeds {fail_ratio_limit:.0%}",
        )
    if n_persist_failed:
        return (
            1,
            logging.ERROR,
            msg + " — some tracks failed to persist (re-run to retry)",
        )
    return 0, logging.INFO, msg


class Counts:
    """Thread-safe honest tally (C3). ``done`` is incremented by the background
    tail only after the canonical push confirms — not at hand-off — so the
    headline can't over-report tracks that later failed to persist."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.done = 0
        self.failed = 0
        self.persist_failed = 0

    def _inc(self, attr: str) -> None:
        with self._lock:
            setattr(self, attr, getattr(self, attr) + 1)

    def done_inc(self) -> None:
        self._inc("done")

    def failed_inc(self) -> None:
        self._inc("failed")

    def persist_failed_inc(self) -> None:
        self._inc("persist_failed")

    def snapshot(self) -> tuple[int, int, int]:
        with self._lock:
            return self.done, self.failed, self.persist_failed


def register_transient(
    tid: int,
    detail: str,
    attempts: dict[int, int],
    failed_tids: set[int],
    counts: Counts,
    log: logging.Logger,
) -> bool:
    """Bounded retry for TRANSIENT failures — rsync/ssh timeout, truncated pull
    (sha mismatch), DB lock (bug C2). Distinct from poison pills (decode errors),
    which are quarantined on first sight. Returns True if the tid was quarantined
    (attempts exhausted → counted as a final failure), False if the caller should
    back off and let it be re-picked.

    NOTE: ``failed_tids`` and ``attempts`` are in-memory, so a genuine poison pill
    is still re-attempted across process restarts. The persistent dead-letter that
    would break the cross-restart cycle needs a canonical-DB migration and is
    deferred to a follow-up (tracked in the audit spec).
    """
    attempts[tid] = attempts.get(tid, 0) + 1
    n = attempts[tid]
    if n >= MAX_TRANSIENT_ATTEMPTS:
        failed_tids.add(tid)
        counts.failed_inc()
        log.error(
            "[%d] transient failure (attempt %d/%d) — quarantining: %s",
            tid,
            n,
            MAX_TRANSIENT_ATTEMPTS,
            detail,
        )
        return True
    log.warning(
        "[%d] transient failure (attempt %d/%d) — will retry: %s",
        tid,
        n,
        MAX_TRANSIENT_ATTEMPTS,
        detail,
    )
    return False


def sha256_file(path: Path, _chunk: int = 1 << 20) -> str:
    """Streamed sha256 of a local file (for B2 post-pull integrity checks)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


def transient_backoff_seconds(attempts: int) -> float:
    """Exponential backoff (capped) for the Nth transient retry of a tid."""
    return float(min(2**attempts, 30))
